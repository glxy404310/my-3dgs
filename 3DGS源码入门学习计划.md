# 3DGS 源码入门学习计划

适用对象：已经理解 3DGS 基本原理，但 Python / PyTorch / 科研代码阅读能力还比较薄弱的初学者。

目标不是系统成为 Python 工程师，也不是刷算法题，而是能逐步看懂 Inria / GraphDeCo 3DGS 官方源码的数据流、训练流和 Gaussian 增删逻辑。

建议周期：4 周  
建议节奏：每天 1.5-2.5 小时，每周至少留 1 天复盘  
学习方式：问题驱动 + 小脚本练习 + 回到 3DGS 源码定位

---

## 总体目标

4 周后应能做到：

1. 看懂 `train.py` 的主训练流程。
2. 理解 `Scene`、`Camera`、`GaussianModel`、`render` 之间的关系。
3. 知道图片、相机、点云、Gaussian 参数分别以什么形式进入训练。
4. 理解 `loss.backward()` 和 `optimizer.step()` 在 3DGS 中的作用。
5. 能解释 clone、split、prune、opacity reset 的基本代码逻辑。
6. 能写简单 Python / NumPy / PyTorch 脚本打印 shape、画图、分析训练结果。

---

# 第 1 周：Python 够用基础

本周目标：不被 Python 语法卡住，能看懂普通科研代码中的变量、循环、函数、类和字典。

不需要学：装饰器、元类、异步、复杂面向对象、LeetCode 算法技巧。

---

## Day 1：变量、基本类型、print 调试

推荐时长：1.5 小时

学习内容：

- `int`
- `float`
- `str`
- `bool`
- `list`
- `dict`
- `print`
- `type`
- `len`

学习方式：

1. 看 30-40 分钟 Python 基础教程。
2. 自己敲 30 分钟例子。
3. 最后用 20 分钟把变量和 3DGS 概念对应起来。

练习任务：

```python
gaussian = {
    "xyz": [1.0, 2.0, 3.0],
    "opacity": 0.1,
    "scale": [0.02, 0.02, 0.02],
    "color": [0.8, 0.3, 0.2]
}

print(gaussian["xyz"])
print(gaussian["opacity"])
print(type(gaussian))
```

对应 3DGS：

一个 Gaussian 在源码中不是字典，但概念上也包含：

```text
位置 xyz
透明度 opacity
尺度 scale
旋转 rotation
颜色 SH features
```

完成标准：

- 能创建一个字典表示 Gaussian。
- 能用 `print` 查看变量内容。
- 知道 `dict["key"]` 是怎么取值的。

---

## Day 2：if / for / 函数

推荐时长：2 小时

学习内容：

- `if`
- `else`
- `for`
- `range`
- `def`
- `return`

学习方式：

1. 学语法 40 分钟。
2. 写小函数 40 分钟。
3. 用 prune 的例子理解条件判断 30 分钟。

练习任务：

```python
def should_prune(opacity):
    return opacity < 0.005

opacities = [0.1, 0.003, 0.02, 0.0005]

for opacity in opacities:
    if should_prune(opacity):
        print(opacity, "should be pruned")
    else:
        print(opacity, "kept")
```

对应 3DGS：

源码中 prune 的核心判断类似：

```text
opacity 太低 -> 删除
屏幕半径太大 -> 删除
世界尺度太大 -> 删除
```

完成标准：

- 能写一个简单函数。
- 能用 `if` 判断是否删除 Gaussian。
- 能理解“条件筛选”是 prune 的基础。

---

## Day 3：list / dict 熟练使用

推荐时长：2 小时

学习内容：

- `append`
- `pop`
- `keys`
- `items`
- list 切片
- list 中存 dict

学习方式：

1. 先写普通 list 操作。
2. 再模拟训练相机列表 `viewpoint_stack`。

练习任务：

```python
cameras = ["cam_001", "cam_002", "cam_003"]

while len(cameras) > 0:
    cam = cameras.pop()
    print("training with", cam)
```

进阶练习：

```python
render_pkg = {
    "render": "predicted_image",
    "viewspace_points": "points_on_screen",
    "radii": "screen_radius"
}

print(render_pkg["render"])
```

对应 3DGS：

`train.py` 里有类似逻辑：

```text
viewpoint_stack 中存训练相机
每次随机取一个 Camera
render_pkg 是 render 返回的字典
```

完成标准：

- 理解 `pop` 是从列表里取元素。
- 理解 `render_pkg["render"]` 是从结果包里取渲染图。

---

## Day 4：class 入门

推荐时长：2 小时

学习内容：

- `class`
- `__init__`
- `self`
- 属性
- 方法

学习方式：

1. 先看一个最小 class 教程。
2. 自己写一个简化版 `Gaussian`。
3. 对照理解 `GaussianModel` 不是一个神秘东西，而是一个对象。

练习任务：

```python
class Gaussian:
    def __init__(self, xyz, opacity):
        self.xyz = xyz
        self.opacity = opacity

    def is_visible(self):
        return self.opacity > 0.005

g = Gaussian([1, 2, 3], 0.1)
print(g.xyz)
print(g.is_visible())
```

对应 3DGS：

源码中的 `GaussianModel` 可以先理解为：

```text
一个管理所有 Gaussian 参数的类
```

里面存：

```text
_xyz
_scaling
_rotation
_opacity
_features_dc
_features_rest
```

完成标准：

- 知道 `self.xxx` 是对象自己的属性。
- 看到 `gaussians.training_setup(opt)` 不慌。

---

## Day 5：import 和文件结构

推荐时长：1.5 小时

学习内容：

- `import`
- `from ... import ...`
- 模块
- 包
- 文件路径

学习方式：

1. 新建两个 `.py` 文件做简单导入练习。
2. 对照 3DGS 的源码导入语句。

练习任务：

文件 `my_math.py`：

```python
def square(x):
    return x * x
```

文件 `main.py`：

```python
from my_math import square

print(square(5))
```

对应 3DGS：

`train.py` 中会有类似：

```python
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
```

完成标准：

- 知道一个文件可以调用另一个文件里的函数或类。
- 知道 `train.py` 只是总入口，细节分布在其他模块。

---

## Day 6-7：第 1 周小项目

推荐时长：3-4 小时，可分两天

任务：写一个简化版 Gaussian prune 脚本。

要求：

1. 创建 10 个 Gaussian。
2. 每个 Gaussian 有 `xyz` 和 `opacity`。
3. 删除 opacity 小于 `0.005` 的 Gaussian。
4. 打印删除前后数量。

参考结构：

```python
gaussians = [
    {"xyz": [0, 0, 0], "opacity": 0.1},
    {"xyz": [1, 0, 0], "opacity": 0.003},
]

kept = []
for g in gaussians:
    if g["opacity"] >= 0.005:
        kept.append(g)

print("before:", len(gaussians))
print("after:", len(kept))
```

完成标准：

- 能独立写完。
- 能解释这和 3DGS 中 `prune_points` 的思想关系。

---

# 第 2 周：NumPy、图像和矩阵

本周目标：看到 shape、矩阵乘法、图片数组不慌。理解图片和点云在代码里本质上都是数组。

---

## Day 8：NumPy array 和 shape

推荐时长：2 小时

学习内容：

- `np.array`
- `np.zeros`
- `np.ones`
- `np.random`
- `.shape`
- `.reshape`

学习方式：

1. 用随机点云模拟 COLMAP 点。
2. 反复打印 shape。

练习任务：

```python
import numpy as np

points = np.random.randn(100, 3)
colors = np.random.rand(100, 3)

print(points.shape)
print(colors.shape)
```

对应 3DGS：

COLMAP 稀疏点云进入 `create_from_pcd` 后，会变成：

```text
points: [N, 3]
colors: [N, 3]
```

完成标准：

- 能说出 `[100, 3]` 表示 100 个三维点。
- 看到 `_xyz` 是 `[N, 3]` 不慌。

---

## Day 9：索引、切片、mask

推荐时长：2 小时

学习内容：

- `arr[0]`
- `arr[:, 0]`
- `arr[mask]`
- 布尔筛选

学习方式：

1. 用 opacity mask 模拟 prune。
2. 理解“删除”本质上是保留 mask 为 True 的项。

练习任务：

```python
import numpy as np

points = np.random.randn(5, 3)
opacity = np.array([0.1, 0.003, 0.02, 0.0001, 0.5])

mask = opacity >= 0.005
kept_points = points[mask]

print(points)
print(mask)
print(kept_points)
```

对应 3DGS：

`prune_points(mask)` 中的逻辑类似：

```text
valid_points_mask = ~mask
只保留没有被 prune 的 Gaussian
```

完成标准：

- 能理解 mask 是一串 True / False。
- 能理解源码中用 mask 删除 Gaussian 的思路。

---

## Day 10：矩阵乘法

推荐时长：2 小时

学习内容：

- `@`
- `.T`
- 对角矩阵
- 旋转矩阵的直觉
- 协方差构造

学习方式：

1. 先用单位矩阵和对角矩阵练习。
2. 再对应论文公式。

练习任务：

```python
import numpy as np

S = np.diag([2.0, 1.0, 0.5])
R = np.eye(3)

Sigma = R @ S @ S.T @ R.T
print(Sigma)
```

对应 3DGS：

\[
\Sigma = RSS^TR^T
\]

源码中通过 scale 和 rotation 构造 covariance。

完成标准：

- 知道 `@` 是矩阵乘法。
- 知道 `S @ S.T` 对应三个方向的方差。

---

## Day 11：图片读写和显示

推荐时长：2 小时

学习内容：

- `PIL.Image.open`
- `np.array(image)`
- `plt.imshow`
- 图片 shape
- `[H, W, 3]` 和 `[3, H, W]`

学习方式：

1. 找一张普通图片读入。
2. 打印 shape。
3. 显示并保存。

练习任务：

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("test.png").convert("RGB")
arr = np.array(img)

print(arr.shape)
plt.imshow(arr)
plt.show()
```

对应 3DGS：

数据加载大致是：

```text
图片文件 -> PIL image -> torch tensor -> Camera.original_image
```

完成标准：

- 知道普通图片数组通常是 `[H, W, 3]`。
- 知道 PyTorch 训练里常用 `[3, H, W]`。

---

## Day 12：画 loss 曲线

推荐时长：1.5 小时

学习内容：

- `matplotlib.pyplot.plot`
- `xlabel`
- `ylabel`
- `savefig`

练习任务：

```python
import matplotlib.pyplot as plt

losses = [1.0, 0.8, 0.6, 0.45, 0.3]

plt.plot(losses)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.savefig("loss_curve.png")
plt.show()
```

对应 3DGS：

训练时你以后会关心：

```text
loss 是否下降
PSNR 是否上升
Gaussian 数量是否增长
```

完成标准：

- 能画一条简单曲线。
- 知道曲线是分析训练过程的重要工具。

---

## Day 13-14：第 2 周小项目

推荐时长：4-5 小时，可分两天

任务：写一个 2D Gaussian 图像生成脚本。

要求：

1. 创建一张空白图片。
2. 放几个 2D Gaussian。
3. 每个 Gaussian 有中心、sigma、颜色、opacity。
4. 生成并保存图片。
5. 画出 alpha 分布。

对应 3DGS：

这相当于先在二维理解：

```text
Gaussian 覆盖像素
离中心近贡献大
多个 Gaussian 混合颜色
```

完成标准：

- 能生成一张带软斑点的图片。
- 能解释这和 3DGS splatting 的关系。

---

# 第 3 周：PyTorch 最小够用

本周目标：理解 `loss.backward()` 和 `optimizer.step()`。这是从“会看数组”到“能理解训练”的关键。

---

## Day 15：Tensor 基础

推荐时长：2 小时

学习内容：

- `torch.tensor`
- `.shape`
- `.float`
- `.cuda`
- `.cpu`
- `requires_grad`

学习方式：

1. 先在 CPU 上练习。
2. 如果环境有 CUDA，再尝试 `.cuda()`。

练习任务：

```python
import torch

x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
print(x)
print(x.shape)
print(x.requires_grad)
```

对应 3DGS：

`GaussianModel` 中这些都是可训练 Tensor：

```text
_xyz
_scaling
_rotation
_opacity
_features_dc
_features_rest
```

完成标准：

- 知道 Tensor 和 NumPy array 很像，但支持自动求导。
- 知道 `requires_grad=True` 表示这个量要被优化。

---

## Day 16：自动求导

推荐时长：2 小时

学习内容：

- `loss.backward`
- `.grad`
- 标量 loss

练习任务：

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
loss = (x - 5) ** 2
loss.backward()

print("loss:", loss.item())
print("grad:", x.grad)
```

理解重点：

```text
loss.backward() 不是更新参数
它只是计算梯度
```

对应 3DGS：

```text
图像 loss backward 后
Gaussian 的位置、尺度、颜色、opacity 会得到梯度
```

完成标准：

- 能解释 `.grad` 是什么。
- 能说出梯度方向和 loss 下降有关。

---

## Day 17：optimizer

推荐时长：2 小时

学习内容：

- `torch.optim.Adam`
- `optimizer.step`
- `optimizer.zero_grad`

练习任务：

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
opt = torch.optim.Adam([x], lr=0.1)

for i in range(100):
    loss = (x - 5) ** 2
    loss.backward()
    opt.step()
    opt.zero_grad()

print(x.item())
```

对应 3DGS：

训练循环本质上也是：

```text
render -> loss -> backward -> optimizer.step
```

完成标准：

- 知道 `step` 才是真正更新参数。
- 知道 `zero_grad` 是清空旧梯度。

---

## Day 18：多个参数组

推荐时长：2 小时

学习内容：

- optimizer 参数组
- 不同学习率
- 参数名和训练行为

练习任务：

```python
import torch

x = torch.tensor([0.0], requires_grad=True)
y = torch.tensor([0.0], requires_grad=True)

opt = torch.optim.Adam([
    {"params": [x], "lr": 0.1},
    {"params": [y], "lr": 0.01},
])

for i in range(100):
    loss = (x - 5) ** 2 + (y - 3) ** 2
    loss.backward()
    opt.step()
    opt.zero_grad()

print(x.item(), y.item())
```

对应 3DGS：

`training_setup` 里不同参数组有不同学习率：

```text
xyz
features_dc
features_rest
opacity
scaling
rotation
```

完成标准：

- 知道为什么位置、颜色、opacity 可以有不同学习率。

---

## Day 19：拟合颜色小例子

推荐时长：2 小时

任务：

随机初始化一个 RGB 颜色，用 PyTorch 优化到目标颜色。

练习任务：

```python
import torch

target = torch.tensor([0.8, 0.2, 0.1])
color = torch.rand(3, requires_grad=True)
opt = torch.optim.Adam([color], lr=0.05)

for i in range(200):
    loss = torch.mean((color - target) ** 2)
    loss.backward()
    opt.step()
    opt.zero_grad()

print(color.detach())
```

对应 3DGS：

这就是颜色参数被 loss 推着改变的最小模型。

完成标准：

- 能解释颜色为什么会慢慢接近 target。
- 能类比 3DGS 中 SH color 的优化。

---

## Day 20-21：第 3 周小项目

推荐时长：5-6 小时，可分两天

任务：极简 2D splatting 训练。

要求：

1. 准备一张简单目标图，例如白底红点。
2. 初始化几个 2D Gaussian。
3. 参数包括：
   - center
   - sigma
   - opacity
   - color
4. 用 PyTorch 渲染预测图。
5. 和目标图算 loss。
6. 用 optimizer 优化参数。

对应 3DGS：

这就是 3DGS 的二维缩小版：

```text
Gaussian 参数 -> 渲染图片 -> 图像 loss -> backward -> 更新 Gaussian
```

完成标准：

- 能跑通一个可训练的小例子。
- 能说出它和 `train.py` 主循环的对应关系。

---

# 第 4 周：回到 3DGS 源码

本周目标：不逐行死读，而是沿着数据流和训练流读核心组件。

---

## Day 22：只看 `train.py` 主循环

推荐时长：2 小时

只找这些关键词：

```text
Scene
GaussianModel
render
loss
backward
densify_and_prune
optimizer.step
```

学习方式：

1. 不看细节函数。
2. 只画流程图。
3. 每看到一个函数，只问：
   - 输入是什么？
   - 输出是什么？
   - 在训练流程中处于哪一步？

完成标准：

能画出：

```text
Camera + Gaussian -> render -> image -> loss -> backward -> step
```

---

## Day 23：看 `GaussianModel` 参数

推荐时长：2 小时

重点看：

```text
_xyz
_scaling
_rotation
_opacity
_features_dc
_features_rest
```

学习方式：

1. 把每个变量和论文符号对应起来。
2. 记住 `_` 开头的是优化器直接优化的原始参数。
3. 记住 `get_xxx` 是经过激活后的物理意义参数。

对应关系：

```text
_xyz              -> Gaussian 中心 mu
_scaling          -> 尺度 S 的原始参数
_rotation         -> 旋转四元数
_opacity          -> opacity 的原始参数
_features_dc      -> SH 颜色 DC 项
_features_rest    -> SH 颜色高阶项
```

完成标准：

- 能解释为什么 scale 要用 `exp`。
- 能解释为什么 opacity 要用 `sigmoid`。
- 能解释为什么 rotation 要 normalize。

---

## Day 24：看初始化 `create_from_pcd`

推荐时长：2 小时

重点问题：

```text
COLMAP 点云如何变成 Gaussian？
```

阅读目标：

```text
pcd.points -> _xyz
pcd.colors -> SH 颜色
点间距 -> scale
单位四元数 -> rotation
0.1 -> 初始 opacity
```

学习方式：

1. 先读函数名和变量名。
2. 不急着看每一行数学。
3. 用表格写下输入和输出。

完成标准：

能说出：

```text
3DGS 不是随机从零生成 Gaussian，
而是用 COLMAP 稀疏点云初始化 Gaussian 中心。
```

---

## Day 25：看训练设置 `training_setup`

推荐时长：2 小时

重点问题：

```text
哪些参数会被优化？
它们各自学习率是多少？
```

阅读目标：

看 optimizer 参数组：

```text
xyz
f_dc
f_rest
opacity
scaling
rotation
```

学习方式：

1. 只看参数组列表。
2. 观察每组名字和学习率。
3. 理解为什么不同参数学习率不同。

完成标准：

- 能说出 Gaussian 的所有主要参数都在 optimizer 里。
- 能解释 `features_rest` 学习率更小是为了让高阶 SH 不要太乱。

---

## Day 26：看 clone 和 split

推荐时长：2.5 小时

重点函数：

```text
densify_and_clone
densify_and_split
```

先记判断逻辑：

```text
梯度大 + Gaussian 小 -> clone
梯度大 + Gaussian 大 -> split
```

clone 做什么：

```text
复制原 Gaussian 的 xyz、features、opacity、scaling、rotation
追加到参数张量后面
```

split 做什么：

```text
围绕原 Gaussian 按其尺度采样新位置
新 Gaussian 继承颜色、opacity、rotation
scale 变小
删除原来的大 Gaussian
```

完成标准：

- 能解释 clone 和 split 的区别。
- 能说出 split 为什么会删除原来的大 Gaussian。

---

## Day 27：看 prune 和 opacity reset

推荐时长：2 小时

重点函数：

```text
prune_points
densify_and_prune
reset_opacity
```

prune 删除条件：

```text
opacity 太低
屏幕半径太大
世界尺度太大
```

opacity reset：

```text
把 opacity 限制回较低值
防止某些 Gaussian 太早变得不透明并挡住后方结构
```

完成标准：

- 能解释 prune 为什么不是简单删除列表，而要同步更新 optimizer 状态。
- 能解释 opacity reset 解决什么训练问题。

---

## Day 28：总结笔记

推荐时长：2-3 小时

任务：写一页自己的 3DGS 源码理解笔记。

必须回答：

1. 数据怎么进来？
2. 图片以什么形式参与训练？
3. 相机参数在哪里用？
4. Gaussian 参数有哪些？
5. 一次 forward 做什么？
6. loss 怎么算？
7. backward 更新什么？
8. clone / split / prune 什么时候发生？
9. 最终保存的模型是什么？

完成标准：

能用自己的话讲出：

```text
Scene 管数据，
GaussianModel 管参数，
render 管前向渲染，
loss.backward 管梯度，
optimizer.step 管参数更新，
densify_and_prune 管 Gaussian 数量变化。
```

---

# 每周复盘问题

每周末回答下面几个问题：

1. 本周我最清楚的概念是什么？
2. 本周最卡的点是什么？
3. 我有没有写出能运行的小脚本？
4. 我有没有把小脚本和 3DGS 源码对应起来？
5. 下一周最需要补的是什么？

---

# 不建议现在投入太多的内容

现阶段暂时不建议：

1. 刷 LeetCode。
2. 系统学习复杂 C++。
3. 深入 CUDA 内核实现。
4. 逐行阅读整个 3DGS 仓库。
5. 一上来尝试改大规模训练逻辑。

这些可以以后补。当前最重要的是建立：

```text
Python 基础 -> NumPy/图片 -> PyTorch 训练 -> 3DGS 主流程
```

---

# 推荐学习资料类型

不强制指定某个课程，但建议按类型选择：

1. Python：
   - 找“Python 快速入门”或“Python for beginners”。
   - 不要选太厚的完整工程课程。

2. NumPy：
   - 找“NumPy array shape indexing broadcasting”相关教程。
   - 重点是 shape、mask、矩阵乘法。

3. PyTorch：
   - 找“PyTorch autograd tutorial”和“PyTorch optimizer tutorial”。
   - 重点是 `requires_grad`、`backward`、`step`。

4. 3DGS：
   - 先看官方 `train.py` 和 `scene/gaussian_model.py`。
   - 暂时把 CUDA rasterizer 当黑盒。

---

# 最小每日学习模板

每天按这个格式做：

```text
20 分钟：看教程或源码讲解
40 分钟：自己敲一个小例子
20 分钟：打印变量 / shape / 中间结果
20 分钟：写三句话总结
```

三句话总结模板：

```text
今天我学了：
它在 3DGS 中对应：
我还没搞懂：
```

---

# 最终检验

完成 4 周后，请尝试不用看资料解释下面这段流程：

```text
COLMAP 点云和图片被 Scene 读入；
每个点初始化成一个 Gaussian；
训练时随机选一个 Camera；
render 根据 Camera 和 Gaussian 参数生成预测图；
预测图与真实图算 loss；
loss.backward 把图像误差传回 Gaussian 参数；
optimizer.step 更新位置、颜色、透明度、尺度和旋转；
densify_and_prune 根据屏幕空间梯度、opacity 和尺度增删 Gaussian；
训练结束后保存一组 Gaussian 参数作为场景表示。
```

如果你能把这段话讲清楚，就已经真正跨过 3DGS 源码入门门槛了。
