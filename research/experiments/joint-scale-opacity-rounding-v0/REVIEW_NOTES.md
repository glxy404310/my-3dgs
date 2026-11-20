# 执行前与独立CPU核验记录

2026-10-23。同模型家族辅助检查，不是正式experiment-audit，不作外部接收或跨家族独立性声明。

## 数学边界（先于主试验）

joint_alpha_math_check确认无截断、同中心、全平面alpha公式和合法opacity舍入，但提供cap/skip反例：S0=.3I、S=.6I、o0=.008，无截断解析选码1，带alpha<1/255跳过时实际最优为码2。因此合同没有把解析选码称为原生最优，预先增加97/193固定网格复核，不按结果重选码。

## 独立数值证人

joint_rounding_cpu_witness只用合成SPD数据另写check_joint_rounding_math.py，math-witness/RESULTS.json记录3个密网格积分与1000个合法码枚举。全部通过，积分相对误差最大7.15e-16；CPU0.074秒。归一化顺序反例分别产生码219和72，明确使用先求相机SSE总和、后按点归一化的目标。

## 主代码静读与运行前修复

joint_rounding_code_check原样最终回复：

> 静读未发现公式、广播、投影、样本选择、训练/验证隔离、三臂或合法码打包的阻断性错误；Linux CPU环境可启动。
>
> 两处判定缺口：
>
> - 333行把cap/skip教师能量为0的点直接排除；若教师峰值低于1/255、候选量化后非零，其新增误差会漏计，应至少记录此类点及其误差。
> - 336–341行只检查C对A/B的排序稳定性；若合同“臂排序相反”包含A与B，当前漏检A/B在97与193网格上的反转。
>
> 本地Windows直接运行会因`import resource`失败；Linux无此问题。数学证人RESULTS全部通过。

两处均在第一次真实pilot启动之前修复：cap/skip用固定未截断教师能量作分母，保留新增alpha误差；网格复核检查三臂全部pair排序。合同在START.json之前写定相同口径。实际使用WSL Linux，不在Windows解释器运行resource。

主pilot START.json/INPUTS.json/CONFIG.json保存实际运行合同与脚本hash；不依赖本笔记中的手写hash。run输出门槛只反映该局部连续alpha代理，不表示真实RGB或GT收益。

## 整模型补充合同

FULL_CONTRACT.md在完整编码和RGB结果之前明确：主门槛是学生对真实照片GT的平均MSE下降至少5%，两bit配置均须胜过A/B、各自最差4图不恶化。教师—学生MSE次指标不能替换主门槛。full-codec/ENCODING_RESULTS.json已经验证全部原公共字段精确保存及8192样本码精确复现；这不是渲染效果证据。
