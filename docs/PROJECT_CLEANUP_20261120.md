# 项目整理记录（2026-11-20）

本次范围为项目整理、Git LFS 配置及推送至 `glxy404310/my-3dgs`。未启动训练或修改实验结论。

## 上传范围

按用户选择：上传代码、结果报告和可视化；大型模型与原始数组仅保留本地。

| 内容 | 保存方式 |
|---|---|
| 源码、脚本、合同、Markdown、JSON/CSV 指标、日志及审计轨迹 | 普通 Git |
| PNG/JPEG 可视化、PDF、Word 文档 | Git LFS |
| `datasets/` 原始数据 | 本地，Git 忽略 |
| `.pth`、`.pt`、`.ckpt`、`.safetensors`、`.ply`、`.npz`、`.npy`、`.bin`、`.dsh8` | 本地，Git 忽略 |
| Python 编译缓存、虚拟环境、编辑器缓存 | 忽略；已有字节码缓存已清理 |

整理前共有 7,203 个文件，约 54.804 GiB（不含 `.git/`）。其中 2,119 个数据及模型/数组文件约 53.242 GiB 留在本地；其余约 1.562 GiB 包括报告、图片与源码，LFS 会对相同内容去重。

[LOCAL_ARTIFACTS.tsv](LOCAL_ARTIFACTS.tsv)记录仅本地保存文件的仓库相对路径与字节大小，属于本次整理时的目录清单，不是内容哈希校验表。远端克隆不包含这些文件；报告内提到的检查点和数组路径可能因此仅在原工作区存在。

原先已跟踪的 83 张 `datasets/desk/originals/` 图片从当前 Git 索引移除，磁盘文件全部保留。既有提交历史保持原样，因此历史提交仍包含曾提交过的图片；本次没有改写历史或强制推送。

## 整理内容

- 将首页改为项目状态、报告入口、目录说明与获取方式。
- 整理前首页原样保存到 `research/archive/2026-11-20/README_before_cleanup.md`。
- 将根目录的两个 `README_before_*` 历史快照归入同一归档目录。三个快照原文中的相对链接以仓库根目录为基准。
- 删除 49 个可重新生成的 Python 字节码缓存文件。
- 删除下列六个没有文件名引用、且已逐字节确认有相同保留副本的历史备份。
- 保留唯一内容、失败实验、原始指标、评审及授权记录；保留被清单引用的旧快照。

| 删除的重复文件 | 内容完全相同的保留文件 |
|---|---|
| `idea-stage/IDEA_REPORT_before_newtopic_20261022.md` | `idea-stage/IDEA_REPORT_INITIAL_20261022.md` |
| `idea-stage/IDEA_REPORT_before_artifact_pivot_20261023_032641.md` | `idea-stage/IDEA_REPORT_20261023_020614.md` |
| `idea-stage/continue-20261022/REPORT_before_artifact_pivot_20261023_032641.md` | `idea-stage/continue-20261022/REPORT_20261023_020614.md` |
| `idea-stage/continue-20261023/ROUND_REPORT_before_artifact_pivot_20261023_032641.md` | `idea-stage/continue-20261023/ROUND_REPORT_20261023_020614.md` |
| `idea-stage/IDEA_REPORT_before_artifact_stop_20261023_035031.md` | `idea-stage/IDEA_REPORT_20261023_032541.md` |
| `idea-stage/continue-20261022/CONTINUATION_STATE_before_artifact_pivot_20261023_032201.json` | `idea-stage/continue-20261022/CONTINUATION_STATE_20261023_022722.json` |

## 后续提交

```bash
git add -A
git commit -m "Update research reports"
git push origin main
```

本仓库已安装本地 LFS pre-push hook，正常 `git push` 会先上传所需的 LFS 对象。新增模型和数组会被忽略；如需以后另行发布，先明确改变上传范围并更新 `.gitignore`。
