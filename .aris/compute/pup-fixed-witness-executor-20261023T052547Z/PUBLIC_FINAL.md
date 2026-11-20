两个文档调用均原样、各执行一次；SH0 完整检查通过后才启动 SH3。

| case | actual outer / worker exit | actual host wall | tool wall / chunk | decision |
|---|---:|---:|---|---|
| SH0 | 0 / 0 | 9.7664205 s | 9.8663137 s / fa56c0 | SYNTHETIC_SH0_FISHER_WITNESS_PASS |
| SH3 | 0 / 0 | 9.9129102 s | 9.9948977 s / 779c5b | SYNTHETIC_SH3_FISHER_WITNESS_PASS |

两次均在初始非 PTY exec_command 调用内结束，因此未签发 ongoing session_id；完整 tool 原始返回已保存。两例独立 120 s / 1 GiB allocated GPU / 4 GiB process-tree RSS 预算均通过。

| case / G | FD relative Frobenius | actual symmetry maxabs | actual-vs-expected maxabs | permutation Fisher relative |
|---|---:|---:|---:|---:|
| SH0 / 0 | 0.00011698789219866518 | 0 | 0.000017861930801499337 | — |
| SH0 / 1 | 0.00006051842750469515 | 0 | 0.0000051195092147793275 | — |
| SH3 / 0 | 0.00012084929188607607 | 0 | 0.00001601641472248616 | 0.0000002612479011701437 |
| SH3 / 1 | 0.0000537807060820774 | 0 | 0.000009478681132657485 | 0.00000010351694275590614 |

SH3 RGB permutation maxabs = 0，permuted Fisher symmetry 均为 0。SH0 private forward/backward = 26/1；SH3 = 28/2；ordinary renderer 调用均为 0。

完整 RESULTS、WATCHDOG、MANIFEST、EXTERNAL_RUN 及 CONFIG/WORKER.log 已检查。SH0 41 个输出 manifest 条目及大小全部匹配，50 个输入 before/after 全一致，25 张 raw RGB 齐全。SH3 为 45 个输出、56 个输入、26 张 raw RGB，同样无差异；其 SH0 证据保护哈希保持一致。两例参数、pool 后参数、相机、文件、tracked source、source verification、finite、nonzero、call-count gate 均通过；SH3 原值/置换值恢复门槛通过。

GPU allocated peak：SH0 8,545,280 bytes；SH3 8,548,352 bytes。process-tree sampled peak RSS：SH0 1,371,734,016 bytes；SH3 1,416,491,008 bytes。supervisor wall：SH0 7.8646721389999925 s；SH3 7.7603953710000155 s。worker kernel maxrss：SH0 1,335,996,416 bytes；SH3 1,381,806,080 bytes。

具体 doc divergence：执行命令与对应文档块逐字符相同（仅去除文档块末尾换行），case、epsilon、阈值、环境、预算及调用计划均未改动；未发现执行/数值合同偏离。WSL 启动时有 localhost/NAT 代理相关警告，工具显示有转码字符，原样保留于外层日志；worker 没有异常。trace 目录名只作标识，时间依据两份 EXTERNAL_RUN.json 的 started_UTC/finished_UTC 字段。

GPU preflight exit 0：RTX 5070，1700 MiB / 12227 MiB，No running processes found。每例输出目录及外层文件在其调用前均不存在。

证据：[SH0 RESULTS](<D:/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/witness-fixed-v1/RESULTS.json>)、[SH3 RESULTS](<D:/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/witness-fixed-sh3-v1/RESULTS.json>)。完整任务 prompt、两条原样 invocation、tool 返回、preflight 与 manifest 复核在 [执行 trace](<D:/Documents/my 3dgs/.aris/compute/pup-fixed-witness-executor-20261023T052547Z/TRACE_INDEX.json>)。

两份结果仍为 environment_ready=false；这里只完成固定 synthetic 工具见证。未训练、未构造 Scene、未加载真实 Truck/teacher/GT、未运行 full219。未创建 root 认证 JSON。