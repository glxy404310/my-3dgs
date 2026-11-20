已完成独立核验，现存数据与后处理足以支持合同规定的排序项容量关。

- 全6144行RGB派生指标误差为0；27行汇总及门槛完全一致，全部输入与脚本哈希通过。
- 独立读取原PLY和相机，重放64个固定中点ray、192个量化输出，最大RGB差 **2.22×10⁻¹⁶**；贡献计数差为0。
- 全缓存unique ID深度最大相对误差 **2.50×10⁻⁷**，低于2×10⁻⁵。
- 8/12/16 bit超阈值比例为 **88.67% / 45.75% / 6.30%**，各8/8视角通过。8与12 bit主关均通过。
- 未发现渲染算法偏离合同；确认最终CSV字段不齐导致序列化失败。原 `FAILED_TOOL_METHOD_UNKNOWN` 与记录的exit1保持不变。源码内部失败返回2，报告已区分内外层退出记录。

仅新增：

- [独立分析脚本](</D:/Documents/my 3dgs/scripts/experiments/analyze_sorting_quantization_capacity.py>)
- [独立分析报告](</D:/Documents/my 3dgs/research/experiments/sorting-quantization-capacity-v0/run/INDEPENDENT_ANALYSIS.json>)

最终核验CPU耗时2.18秒。证据仅支持固定支持域的排序反事实容量；不支持完整位置量化、实际压缩收益、速度、照片改善或创新性。
