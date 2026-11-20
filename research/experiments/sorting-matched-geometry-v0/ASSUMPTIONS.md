# 实施前假设账本

2026-10-23。继续#9强控制，不训练，不改变原已停止方案。原容量检查冻结旧投影/支持域，因此这次必须为每个量化结果Q重新取得原生投影、conic、SH及tile支持。

1. 精度主项为Morton21位空间排序后每4096点局部bbox、PLEDGE式tau=.25px；另有同式global、local tau=.125敏感性，以及global uniform8/12/16/20/24/32。不得按GT选择配置。
2. 位置编码以实际float32原生输入为准。每块/轴选最小整数b，使半量化步长加float32解码舍入上界不超过epsilon；b>24或不可安全表示时float32逃逸。uniform32为原值。Morton仅为分块，不能免费隐去索引：本首版将完整uint32原索引permutation计入local文件，明确不是优化成熟codec。
3. CPU codec采用little-endian固定header、每块count+六个float64端点+三个uint8位宽，逐轴bit-plane存储，每平面ceil(count/8)字节；32位轴直接float32。所有文件必须解码回原索引顺序且逐元素等于所评价Q，计全部header/permutation/padding/逃逸字节。不用名义bit数冒充实际文件。
4. PLEDGE域使用原生z近裁.2、八相机实际979x546的fmax和最大FOV半角，epsilon=tau*.2/[sqrt(3)*(fmax*sec(phi)+tau)]。该定理只约束中心在声明视锥内的点；画外中心但足迹入画的实际位移、近裁变化另报，不假称定理覆盖它们。C/D始终共用同一Q，实际位移逐点完全相同。
5. 首次fresh复核指出近似F32 fallback不能证明排序，因此废止该实现前候选。深度C必须来自原生已初始化depth，平局按original ID；通用lexsort在base必须逐项重建原tile列表。为覆盖原视野外但量化后新增支持的点，拟额外做同world_view_transform、原xyz的1x1宽FOV深度探针（FoVx/FoVy=pi-1e-4，仅改变投影和画布，绝不改view matrix），使用同一原生preprocess。其与原画布共同有效点的depth必须逐bit相同。所有Q新候选都必须能从这个原生探针取得depth；若含原近裁面后且无法取得的点或探针不覆盖，记工具FAILED，不用近似或删点代替。D使用Q原生有序tile列表，所有Q的D对nativeRGB<=2e-5。
6. 原生缓存入口沿用已验证ray-alpha helper，需记source/binary hash，避免再次读取仅备用而未初始化的internal_radii。只读取返回radii>0项。GPU用于冻结前向，CPU用于重放，零backward/optimizer。

本账本在实现依赖这些选择前写定；最终合同仍须完成一次数学/工具可行性核对后锁定，且在读取新效果前保存。

7. 位移强控制在原/Q候选支持并集检查实际逐轴max位移<=tau（仅允许FP32投影数值误差2e-4px），包括画外中心。若不满足，整个对应强控制记INCONCLUSIVE_DISPLACEMENT_CONTROL_NOT_MET，完整像素结果仍保存，不把它当方法失败，也不删点/调tau。近裁变化数量及无有效原投影点单独记录，后者不允许通过该强控制。主C/D容量与GT关原样不变。
