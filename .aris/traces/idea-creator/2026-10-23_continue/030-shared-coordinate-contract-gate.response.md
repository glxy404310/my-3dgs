**same-family / provisional：主体无逻辑阻塞，但需消除一处关键实施歧义。**

在[合同](/D:/Documents/my%203dgs/research/experiments/shared-coordinate-residual-v0/CONTRACT.md)的“拟合后原生验证”补明：

> 参数域检查及原生重渲线性化验证，适用于每个 fit cell 中全部实际拟合的训练图和评价图 nuisance q。任何训练图 q 验证失败，同样记整体 INCONCLUSIVE；不得只验证 heldout q。

原因是共享系数依赖训练图 nuisance 的分解；训练 nuisance 超出有效线性域时，评价图验证通过也不能补证。此前约192次额外前向只覆盖评价侧；三个含共享参数的 pose 模型还产生最多144个训练侧拟合解，完整计数应相应记录，300秒硬预算不变。

dy/dx、C2W右乘、双重crossfit、四cell场拼接、noise定义和结论边界均一致。未运行实验或修改文件。
