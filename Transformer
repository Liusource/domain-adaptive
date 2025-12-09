# Transformer
1 Kolmogorov-Arnold Transformer[papers][https://github.com/Adamdad/kat?tab=readme-ov-file]  
 Transformer是深度学习的基石。传统上，这些模型依赖多层感知器（MLP）层来在信道间混合信息。本文介绍了 Kolmogorov-Arnold 变换器（KAT），
这是一种新颖架构，用 Kolmogorov-Arnold 网络（KAN）层取代 MLP 层，以提升模型的表现力和性能。然而，将 KAN 集成到变换器中并非易事，尤其是在放大时。
具体来说，我们识别出三个关键挑战：（C1） 基础函数。KAN 中使用的标准 B 样条函数未针对现代硬件上的并行计算进行优化，导致推理速度较慢。（C2）参数与计算效率低下。
KAN 需要为每对输入输出使用唯一函数，使计算量极大。（C3）权重初始化。由于其可学习激活函数，KAN 权重的初始化尤其具有挑战性，而激活函数对于实现深度神经网络中的收敛至关重要。
为克服上述挑战，我们提出了三个关键解决方案：（S1）有理基。我们将 B 样条函数替换为有理函数，以提升与现代 GPU 的兼容性。通过在 CUDA 中实现这一功能，我们实现了更快的计算速度。
（S2）组 KAN。我们通过一组神经元共享激活权重，以降低计算负载而不牺牲性能。（S3）保持方差初始化。我们仔细初始化激活权重，确保激活方差在各层间保持一致。
通过这些设计，KAT 在扩展性和表现上优于传统的基于 MLP 的变换器。
[papers][code]
[papers][code]
[papers][code]
[papers][code]
[papers][code]

[papers][code]
[papers][code]
[papers][code]
[papers][code]

[papers][code][papers][code]
[papers][code]
