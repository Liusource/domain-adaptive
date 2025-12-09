# domain-adaptive
## Papers
### Survey
1、 Video Unsupervised Domain Adaptation with Deep Learning: A Comprehensive Survey [[papers](https://arxiv.org/abs/2505.21046)] [[project](https://github.com/xuyu0010/awesome-video-domain-adaptation)][[code](https://github.com/zhaoxin94/awesome-domain-adaptation)]  
2、 A Survey on Deep Domain Adaptation for LiDAR Perception [[7 Jun 2021](https://arxiv.org/abs/2106.02377)]  
3、 A Comprehensive Survey on Transfer Learning [[7 Nov 2019](https://arxiv.org/abs/1911.02685)]  
4、 Transfer Adaptation Learning: A Decade Survey [[12 Mar 2019](https://arxiv.org/abs/1903.04687)]  
5、 A review of single-source unsupervised domain adaptation [16 Jan 2019]  
6、 An introduction to domain adaptation and transfer learning [31 Dec 2018]  
7、 A Survey of Unsupervised Deep Domain Adaptation [6 Dec 2018]  
8、 Transfer Learning for Cross-Dataset Recognition: A Survey [2017]  
9、 Domain Adaptation for Visual Applications: A Comprehensive Survey [2017] 

## Fault diagnosis
1、A Domain Adaptation Neural Network for Digital Twin-Supported Fault Diagnosis[[code](https://github.com/JialingRichard/Digital-Twin-Fault-Diagnosis)][[paper](https://arxiv.org/abs/2505.21046)]  
基于领域对抗神经网络（DANN）的故障诊断框架，实现了从模拟（源域）向现实世界（目标领域）数据的知识传输。我们利用一个公开的机器人故障诊断数据集评估该框架.DANN方法与常用的轻量级深度学习模型如CNN、TCN、Transformer和LSTM进行了比较。  
2、Transfer Learning Library for Fault Diagnosis[[code](https://github.com/Feaxure-fresh/TL-Fault-Diagnosis-Library)]  
单源无监督域适应、多源无监督域适应和域脑化。  
3、Adversarial Multiple-Target Domain Adaptation for Fault Classification [[paper](https://ieeexplore.ieee.org/abstract/document/9141312)][[code](https://github.com/mohamedr002/AMDA?tab=readme-ov-file)]  
提出了一种针对单一来源多目标（1SmT）场景的新型对抗性多靶域DA（AMDA）方法，该模型可以同时推广到多目标域。对抗适应技术应用于将多目标域特征转换为与单一来源域特征不变。

4 Integrating Expert Knowledge with Domain Adaptation for Unsupervised Fault Diagnosis[[papers](https://ieeexplore.ieee.org/document/9612159)][[code](https://github.com/qinenergy/syn2real)]  
将专家知识与领域适配整合以实现无监督故障诊断  
5 Unsupervised Cross-domain Fault Diagnosis Using Feature Representation Alignment Networks for Rotating Machinery[[papers](https://ieeexplore.ieee.org/document/9301443)][[code](https://github.com/JiahongChen/FRAN)]  
通过对齐从两个数据域提取的特征，来减轻从实验平台（源域）和作平台（目标域）收集的数据之间的域位移。最大化目标特征空间与整个特征空间之间的互信息，以提升标注数据在源域中的知识可转移性。此外，两个域之间的特征级差异被最小化，以进一步提高诊断准确性。

6 Interpretable Physics-informed Domain Adaptation Paradigm for Cross-machine Transfer Diagnosis[[papers](https://www.sciencedirect.com/science/article/abs/pii/S0950705124001345?via%3Dihub)][[code](https://github.com/liguge/WIDAN?tab=readme-ov-file)]  
尽管基于迁移学习的智能诊断取得了显著突破，但鉴于不同机器源域与目标域数据分布差异日益显著，现有知名方法的性能仍需紧迫提升。为了解决这个问题，我们没有设计域差异的统计指标或复杂的网络架构，而是深入探讨信号处理与域适应之间的相互作用与相互促进。受小波技术和权重初始化的启发，巧妙地设计出端到端、简洁且高性能的物理知情小波域适应网络（WIDAN），将可解释的小波知识集成到带有独立权重的双流卷积层中，以应对极具挑战性的跨机诊断任务。具体来说，CNN的第一层权重会被更新为优化且信息丰富的拉普拉斯权重或莫莱权重。这种方法缓解了参数选择的麻烦，因为具有特定物理解释的缩放和平移因子受卷积核参数的限制。此外，引入了平滑辅助的缩放因子，以确保与神经网络权重的一致性。此外，双流瓶颈层设计用于学习合理权重，预先将不同领域数据转换成统一的公共空间。这可以促进WIDAN提取域不变特征。
7 Both Reliable and Unreliable Predictions Matter: Domain Adaptation for Bearing Fault Diagnosis without Source Data[[papers](https://www.sciencedirect.com/science/article/abs/pii/S0925231225023331)][[code](https://github.com/BdLab405/SDALR?tab=readme-ov-file)]  
然而，现有方法并不理想，因为它们仅仅利用了自信地伪标记的目标样本，同时忽视了特征空间的内在结构特征。此外，故障伪标签的可靠性总是以熵估计，其准确性可以通过更复杂的策略提升。为解决这些问题，我们计划探讨目标域中特征与伪标签之间的相关性，以维持特征辨别性与特征多样性之间的平衡。此外，我们还开发了一种基于投票的策略，结合数据增强，以更准确地估计故障伪标签的可靠性。该方法能够通过自监督训练和分布结构发现，分别利用可靠样本和不可靠样本进行诊断模型转移。

8、Intelligent Ball Screw Fault Diagnosis Using Deep Learning Based Domain Adaptation and Transfer Learning[papers][[code](https://github.com/Fabian0597/MMD_PHMS)]


[papers][code]
