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
数字孪生通过生成模拟数据用于模型训练，为深度学习故障诊断中缺乏足够标记数据提供了有前景的解决方案。然而，仿真与现实系统之间的差异可能导致模型在实际场景中应用时性能显著下降。为解决这一问题，我们提出了基于领域对抗神经网络（DANN）的故障诊断框架，实现了从模拟（源域）向现实世界（目标领域）数据的知识传输。我们利用一个公开的机器人故障诊断数据集评估该框架，该数据集包含3600条由数字孪生模型生成的序列和90条从物理系统收集的真实序列。DANN方法与常用的轻量级深度学习模型如CNN、TCN、Transformer和LSTM进行了比较。实验结果表明，纳入结构域适应显著提升了诊断性能。例如，将DANN应用于基线CNN模型，其在真实世界测试数据上的准确率从70.00%提升到80.22%，展示了领域适应在弥合模拟与现实差距中的有效性。    
2、Transfer Learning Library for Fault Diagnosis[[code](https://github.com/Feaxure-fresh/TL-Fault-Diagnosis-Library)]  
单源无监督域适应、多源无监督域适应和域脑化。  
3、Adversarial Multiple-Target Domain Adaptation for Fault Classification [[paper](https://ieeexplore.ieee.org/abstract/document/9141312)][[code](https://github.com/mohamedr002/AMDA?tab=readme-ov-file)]  
数据驱动的故障分类方法正受到广泛关注，因为它们可以应用于许多实际应用。然而，它们假设训练数据和测试数据来自同一分布。实际场景的作条件各异，导致域移问题显著降低诊断性能。最近，领域适应（DA）被探索，通过将知识从有标签源域（如源工作状态）转移到无标记目标域（如目标工作条件）来解决域转移问题。然而，所有现有方法都在单源单目标（1S1T）条件下运行。因此，每个新的目标域都需要训练一个新的模型。这显示出在处理多种工作条件时的可扩展性有限，因为不同模型应针对不同的目标工作条件进行训练，这显然在实际中并非可行的解决方案。为解决该问题，我们提出了一种针对单源多靶（1SmT）场景的新型对抗性多靶域DA（AMDA）方法，该模型可以同时推广到多靶域。对抗性适应被应用来将多目标域特征转换为与单一源域特征不变。这带来了一个具有可扩展性、能够推广到多靶域的新颖能力的模型。在两个公开数据集和一个自收集数据集上的大量实验表明，所提方法始终优于最先进的方法。  

4 Integrating Expert Knowledge with Domain Adaptation for Unsupervised Fault Diagnosis[[papers](https://ieeexplore.ieee.org/document/9612159)][[code](https://github.com/qinenergy/syn2real)]  
将专家知识与领域适配整合以实现无监督故障诊断  
5 Unsupervised Cross-domain Fault Diagnosis Using Feature Representation Alignment Networks for Rotating Machinery[[papers](https://ieeexplore.ieee.org/document/9301443)][[code](https://github.com/JiahongChen/FRAN)]  
通过对齐从两个数据域提取的特征，来减轻从实验平台（源域）和作平台（目标域）收集的数据之间的域位移。最大化目标特征空间与整个特征空间之间的互信息，以提升标注数据在源域中的知识可转移性。此外，两个域之间的特征级差异被最小化，以进一步提高诊断准确性。

6 Interpretable Physics-informed Domain Adaptation Paradigm for Cross-machine Transfer Diagnosis[[papers](https://www.sciencedirect.com/science/article/abs/pii/S0950705124001345?via%3Dihub)][[code](https://github.com/liguge/WIDAN?tab=readme-ov-file)]  
尽管基于迁移学习的智能诊断取得了显著突破，但鉴于不同机器源域与目标域数据分布差异日益显著，现有知名方法的性能仍需紧迫提升。为了解决这个问题，我们没有设计域差异的统计指标或复杂的网络架构，而是深入探讨信号处理与域适应之间的相互作用与相互促进。受小波技术和权重初始化的启发，巧妙地设计出端到端、简洁且高性能的物理知情小波域适应网络（WIDAN），将可解释的小波知识集成到带有独立权重的双流卷积层中，以应对极具挑战性的跨机诊断任务。具体来说，CNN的第一层权重会被更新为优化且信息丰富的拉普拉斯权重或莫莱权重。这种方法缓解了参数选择的麻烦，因为具有特定物理解释的缩放和平移因子受卷积核参数的限制。此外，引入了平滑辅助的缩放因子，以确保与神经网络权重的一致性。此外，双流瓶颈层设计用于学习合理权重，预先将不同领域数据转换成统一的公共空间。这可以促进WIDAN提取域不变特征。  

7 Both Reliable and Unreliable Predictions Matter: Domain Adaptation for Bearing Fault Diagnosis without Source Data[[papers](https://www.sciencedirect.com/science/article/abs/pii/S0925231225023331)][[code](https://github.com/BdLab405/SDALR?tab=readme-ov-file)]  
然而，现有方法并不理想，因为它们仅仅利用了自信地伪标记的目标样本，同时忽视了特征空间的内在结构特征。此外，故障伪标签的可靠性总是以熵估计，其准确性可以通过更复杂的策略提升。为解决这些问题，我们计划探讨目标域中特征与伪标签之间的相关性，以维持特征辨别性与特征多样性之间的平衡。此外，我们还开发了一种基于投票的策略，结合数据增强，以更准确地估计故障伪标签的可靠性。该方法能够通过自监督训练和分布结构发现，分别利用可靠样本和不可靠样本进行诊断模型转移。

8、Intelligent Ball Screw Fault Diagnosis Using Deep Learning Based Domain Adaptation and Transfer Learning[papers][[code](https://github.com/Fabian0597/MMD_PHMS)]
Source-Free Domain Adaptation via Multimodal Space-Guided Alignment[[papers](https://www.sciencedirect.com/science/article/abs/pii/S0031320325014906)][[code](https://github.com/YunxiangBai0/MMGA/)]
传统的UDA要求访问源域，在信息安全和隐私保护场景中使源域失效。相比之下，无源域适配（SFDA）涉及在源数据缺失时，将预训练的源模型转移到未标记的目标域。然而，基于自监督学习的早期方法由于缺乏源数据，难以找到高质量的域不变表示空间。为应对这一挑战，本研究提出利用视觉语言预训练（ViL）模型（如CLIP）的成功经验。为了更有效地整合ViL模型的领域通用性和源模型的任务特异性，我们引入了一种新颖的MultiModal Space-Guided A木质（MMGA）方法。具体来说，我们从多模态特征校准开始，以实现目标视觉域与多模态空间之间的粗略比对。然而，该 ViL 空间仍不是定义域不变量空间，因为它训练于大量样本。为了进一步实现向域不变量空间的细粒度比对，我们设计了两种方法：潜在类别一致性和预测一致性一致性对齐。这些方法使潜在类别分布和预测分布分别更接近ViL模型和适应源模型融合的伪监督。该策略纠正特征对齐到 ViL 空间的错误。大量实验表明，我们的MMGA方法显著优于现有最先进的替代方案。

[papers][code]
[papers][code]
[papers][code]
[papers][code][papers][code][papers][code][papers][code][papers][code][papers][code][papers][code][papers][code][papers][code][papers][code]
