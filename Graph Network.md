“Graph Neural Network (GNN)” 与“振动 / 结构‑动力学 / 传递 (vibration / structural‑dynamics / vibration‑propagation / vibration‑transmission path)” 融合／应用。

“振动传递路径 (vibration transmission path)” 与 GNN — 机会与现状

“振动传递路径 (vibration transmission path)” 多指：振动 /力 /波 在结构中 (或结构组件 /子系统之间) 的传播路径 — 可能通过梁 /杆 /板 /连接节点 /界面 /耦合 /支撑 /传感器网络等。将 GNN 融入振动传递路径分析／建模，有以下潜力 & 对应现状／限制：

✅ 潜力 / 优点

高维、结构复杂系统可用 Graph 表示：结构 (beam, plate, shell, 复杂 assembly) 可离散为节点 (元素、质点、连接点) + 边 (连接 /相互作用 /耦合)，构成 graph。相比传统 FEM 网格、有限差分 (finite difference) 或有限元 mesh，graph 表示更灵活，容易处理非规范 / 不规则结构、多物理耦合 (结构‑声‑流体／声‑振动) 等复杂性。

高效 surrogate 模拟：如 GNSS、结构动力学响应预测、声振动 surrogate 模型，都显示 GNN 能在数值速度与准确度之间取得良好平衡 → 对于需要大量参数 sweep、优化 (design optimisation)、实时监测 (real-time SHM)、传感器布局优化 (sensor placement)、快速评估结构响应 (impact, excitation, vibration) 等任务非常有利。

对稀疏 / 不完整 /不规则测点 (sensor) 的适应性：通过图 (graph) 的方式整合结构拓扑 + 振动 /模态 /频域 /时域 数据 (如 modal shapes, PSD, acceleration measurements)，GNN 可用于模态分析、健康监测 (damage detection / localization)、传感器网络数据融合；适合 “现实世界” 中结构复杂、测点少、噪声大、不规则布点的场景。

⚠️ 当前研究 / 应用的限制与挑战

真实 “传递路径 (transmission path)” 信息少被显式建模：虽然已有研究把结构 /振动 /噪声系统离散为 graph，但多数为 surrogate / 近似模型 (approximation / regression)，它们对 “路径 (path)” — 即振动从一个点 → 通过连接 → 到另一个点的具体传播路径 — 的物理机制 (波传播、耦合、阻尼、界面条件等) 未必做显式建模／分析，而是隐式学习 (implicit learning)。也就是说，目前 GNN 更多担当“黑盒 surrogate 工具”，较少有研究专门分析哪条路径 (哪些边／连接／子结构) 对振动／噪声传递贡献最大 — 这对工程可解释性 (interpretability)、设计指导 (design insight) 是挑战。

数据 /标签 (ground‑truth) 获取困难：要训练 GNN 来预测振动传递路径 / 模态传播 /结构响应，需要大量高质量、带有时间／空间分辨率 (full‑field) 的数据 (例如通过仿真 FEM / 实验振动测试 /声‑结构耦合测试获得)；这些数据往往昂贵、耗时、难以获取，尤其是对于大型复杂结构 (工业设备、汽车底盘、航空结构等)。

规模 / 计算复杂度：对于大规模结构 (例如建筑、大型机械装置、耦合子系统) → 节点 /边数很多时，graph 构建 + GNN 推理 + time‑stepping rollout 的开销可能仍然很大。虽然 surrogate 思路比传统 FEM 要快，但仍需 careful design (例如图稀疏化, 局部连接半径, multi‑scale 模型, hierarchical GNN) 才能兼顾效率与准确性。

物理一致性 / 通用性 / 泛化问题：GNN 模型训练于某一结构 /加载 /boundary 条件下，泛化到不同结构 /不同类型激励 /不同耦合条件 (例如连接刚度变化、支撑条件、材料非线性、阻尼等) 时，其预测是否仍能保持物理一致性 (如正确传播路径、相位 / 波速、模态形状, 能量耗散) — 尚缺乏足够公开研究与验证。

1、Graph Network-based Structural Simulator: Graph Neural Networks for Structural Dynamics[[paper](https://arxiv.org/abs/2510.25683?utm_source=chatgpt.com)][code] [[Graph Network Simulator (GNS) and MeshNet](https://github.com/geoelements/gns)] [[graph_network_based_simulator](https://github.com/davidsj/graph_network_based_simulator)]
图神经网络（GNNs）最近被探索为数值模拟的替代模型。虽然它们在计算流体力学中的应用已被研究，但对结构问题，尤其是动力学案例，关注甚少。为弥补这一空白，我们引入了基于图网络的结构模拟器（GNSS），这是一个用于动态结构问题替代建模的GNN框架。GNSS遵循基于GNN的机器学习模型典型的编码-过程-解码范式，其设计使其特别适合动态仿真，这得益于三个关键特性：（i）在节点固定的局部帧中表达节点运动学，避免有限差分速度下的灾难性抵消;（ii）采用符号感知回归损耗，以减少长距离展开中的相位误差;以及（iii）采用波长知情的连通半径，优化图的构建。我们基于一个涉及50kHz汉宁调制脉冲激发束的案例研究来评估GNSS。结果显示，GNSS能够准确地在数百个时间步中重现问题的物理现象，并推广到未见的负载条件，即现有GNN未能收敛或无法提供有意义的预测。与显式有限元基线相比，GNSS在保持空间和时间真实性的同时实现了显著的推理加速。这些发现表明，具有物理一致性更新规则的保持局域性GNN是动态波支配结构仿真的竞争替代方案。
2 Machine learning prediction of structural dynamic responses using graph neural networks[[paper](https://www.sciencedirect.com/science/article/pii/S0045794923002183?utm_source=chatgpt.com)][code]
结构响应的预测对于分析受动力载荷下的结构行为至关重要。现有方法在多方面存在局限。由于需要高强度的劳动时间和专用设备，实验测试成本较高。数值模型在经过适当验证后，能够提供高保真度和劳动高效的冲击测试模拟，但计算成本高昂，这也阻碍了设计办公室中用于密集和大规模仿真的数值方法。数据驱动的机器学习方法也应用于结构响应预测，但它们通常利用直接输入-输出映射方案，预测静态场变量而不捕捉动态响应过程。为弥补这些空白，我们提出了一种基于图神经网络（GNN）的新型机器学习方法，用于全场结构动力学预测。我们的方法采用离散化的结构表示，并采用迭代展开预测方案，因此能够模拟全面的时空结构动力学，充分发挥结构动力学分析的潜力。通过多项基准测试，我们的方法能够准确预测相关场变量，如位移、应变和应力，适用于输入参数范围广的结构，如结构几何形态、冲击速度和位置。还进行了额外的插值和外推测试，以证明我们的方法具有固有的普遍性，即使所有输入都采样在训练分布之外，也能产生令人满意的预测。我们的方法同样高效，运行速度比常用的数值竞争者快一个数量级。作为首次使用GNN进行结构动力学预测且结果令人期待，GNN非常适合有效且高效的动态结构响应预测。



3 Using Graph Neural Networks and Frequency Domain Data for Automated Operational Modal Analysis of Populations of Structures[[paper](https://arxiv.org/abs/2407.06492?utm_source=chatgpt.com)][code]  
基于人口结构健康监测（PBSHM）范式最近被提出，作为一种有前景的方法，通过促进具有一定相似性的结构之间的迁移学习，增强工程结构的数据驱动评估。在本研究中，我们将这一概念应用于结构系统的自动模态识别。我们引入基于图神经网络（GNN）的深度学习方案，基于空间稀疏振动测量的功率谱密度（PSD），识别工程结构的模态属性，包括固有频率、阻尼比和模态形状。通过系统数值实验评估所提出的模型，采用两种具有相似拓扑特征但几何（尺寸、形状）和材料（刚度）特性不同的桁架群。结果表明，一旦训练完成，基于GNN的模型即使在存在测量噪声和稀疏测量位置的情况下，也能以良好的效率和可接受的准确度识别同一结构群体内未见结构的模态属性。基于GNN的模型在识别速度方面相比经典频域分解（FDD）方法具有优势，同时在识别精度方面也优于其他多层感知器（MLP）架构，使其成为PBSHM应用中极具前景的工具。

4 A review of graph neural network applications in mechanics-related domains[[paper](https://link.springer.com/article/10.1007/s10462-024-10931-y?utm_source=chatgpt.com)][code]  
与力学相关的任务在实现准确的几何和物理表示方面常常面临独特挑战，尤其是对于非均匀结构。图神经网络（GNN）已成为应对这些挑战的有前景工具，能够通过巧妙地从具有不规则底层结构的图数据中学习。因此，近年来受GNN进步启发，复杂力学相关应用激增。尽管有这一过程，但目前仍缺乏系统性综述来探讨GNN近期在力学相关任务中的进展。为弥合这一空白，本文旨在深入概述GNN在力学相关领域的应用，同时识别关键挑战并概述未来潜在的研究方向。在这篇综述文章中，我们首先介绍了在力学相关应用中广泛应用的GNNs基础算法。我们简明扼要地阐述了其基本原理，以建立坚实的理解，为探索GNN在力学相关领域的应用提供基础。本文的范围旨在涵盖文献对固体力学、流体力学和跨学科力学相关领域的分类，全面总结图表示方法、GNN架构及其子领域内的进一步讨论。此外，还总结了与这些应用相关的开放数据和源代码，方便未来研究者。本文推动GNN与力学的跨学科整合，并为有兴趣将GNN应用于解决复杂力学相关任务的研究者提供指导。

5 GNS: A generalizable Graph Neural Network-based simulator for particulate and fluid modeling[[paper](https://arxiv.org/abs/2211.10228?utm_source=chatgpt.com)][[code](https://github.com/geoelements/gns)]  
我们开发了基于PyTorch的图网络模拟器（GNS），能够学习物理原理并预测颗粒和流体系统的流动行为。GNS将领域离散化，节点代表物质点集合，连接节点的链接代表粒子或粒子簇之间的局部相互作用。GNS通过图上的消息传递学习交互定律。GNS包含三个部分：（a）编码器，将粒子信息嵌入潜图，边是学习的函数;（b） 处理器，允许数据传播并计算跨步节点交互;以及（c）解码器，从图中提取相关动力学（例如粒子加速度）。我们引入了受物理启发的简单归纳偏差，例如惯性框架，使学习算法能够优先考虑一种解（恒定引力加速度），从而减少学习时间。GNS实现采用半隐式欧拉积分，根据预测加速度更新下一状态。基于轨迹数据训练的GNS可推广预测训练中未见的复杂边界条件下粒子运动学。训练好的模型能准确预测其相关材料点方法（MPM）模拟误差的5%范围内。预测速度比传统MPM快5000倍（MPM模拟为2.5小时，GNS颗粒流模拟为20秒）。GNS替代器在解决优化、控制、临界区域预测（即原位）以及逆型问题方面非常受欢迎。GNS代码可通过开源MIT许可证访问此 https URL。

6 StructGNN: An Efficient Graph Neural Network Framework for Static Structural Analysis[paper][[code](https://github.com/CMMAi/StructGNN?utm_source=chatgpt.com)]  
在结构分析通过监督学习进行预测领域，神经网络被广泛应用。图神经网络（GNN）的最新进展扩展了其能力，使得利用图表示和GNN的消息传递机制，预测具有多样几何形状的结构。然而，GNN中的传统消息传递与结构性质不一致，导致计算效率低下，且推广到外推数据集的推广有限。为此，提出了一种新颖的结构图表示法，将伪节点作为每个故事中的刚性隔膜结合，并结合一个高效的GNN框架StructGNN。StructGNN 采用了根据结构故事数量身定制的自适应消息传递机制，使输入加载特征能够无缝地在结构图中传输。大量实验验证了该方法的有效性，预测位移、弯曲力矩和剪切力的准确率超过99%。StructGNN在非GNN模型上也表现出强烈的泛化性，在较高且未被看见的结构上平均准确率为96%。这些结果凸显了StructGNN作为可靠且计算高效工具的静态结构响应预测工具的潜力，有望应对结构分析中动态地震载荷相关的挑战。

7 pytorch_geometric [paper][[code](https://github.com/pyg-team/pytorch_geometric?utm_source=chatgpt.com)]  
提供最基础、广泛使用的 GNN 构建 /训练 /图数据处理 / mesh / point‑cloud 支持。非常适合作为基础库，从零开始构建结构‑动力学 /振动传播模型 (例如将 mesh → graph, 自定义 message‑passing / edge features / time‑stepping) 时使用。  

8 Machine learning prediction of structural dynamic responses using graph neural networks[[paper](https://www.sciencedirect.com/science/article/pii/S0045794923002183?utm_source=chatgpt.com)][code]  
结构响应的预测对于分析受动力载荷影响的结构行为至关重要。现有的方法在不同方面都有局限。由于需要高强度的劳动时间和专用设备，实验测试成本较高。数值模型在经过适当验证后可以提供高保真度和劳动效率的冲击测试模拟，但其计算成本高昂，这限制了设计办公室中用于密集和大规模模拟的数值方法。数据驱动机器学习方法也应用于结构响应预测，但通常利用直接输入输出映射方案，预测静态场变量而不捕捉动态响应过程。为弥补这些空白，我们提出了一种基于图神经网络（GNN）的新型机器学习方法，用于全场结构动力学预测。我们的方法采用结构的离散表示，并采用迭代展开预测方案，因此能够模拟全面的时空结构动力学，充分发挥结构动力学分析的潜力。通过多项基准测试，我们的方法证明能够准确预测相关场变量，如位移、应变和应力，适用于输入参数范围广泛的结构，如结构几何形状、冲击速度和位置。还进行了额外的插值和外推测试，以证明我们的方法具有固有的普遍性，即使所有输入都被抽样在训练分布之外，也能产生令人满意的预测。我们的方法同样高效，运行速度比常用的数值竞争者快一个数量级。由于首次尝试使用GNN进行结构动力学预测且结果令人满意，人们认为GNN非常适合有效且高效的动态结构响应预测。

9 Using graph neural networks and frequency domain data for automated operational modal analysis of populations of structures[[paper](https://www.cambridge.org/core/journals/data-centric-engineering/article/using-graph-neural-networks-and-frequency-domain-data-for-automated-operational-modal-analysis-of-populations-of-structures/5834E459A2DBFE3F881EE88645BF0EA3?utm_source=chatgpt.com)][code]  
基于人群的结构健康监测范式最近被认为是一种有前景的方法，通过促进具有一定相似性的结构之间的迁移学习，增强工程结构的数据驱动评估。在本研究中，我们将这一概念应用于结构系统的自动模态识别。我们引入基于图神经网络（GNN）的深度学习方案，基于空间稀疏振动测量的功率谱密度，识别工程结构的模态属性，包括固有频率、阻尼比和模态形状。通过系统数值实验评估所提出模型，采用两种截然不同的桁架群，它们具有相似的拓扑特征，但在几何（尺寸和形状）及材料（刚度）特性上有所不同。结果表明，一旦训练完成，基于GNN的模型即使在存在测量噪声和稀疏测量位置的情况下，也能以良好的效率和可接受的准确度识别同一结构群体内未见结构的模态属性。基于GNN的模型在识别速度方面相较经典频域分解方法具有优势，同时在识别精度方面也优于替代多层感知器架构，使其成为PBSHM应用中极具前景的工具。

10 Structural damage detection framework based on graph convolutional network directly using vibration data[[paper](https://www.sciencedirect.com/science/article/pii/S2352012422000662?utm_source=chatgpt.com)][code]  
本研究开发了一个新颖、高精度且稳健的框架，称为g-SDDL，用于结构损伤检测（SDD），直接利用振动数据，无需手工设计特征。传统的结构健康监测方法需要先进技术和领域专业知识来预处理振动信号以获得高精度结果，但这可能影响实时监测任务的执行。因此，直接利用振动数据是开辟这一雄心勃勃目标新路径的研究方向之一，这也是本研究的核心主题。为了有效利用振动数据，可以利用图神经网络捕捉传感器位置的固有空间相关性，并利用卷积作提取底层振动信号模式。此外，多个g-SDDL模型可以叠加，以应对多重损坏场景。该方法的可行性通过三个案例研究定量验证，复杂度逐渐增加，从一维连续混凝土梁到二维框架结构，再到文献中的实验数据库。即使在多伤害场景中，也始终保持超过90%的高伤害检测准确率。此外，通过比较、噪声注入和参数研究，研究了g-SDDL的性能和鲁棒性。

11 A Computational Framework for Modeling Complex Sensor Network Data Using Graph Signal Processing and Graph Neural Networks in Structural Health Monitoring[[paper](https://arxiv.org/abs/2105.05316?utm_source=chatgpt.com)][code]  
复杂网络适合建模多维数据，如关系型和/或时间型数据。特别是在需要形式化如此复杂数据及其内在关系时，复杂的网络建模及其生成的图表示方式提供了多种强大的解决方案。本文聚焦于结构健康监测图表上的具体机器学习方法，从分析和预测（维护）视角出发。具体来说，我们提出了一个基于复杂网络建模的框架，整合了图信号处理（GSP）和图神经网络（GNN）方法。我们在针对性的结构健康监测（SHM）应用领域中展示了该框架。我们特别关注一个显著的现实结构健康监测应用场景，即对荷兰一座大型桥梁的传感器数据（应变、振动）进行建模和分析。在我们的实验中，我们表明GSP能够识别最重要的传感器，因此我们研究了一套搜索和优化方法。此外，GSP能够检测特定的图信号模式（模态形状），捕捉复杂网络中传感器的物理功能特性。此外，我们还展示了在这类数据中应用GNN进行应变预测的有效性。

12 Graph Neural Network Assisted Genetic Algorithm for Structural Dynamic Response and Parameter Optimization[[paper](https://arxiv.org/abs/2510.22839?utm_source=chatgpt.com)][code]  
结构参数的优化，如质量（m）、刚度（k）和阻尼系数（c），对于设计高效、韧性和稳定结构至关重要。传统的数值方法，包括有限元法（FEM）和计算流体力学（CFD）仿真，能提供高保真度的结果，但对于迭代优化任务来说计算成本较高，因为每次评估都需要解每个参数组合的控制方程。本研究提出了一个混合数据驱动框架，将图神经网络（GNN）替代模型与遗传算法（GA）优化器整合，以克服这些挑战。GNN经过训练，能够准确学习结构参数与动态位移响应之间的非线性映射，从而实现快速预测，无需反复求解系统方程。利用Newmark Beta方法生成了涵盖不同质量、刚度和阻尼配置的单自由度（SDOF）系统响应数据集。GA随后通过最小化预测位移和增强动态稳定性，寻找全局最优参数集。结果表明，GNN和GA框架实现了强收敛性、稳健的泛化能力，并且相比传统仿真显著降低了计算成本。这种方法凸显了将机器学习替代工具与进化优化相结合，实现自动化和智能结构设计的有效性。  

13 Graphs4CFD[[code](https://github.com/mario-linov/graphs4cfd)]
[paper][code][paper][code][paper][code][paper][code][paper][code][paper][code][paper][code][paper][code][paper][code][paper][code][paper][code][paper][code]
