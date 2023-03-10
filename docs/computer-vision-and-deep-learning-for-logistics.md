# 面向物流的计算机视觉和深度学习

> 原文：<https://pyimagesearch.com/2022/11/14/computer-vision-and-deep-learning-for-logistics/>

* * *

## **目录**

* * *

## [**计算机视觉与深度学习用于物流**](#TOC)

在当今竞争激烈的市场中，拥有高效灵活的供应链是一项重要的资产。因此，公司正在寻找优化供应链的方法，以帮助他们做出决策，提高运营效率和客户满意度，并减少对环境的影响。

据[麦肯锡](https://www.mckinsey.com/industries/travel-logistics-and-infrastructure/our-insights/automation-in-logistics-big-opportunity-bigger-uncertainty)报道(**图 1** )，人工智能将在 2030 年定义一种新的“物流范式”。它将在未来 20 年内每年产生 1.3-2 万亿美元的收入，因为它在重复但关键的任务上继续优于人类。在另一项类似的[研究](https://www.mckinsey.com/industries/metals-and-mining/our-insights/succeeding-in-the-ai-supply-chain-revolution#:~:text=Successfully%20implementing%20AI%2Denabled%20supply,stake%2C%20multiple%20solutions%20have%20emerged.)、[中，麦肯锡](https://www.mckinsey.com/industries/metals-and-mining/our-insights/succeeding-in-the-ai-supply-chain-revolution#:~:text=Successfully%20implementing%20AI%2Denabled%20supply,stake%2C%20multiple%20solutions%20have%20emerged.)报告称，通过使用人工智能，企业可以将物流、库存和服务成本分别降低 15%、35%和 65%。

本系列是关于工业和大企业应用的 CV 和 DL。这个博客将涵盖在物流中使用深度学习的好处、应用、挑战和权衡。

本课是 5 课课程中的第 3 课:**工业和大型企业应用的 CV 和 DL 102**。

1.  [*计算机视觉与石油天然气深度学习*](https://pyimg.co/zfg51)
2.  [*用于交通运输的计算机视觉和深度学习*](https://pyimg.co/1u4c9)
3.  [***【计算机视觉与深度学习】用于物流***](https://pyimg.co/ux28n) **(本教程)**
4.  *用于医疗保健的计算机视觉和深度学习*
5.  *用于教育的计算机视觉和深度学习*

**要了解计算机视觉和深度学习对于物流来说，** ***只要坚持阅读。***

* * *

## [**计算机视觉与深度学习用于物流**](#TOC)

* * *

### [**好处**](#TOC)

* * *

#### [**丰富数据质量**](#TOC)

机器学习和自然语言处理可以依靠每天涌入物流公司的海量信息。他们可以理解术语和短语，建立联系以创建一个能够在运输中实现最佳数据使用的环境，避免风险，创建更好的解决方案，最大限度地利用资源并削减成本。这些丰富的数据可以帮助物流公司了解所需劳动力和资产的准确数量，并帮助他们优化日常运营。

* * *

#### [**战略资产定位**](#TOC)

2021 年，从中国运送一个标准的 40 英尺集装箱到美国东海岸要花费超过 20，000 美元。人工智能算法可以帮助规划者和物流公司安全地定位和保护他们的资产。这些算法可以通过减少空集装箱的装运和减少道路上的车辆数量来提高预测能力匹配的利用率。通过减少和重新安排运输到高需求地点所需的车辆，可以保证资产位置的效率和成本降低。

* * *

#### [**改进的预测分析**](#TOC)

计算从 A 点到 B 点的单次运输的最佳运输需要通过数据分析、容量估计和网络分析进行排序。人脑几乎不可能完成这些操作，因为它们既费时又容易出错。这就是人工智能的预测能力发挥作用的地方。人工智能可以很容易地汇编准确的数据，结合外部因素，并执行所有这些逻辑程序，以估计即将到来的需求。这可以帮助物流公司在运输行业中获得竞争优势，并削减不必要的成本。

例如，国际运输领导者 [DHL](https://www.dhl.com/global-en/home.html#:~:text=Learn%20More-,Our%20Divisions,transporting%20letters%2C%20goods%20and%20information.) 使用一个平台来监控在线和社交媒体帖子，以识别潜在的供应链问题。他们的人工智能系统可以识别短缺、访问问题、供应商状态等。

**图 2** 展示了 AI 在供应链每一步的好处。

* * *

### [**应用**](#TOC)

* * *

#### [**预测和计划**](#TOC)

人工智能支持的需求预测方法比 ARIMA(自回归综合移动平均)和指数平滑等传统预测方法更准确。这些方法考虑了人口统计、天气、历史销售、当前趋势以及在线和社交媒体帖子。改进的需求预测性能有助于制造商通过减少调度的车辆数量来降低运营成本，从而提高资产利用率。

通常有两种需求预测模型:中长期预测和短期预测。公司通常使用中期到长期的预算和计划来购买新资产(如仓库、车辆、配送中心等)。).这些预测的范围可以是 1-3 年。

但业内最广泛使用的预测是短期的，这极大地影响了运营规划，并提高了低利润率公司的底线。它们通常从几天到几周不等。例如，预测可以以超过 98%的准确率预测两周前的车辆需求/销售，以 95%的准确率预测六周前的需求/销售(**图 3** )。

在需求预测的帮助下，公司可以确保手头有适量的材料，并计划他们的生产活动。结果可以与关于成本、容量等的其他相关数据相结合。此外，假设在供应链管理过程中出现任何问题(例如，客户决定不下订单)。在这种情况下，预测性解决方案可以通过在潜在问题发生之前识别它们并进行相应的调整来避免没有人想要的产品的过量生产。

为了预测未来一周的购买量， [OTTO](https://ottomotors.com/) 开发了一种深度学习算法，分析了 30 亿条数据记录和 200 个变量，包括交易、OTTO 网站上的搜索查询和天气预报。因此，该系统对未来 30 天内将要出售的商品做出了 90%的预测。这帮助 OTTO 每月提前订购约 200，000 件商品，并在客户下订单时更快地一次性发货。

* * *

#### [**优化**](#TOC)

**路线优化:**路线优化使最后一英里配送的成本合理化，这是物流行业的一项重大开销。AI 可以分析历史行程、现有路线以及地理、环境和交通数据，以使用最短路径图算法，并为物流卡车确定最有效的方式(**图 4** )。这将减少运输成本和碳足迹。

[Zalando](https://engineering.zalando.com/posts/2015/12/accelerating-warehouse-operations-with-neural-networks.html) 已经训练了一个名为 OCaPi 算法(optimal cart pick)的神经网络，它可以让员工之间的拣货工作更加有效，并加快拣货过程。该算法不仅考虑员工的路线，还考虑转盘运载器的路径，当员工从货架上收集物品时，转盘运载器有时会停在横向过道中。这样，它就能找到最短的路线。

**成本和价格优化:**根据需求和供应，商品价格会有所波动。基于过去关于销售、数量、市场条件、货币汇率和通货膨胀的历史数据，预测分析可以帮助公司最大限度地降低错误定价的风险。这些模型可以告诉公司是否应该降低价格或增加利润，帮助他们在市场中脱颖而出(**图 5** )。

**库存优化:**库存优化帮助企业充分利用供应链。过多的库存而没有销售会导致贬值和损失——尤其是食品、药品等易腐商品。预测模型可以帮助组织始终保持正确的供应水平，从而降低投资成本和因生产过剩或库存不足造成的浪费。此外，这些模型可以使用有关客户行为和即将到来的事件(如假期)的历史数据来进行预测。

* * *

#### [**自动化仓库**](#TOC)

 **自动化仓储有两种类型:帮助搬运货物的设备和改善货物搬运的设备。在第一种类型中，自动导向车(AGV)可以实现箱子和托盘的移动。他们可以配备软件来改造标准叉车，使其实现自主。其他新技术，如 swarm 机器人(如亚马逊的 Kiva 机器人)，可以帮助将货架上的商品移动到目的地和传送带上。此外，先进的自动化存储/检索系统可以在大型货架上存储货物，并有机器人穿梭器，可以使用连接到结构的轨道在三维空间移动。

比如零售巨头[亚马逊](https://www.google.com/url?q=https://www.aboutamazon.com/news/operations/10-years-of-amazon-robotics-how-robots-help-sort-packages-move-product-and-improve-safety&sa=D&source=docs&ust=1662773573539191&usg=AOvVaw1SDO4zYWwVF8JwjtSUGnIS)2012 年收购 Kiva Systems，2015 年更名为亚马逊机器人。今天，亚马逊有 20 万个机器人在他们的仓库里工作。在亚马逊 175 个履行中心中的 26 个，机器人帮助挑选、分类、运输和装载包裹。

搬运设备可以自动进行货物的拣选、分类和码垛。他们通常有传感器，可以确定物体的形状和结构。然后，使用类似的 AI 算法，这些设备可以过滤掉任何不好的东西(例如，Magazino 的新 [TORU cube](https://www.magazino.eu/products/toru/?lang=en) )。甚至传送带也可以通过使用人工智能自动传感器来推进，这些传感器可以扫描包裹任何一侧的条形码，并确定适当的行动。

除了这些机器人机器，各种其他创新也可以提高仓库中人的生产率。

*   外骨骼可以通过手套或对腿部的额外支撑，用机械动力增强人类的运动。该系统允许人们移动更多的货物(例如，更重的物品)或者更有效和安全地移动货物。
*   AI 可以通过会计信息(例如，产品尺寸、重量等)使用计算机视觉对库存中存储的商品进行自动分类和识别。).AI 可以在没有人类协助的情况下快速定位仓库中的这些物品。
*   如果人工智能机器人负责操作危险设备和储存库存，员工的安全将得到改善。计算机视觉算法可以跟踪员工的工作，监控各种安全问题，并识别任何可疑行为。

* * *

#### [**预见性维护**](#TOC)

预测性维护(**图 7** ) 涉及通过分析实时传感器数据来检测工厂机器故障。为了使预测性维护正常工作，传感器必须记录与部件运动相关的所有参数。例如，这些因素包括码头的打开和关闭事件、控制系统动作、压力缸和滚轮磨损。

下一步是规定维护(**图 7** ) ，它包括根据对下一次故障、服务日期和时间、要供应的备件等的预测，主动安排非高峰时段的维护。该计划可能会影响相关设备的参数，从而使可能加剧损坏的动作不再以全功率执行。而是尽可能仔细地优化基础设施的使用，以便不在预定的维护日期之前触发故障。

此外，假设这是一种仅仅依靠传感器的情况是不正确的。现代传感器和传感器的链接使得检测不正确的传感器值以及人工智能结合其他测量值对这些值进行插值成为可能。然而，问题并不总是机器。传感器本身可能存在故障，但不一定会导致计划内停机。

* * *

#### [**后台及客户体验**](#TOC)

**后台运营:**每个企业都有手工完成的后台任务。自动化人工智能解决方案可以应用于此类任务，以下列方式提高后台运营效率:

*   **自动化文件处理:**文件自动化技术可用于通过自动化数据输入、错误核对和文件处理来快速处理发票/提货单/价目表文件(**图 8** )。
*   **调度和跟踪:**人工智能系统可以调度运输，组织货物管道，分配和管理不同的员工到特定的车站，并跟踪仓库中的包裹。
*   **报告生成:**机器人流程自动化(RPA)工具可以自动生成定期报告，分析其内容并通知利益相关者和员工。

**客户体验:**预测解决方案可以洞察客户的行为，从而帮助改善客户体验。他们可以确定客户下一步可能会购买什么，何时可以取消或退回产品，购买角色的最新趋势等。这种策略有助于公司留住客户，同时吸引新客户。这些预测算法还可以根据客户的选择和行为对客户进行细分，使公司能够根据需求更早地调整供应链和产品价格。

销售和营销团队可以根据客户的产品有效地锁定特定的客户群。然后，经理可以了解他们的营销策略如何影响客户的购买决策(例如，为什么有些人停止使用他们的产品)。是什么让他们转投其他品牌？预测分析还可以分析社交媒体帖子，如 Twitter、脸书和其他产品上的提及，以便及时获得消费者的反馈并改善他们的服务。

* * *

### [**挑战**](#TOC)

在物流行业应用人工智能有其自身的风险和挑战。

* * *

#### [**限制访问历史数据**](#TOC)

为了让这些预测性人工智能解决方案有效工作，公司必须访问跨各个业务部门和供应链收集的大量历史高质量数据。质量和数量取决于公司的规模、地理位置和已经采用的 IT 解决方案。因此，企业需要投入资源和时间来建立解决方案和设备，以战略性地收集有关其业务的相关信息。一个好的建议将是从低投资的人工智能设计冲刺研讨会开始，以处理数据收集和预测建模实施。

* * *

#### [**缺乏 360°视野**](#TOC)

企业仍然依赖于缺乏与整个供应链网络中的其他系统集成的遗留解决方案。软件解决方案通常不能覆盖所有流程，并且与不同的供应商不兼容，这使得跨平台合并数据变得更加困难。缺乏对供应链的 360 度全方位了解是运用预测分析的最大挑战。

* * *

#### [**缺乏人工智能专业人才**](#TOC)

各行业需要聘请顶尖的数据科学家和机器学习专家来构建和设计他们的算法和系统。不幸的是，对这类专家的需求超过了供给，这使得企业很难找到并雇用在业务领域知识、数据科学、数学和统计方面具有扎实专业知识的分析专业人员。因此，许多公司未能实施预测分析解决方案，因为他们没有足够的合格员工来开展复杂的人工智能项目。

* * *

* * *

## [**汇总**](#TOC)

公司正在寻找优化供应链的方法，以帮助他们做出决策，提高运营效率和客户满意度，并减少对环境的影响。人工智能正在定义一种新的“物流范式”，为物流行业的各种任务提供服务:

*   **预测和计划:**在需求预测的帮助下，公司可以确保手头有适量的材料，并计划他们的生产活动。结果可以与关于成本、容量等的其他相关数据相结合。
*   **优化:** AI 可以分析历史行程、现有路线以及地理、环境和交通数据，以使用最短路径图算法，并为物流卡车确定最有效的方式。此外，基于过去关于销售、数量、市场条件、货币汇率和通货膨胀的历史数据，预测分析可以帮助公司最大限度地降低错误定价的风险。
*   **自动化仓库:**自动导向车(AGVs)可以实现箱子和托盘的移动。他们可以配备软件来改造标准叉车，使其实现自主。
*   **预测和规定维护:**它涉及通过分析实时传感器数据来检测工厂机器故障，并根据对下一次故障、服务日期和时间、要供应的备件等的预测，主动安排非高峰时段的维护。
*   **后台操作:**通过自动化数据输入、错误核对和文档处理，文档自动化技术可用于快速处理发票/提单/价目表文档。

然而，物流行业的人工智能也带来了挑战。

*   **对历史数据的有限访问:**为了让这些预测性人工智能解决方案有效工作，公司必须访问从各个业务部门和供应链收集的大量历史高质量数据。质量和数量取决于公司的规模、地理位置和已经采用的 IT 解决方案。
*   **缺乏 360°视野:**企业仍然依赖于遗留解决方案，这些解决方案缺乏与整个供应链网络中其他系统的集成。软件解决方案通常不能覆盖所有流程，并且与不同的供应商不兼容，这使得跨平台合并数据变得更加困难。
*   **缺乏具有人工智能技能的专业人士:**各行业需要聘请顶尖的数据科学家和机器学习专家来构建和设计他们的算法和系统。不幸的是，对这类专家的需求超过了供给，这使得企业很难找到并雇用在业务领域知识、数据科学、数学和统计方面具有扎实专业知识的分析专业人员。

我希望这篇文章能帮助你理解在物流领域使用深度学习的好处、应用、挑战和权衡。请继续关注即将到来的课程，我们将讨论深度学习和计算机视觉在医疗保健行业的应用。

* * *

### [**咨询服务**](#TOC)

你的人工智能工业应用需要帮助吗？了解有关我们的[咨询服务](https://pyimagesearch.com/consulting-2/)的更多信息。

* * *

### [**引用信息**](#TOC)

**Mangla，P.** “用于物流的计算机视觉和深度学习”， *PyImageSearch* ，P. Chugh，A. R. Gosthipaty，S. Huot，K. Kidriavsteva 和 R. Raha 编辑。，2022 年，【https://pyimg.co/ux28n 

```py
@incollection{Mangla_2022_CVDLL,
  author = {Puneet Mangla},
  title = {Computer Vision and Deep Learning for Logistics},
  booktitle = {PyImageSearch},
  editor = {Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha},
  year = {2022},
  note = {https://pyimg.co/ux28n},
}
```

* * ***