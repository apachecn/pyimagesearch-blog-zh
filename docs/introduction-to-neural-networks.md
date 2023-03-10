# 神经网络导论

> 原文：<https://pyimagesearch.com/2021/05/06/introduction-to-neural-networks/>

我们将深入研究神经网络的基础。我们将从讨论人工神经网络开始，以及它们是如何受到我们自己身体中现实生活的生物神经网络的启发。从那里，我们将回顾经典的感知器算法和它在神经网络历史中扮演的角色。

在感知器的基础上，我们还将研究现代神经学习的基石*反向传播算法*——没有反向传播，我们将无法有效地训练我们的网络。我们还将从头开始用 Python 实现反向传播，确保我们理解这个重要的算法。

当然，现代神经网络库如 Keras 已经内置了(高度优化的)反向传播算法。每次我们希望训练神经网络时，手动实现反向传播就像每次我们处理通用编程问题时，从头开始编码链表或哈希表数据结构一样——不仅不现实，而且浪费我们的时间和资源。为了简化这个过程，我将演示如何使用 Keras 库创建标准的前馈神经网络。

最后，我们将讨论构建任何神经网络时需要的四个要素。

## **神经网络基础知识**

在使用卷积神经网络之前，我们首先需要了解神经网络的基础知识。我们将回顾:

人工神经网络及其与生物学的关系。开创性的感知器算法。
反向传播算法及其如何用于有效训练*多层*神经网络。
如何使用 Keras 库训练神经网络。

当你完成时，你将对神经网络有深刻的理解，并能够继续学习更高级的卷积神经网络。

## **神经网络简介**

神经网络是深度学习系统的构建模块。为了在深度学习方面取得成功，我们需要从回顾神经网络的基础知识开始，包括*架构*、*节点*、*类型*，以及用于“教导”我们的网络的*算法。*

我们将从神经网络及其背后的动机的高级概述开始，包括它们与人类大脑中生物学的关系。从那里我们将讨论最常见的架构类型，**前馈神经网络**。我们还将简要讨论*神经学习*的概念，以及它将如何与我们用来训练神经网络的算法相关联。

### **什么是神经网络？**

许多涉及智能、模式识别和物体探测的任务*极难* *自动化*，然而*似乎可以由动物和小孩轻松自然地完成*。例如，你家的狗是如何认出你这个主人，而不是一个完全陌生的人？一个小孩子如何学会辨别*校车*和*公交巴士*的区别？我们自己的大脑是如何每天下意识地执行复杂的模式识别任务，而我们却没有注意到呢？

答案就在我们自己的身体里。我们每个人都包含一个现实生活中的生物神经网络，它与我们的神经系统相连——这个网络由大量相互连接的*神经元*(神经细胞)组成。

*【神经】*一词是*【神经元】**【网络】*的形容词形式，表示一种图形状结构；因此，*“人工神经网络”*是一个试图模仿(或者至少是受其启发)我们神经系统中的神经连接的计算系统。人工神经网络也被称为*【神经网络】*或*【人工神经系统】*通常缩写人工神经网络，称之为*“ANN”*或简称为*“NN”——*我将使用这两个缩写。

对于一个被认为是神经网络的系统，它必须包含一个带标签的有向图结构，其中图中的每个节点执行一些简单的计算。从图论中，我们知道有向图由一组节点(即顶点)和一组将节点对链接在一起的连接(即边)组成。在**图 1 中，**我们可以看到这样一个 NN 图的例子。

每个节点执行一个简单的计算。然后，每个连接将一个*信号*(即计算的输出)从一个节点传送到另一个节点，由一个*权重*标记，指示信号被放大或减弱的程度。一些连接具有放大信号的大的、*正权重*，表明在进行分类时信号非常重要。其他的具有*负权重，*减弱信号的强度，从而指定节点的输出在最终分类中不太重要。如果这样一个系统由一个带有连接权重的图结构(如**图 1** 所示)组成，并且连接权重可以使用学习算法*修改*，那么我们称之为*人工神经网络*。

### **与生物学的关系**

我们的大脑由大约 100 亿个神经元组成，每个神经元都与大约 10，000 个其他神经元相连。神经元的细胞体被称为*胞体*，在这里输入(*树突*)和输出(*轴突*)将胞体与其他胞体连接起来(**图 2** )。

每个神经元在其树突处接收来自其他神经元的电化学输入。如果这些电输入足够强大，可以激活神经元，那么被激活的神经元就会沿着其轴突传递信号，并将其传递给其他神经元的树突。这些附着的神经元也可能会放电，从而继续传递信息。

这里的关键要点是，神经元触发是一个**二元运算** — **神经元要么*触发*，要么*不触发*** 。开火没有不同的“等级”。简而言之，神经元只有在体细胞接收到的总信号超过给定阈值时才会激活。

然而，请记住，人工神经网络只是受到我们对大脑及其工作方式的了解的启发。深度学习的目标是*而不是*模仿我们的大脑如何运作，而是获取我们*理解的片段*，并允许我们在自己的工作中绘制类似的类比。最终，我们对神经科学和大脑的深层功能了解不足，无法正确模拟大脑的工作方式——相反，我们带着我们的*灵感*继续前进。

### **人工模型**

让我们先来看看一个基本的神经网络，它对图 3 中的输入进行简单的加权求和。值 *x* [1] *、x* [2] *和 x* [3] 是我们 NN 的**输入**，通常对应于我们设计矩阵中的*单行*(即数据点)。常数值 1 是我们的偏差，假设它嵌入到设计矩阵中。我们可以认为这些输入是神经网络的输入特征向量。

在实践中，这些输入可以是用于以系统的、预定义的方式量化图像内容的向量(例如，颜色直方图、[方向梯度直方图](http://dx.doi.org/10.1109/CVPR.2005.177)、[局部二元模式](https://doi.org/10.1109/TPAMI.2002.1017623)等)。).在深度学习的背景下，这些输入是图像本身的*原始像素强度*。

每个 *x* 通过权重向量 ***W*** 连接到一个神经元，权重向量由 *w* [1] *，w* [2] *，…，w [n]* 组成，这意味着对于每个输入 *x* 我们也有一个关联的权重 *w* 。

最后，**图 3** 右边的*输出节点*取加权和，应用一个激活函数 *f* (用于确定神经元是否“触发”)，输出一个值。用数学方法表示输出，您通常会遇到以下三种形式:

```py
• f(w[1]x[1] +w[2]x[2] + ··· +w[n]x[n])
• f(∑[i=1 -> n]w[i]x[i])
• Or simply, f(net), where net = ∑ni=1wixi
```

不管输出值是如何表示的，请理解我们只是简单地取输入的加权和，然后应用激活函数 *f* 。

### **激活功能**

最简单的激活函数是感知器算法使用的“阶跃函数”。

![ f(\textit{net}) \begin{cases} 1 & \textit{if net} > 0 \\ 0 & \textit{otherwise} \end{cases} ](img/0ff383a2d0773e67b4de13a69b74c197.png " f(\textit{net}) \begin{cases} 1 & \textit{if net} > 0 \\ 0 & \textit{otherwise} \end{cases} ")

从上面的等式可以看出，这是一个非常简单的阈值函数。如果加权求和`∑[i=1 -> n]w[i]x[i]>0`，我们输出 1，否则，我们输出 0。

沿着*x*-轴绘制输入值，沿着 *y 轴绘制 *f* ( *net* )的输出，*我们可以看到为什么这个激活函数得到它的名字(**图 4** ，*左上*)。当 *net* 小于或等于零时 *f* 的输出始终为零。如果 *net* 大于零，那么 *f* 将返回一。因此，这个功能看起来像一个楼梯台阶，与你每天上下的楼梯没有什么不同。

然而，虽然直观且易于使用，但阶跃函数是不可微的，这在应用梯度下降和训练我们的网络时会导致问题。

相反，在神经网络文献的历史中使用的更常见的激活函数是 sigmoid 函数(**图 4** ，*右上*)，其遵循以下等式:

**【①**

 **sigmoid 函数是比简单阶跃函数更好的学习选择，因为它:

1.  在任何地方都是连续且可微的。
2.  关于 y 轴对称。
3.  渐近地接近其饱和值。

这里的主要优点是 sigmoid 函数的平滑性使得设计学习算法更容易。然而，sigmoid 函数有两个大问题:

1.  sigmoid 的输出不以零为中心。
2.  饱和的神经元基本上消除了梯度，因为梯度的增量将非常小。

双曲正切，或 *tanh* (具有类似于 s 形的形状)也大量用作激活函数，直到 20 世纪 90 年代后期(**图 4** 、*中左*):tanh 的公式如下:

**(2)***f*(*z*)=*tanh*(*z*)=(*e^(z)e^(—z)*)*/*(*e^(z)*+*e^(—z)*)

 ***tanh* 函数以零为中心，但当神经元饱和时，梯度仍然会消失。

我们现在知道激活函数有比 sigmoid 和 *tanh* 函数更好的选择。具体来说，[哈恩洛瑟等人的工作在其 2000 年的论文*中，数字选择和模拟放大共存于一个受皮层启发的硅电路*](https://doi.org/10.1038/35016072) 中，介绍了**整流线性单元(ReLU)** ，定义为:

**(3)***f*(*x*)=*max*(0*，x* )

由于绘制时的外观，ReLUs 也被称为“斜坡函数”(**图 4** 、*中右*)。请注意，对于负输入，函数为零，但对于正值，函数线性增加。ReLU 函数是不饱和的，而且计算效率极高。

根据经验，在几乎所有的应用中，ReLU 激活函数的表现都优于 sigmoid 函数和 T2 函数。结合 Hahnloser 和 Seung 在他们 2003 年的后续论文 [*对称阈值线性网络*](http://dx.doi.org/10.1162/089976603321192103) 中的允许和禁止集的工作，发现 ReLU 激活函数比以前的激活函数家族具有更强的生物学动机，包括更完整的数学证明。

截至 2015 年，ReLU 是深度学习中使用的*最流行的*激活函数 [(LeCun，Bengio，and Hinton，2015)](https://doi.org/10.1038/nature14539) 。然而，当我们的值为零时，问题就出现了——*梯度不能取*。

ReLUs 的一个变体，称为*泄漏 ReLUs* [(Maas、Hannun 和 Ng，2013)](https://sites.google.com/site/deeplearningicml2013/relu_hybrid_icml2013_final.pdf?attredirects=0&d=1) 当装置不工作时，允许一个小的非零梯度:

![f(\textit{net}) = \begin{cases} \textit{net} & \textit{if net} >= 0 \\ \alpha \times \textit{net} & \textit{otherwise} \end{cases}](img/20256a171315b6dca61a9edd80537d5e.png "f(\textit{net}) = \begin{cases} \textit{net} & \textit{if net} >= 0 \\ \alpha \times \textit{net} & \textit{otherwise} \end{cases}")

在**图 4** ( *左下方*中绘制该函数，我们可以看到该函数确实允许取负值，不像传统的 ReLUs 将函数输出“箝位”在零。

参数 ReLUs，或简称为 PReLUs[(**何，张，任，孙，2015** )](https://arxiv.org/abs/1512.03385v1) ，建立在泄漏 ReLUs 的基础上，并允许在逐个激活的基础上学习参数 *α* ，这意味着网络中的每个节点都可以学习与其他节点不同的“泄漏系数”。

最后，我们还有 [Clevert 等人在他们 2015 年的论文*中介绍的*指数线性单元(ELUs)* 通过指数线性单元(ELUs)*](https://arxiv.org/abs/1511.07289) 进行快速准确的深度学习:

![f(\textit{net}) = \begin{cases} \textit{net} & \textit{if net} >= 0 \\ \alpha \times (\textit{exp}(\textit{net}) - 1) & \textit{otherwise} \end{cases}](img/47b7e8c5428ef5341a81123f7da35471.png "f(\textit{net}) = \begin{cases} \textit{net} & \textit{if net} >= 0 \\ \alpha \times (\textit{exp}(\textit{net}) - 1) & \textit{otherwise} \end{cases}")

*α* 的值是恒定的，并且*是在网络架构被实例化*时设置的——这不像在 PReLUs 中学习 *α* 。 *α* 的典型值为 *α* = 1 *。* 0。**图 4** ( *右下*)可视化 ELU 激活功能。

通过 Clevert 等人的工作(以及我自己的轶事实验)，ELUs 往往比 ReLUs 获得更高的分类精度。eLU 的性能很少比标准的 ReLU 函数差。

### 我使用哪个激活功能？

鉴于深度学习的最新化身的流行，激活功能也出现了相关的爆炸。由于激活功能的选择数量，现代(ReLU，漏 ReLU，eLU 等。)和“古典”的(step、sigmoid、*、tanh* 等。)，选择一个合适的激活函数可能看起来是一项艰巨的，甚至是压倒性的任务。

然而，在几乎所有情况下，我建议从 ReLU 开始，以获得基线准确性(正如在深度学习文献中发表的大多数论文一样)。从那里你可以试着把你的标准 ReLU 换成一个泄漏的 ReLU 变体。

我个人的偏好是从一个 ReLU 开始，调整我的网络和优化器参数(架构，学习率，正则化强度等。)，并注意准确性。一旦我对准确度相当满意，我就换上 ELU，并经常注意到分类准确度提高了 1*-*5 %,这取决于数据集。同样，这只是我的轶事建议。你应该运行你自己的实验并记录你的发现，但是作为一般的经验法则，从一个正常的 ReLU 开始并调整你的网络中的其他参数——然后换入一些更“奇特”的 ReLU 变体。

### **前馈网络架构**

虽然有许多不同的神经网络架构，但最常见的架构是*f*前馈网络，如图**图 5** 所示。

在这种类型的架构中，从层 *i* 中的节点到层 *i* +1 中的节点之间的连接*只允许*(因此有了术语*前馈*)。不允许向后或层间连接。当前馈网络包括*反馈连接*(反馈到输入的输出连接)时，它们被称为**递归神经网络**。

我们专注于前馈神经网络，因为它们是应用于计算机视觉的现代深度学习的基石。卷积神经网络只是前馈神经网络的一个特例。

为了描述一个前馈网络，我们通常使用一个整数序列来快速简洁地表示每层中的节点数。例如**图 5** 中的网络是一个 *3-2-3-2* 前馈网络:

**层 0** 包含 3 个输入，我们的*x**I*[值。这些可以是图像的原始像素强度或从图像中提取的特征向量。]

**层 1 和 2** 是*隐藏层*，分别包含 2 和 3 个节点。

**第 3 层**是*输出层或可见层* —在那里我们从我们的网络获得整体输出分类。输出层通常具有与类标签一样多的节点；每个潜在输出一个节点。例如，如果我们要建立一个神经网络来分类手写数字，我们的输出层将由 10 个节点组成，每个节点对应一个数字 *0-9* 。

### **神经学习**

神经学习指的是修改网络中节点之间的权重和连接的方法。从生物学上来说，我们根据 Hebb 的原则来定义学习:

> *当细胞 A 的轴突足够靠近以激发细胞 B，并且重复或持续地激发它时，在一个或两个细胞中发生一些生长过程或代谢变化，使得作为激发 B 的细胞之一的 A 的效率增加。*
> 
> *—* [Donald Hebb (1949)](https://www.amazon.com/Organization-Behavior-Neuropsychological-Theory/dp/0805843000)

就人工神经网络而言，这一原则意味着，当输入相同时，输出相似的节点之间的连接强度应该增加。我们称之为*相关性学习*，因为神经元之间的连接强度最终代表了输出之间的相关性。

### **神经网络是用来做什么的？**

神经网络可用于监督、非监督和半监督学习任务，当然，前提是使用适当的架构。神经网络的常见应用包括分类、回归、聚类、矢量量化、模式关联和函数逼近等。

事实上，对于机器学习的几乎每个方面，神经网络都以某种形式被应用。我们将使用神经网络进行计算机视觉和图像分类。

### **神经网络基础知识概述**

今天，我们回顾了人工神经网络的基础知识。我们从检查人工神经网络背后的生物动机开始，然后学习我们如何能够*用数学方法定义*一个模拟神经元激活的函数(即激活函数)。

基于神经元的这个模型，我们能够定义一个网络的*架构*，该网络由(最低限度的)*输入层*和*输出层*组成。一些网络架构可能在输入层和输出层之间包含多个隐藏层。最后，每层可以有一个或多个节点。输入层*中的节点*不包含激活函数(它们是我们图像的单个像素强度被输入的“地方”)；然而，隐藏层和输出层*中的节点*包含一个激活函数。

我们还回顾了三个流行的激活函数: *sigmoid* 、 *tanh* 和 *ReLU* (及其变体)。

传统上，sigmoid 和 *tanh* 函数被用于训练网络；然而，自从[哈恩洛瑟等人 2000 年的论文](https://doi.org/10.1038/35016072)以来，ReLU 函数被更多地使用。

2015 年，ReLU 是*到目前为止*深度学习架构中使用的最流行的激活函数 [(LeCun，Bengio，和 Hinton，2015)](https://doi.org/10.1038/nature14539) 。基于 ReLU 的成功，我们还有*泄漏 ReLU*，一种 ReLU 的变体，它通过允许函数取负值来改善网络性能。leaky ReLU 系列函数由标准 Leaky ReLU 变体、PReLUs 和 eLUs 组成。

最后，值得注意的是，尽管我们严格地在*图像分类*的背景下关注深度学习，但神经网络已经以某种方式用于几乎所有的机器学习领域。****