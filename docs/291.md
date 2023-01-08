# 影像分类基础

> 原文：<https://pyimagesearch.com/2021/04/17/image-classification-basics/>

> *一图抵千言*。
> 
> — English idiom

我们在生活中无数次听到这句格言。它只是意味着一个复杂的想法可以在一个单一的图像中传达。无论是检查我们的股票投资组合的折线图，还是查看即将到来的足球比赛的分布，或者仅仅是欣赏绘画大师的艺术和笔触，我们都在不断地吸收视觉内容，解释意义，并将知识存储起来以备后用。

然而，对于计算机来说，解释图像的内容就不那么简单了——我们的计算机看到的只是一个大的数字矩阵。它对图像试图传达的思想、知识或意义毫无概念。

为了理解一幅图像的内容，我们必须应用 ***图像分类*** ，这是使用计算机视觉和机器学习算法从一幅图像中提取意义的任务。这个操作可以简单到给图像所包含的内容分配一个标签，也可以高级到解释图像的内容并返回一个人类可读的句子。

图像分类是一个非常大的研究领域，包括各种各样的技术——随着深度学习的流行，它还在继续增长。

现在是驾驭深度学习和图像分类浪潮的时候了——那些成功做到这一点的人将获得丰厚的回报。

图像分类和图像理解目前是(并将继续是)未来十年计算机视觉最热门的子领域。在未来，我们会看到像谷歌、微软、百度和其他公司迅速获得成功的图像理解创业公司。我们将在智能手机上看到越来越多的消费应用程序，它们可以理解和解释图像的内容。甚至战争也可能使用由计算机视觉算法自动导航的无人驾驶飞机。

在这一章中，我将提供什么是图像分类的高层次概述，以及图像分类算法必须克服的许多挑战。我们还将回顾与图像分类和机器学习相关的三种不同类型的学习。

最后，我们将通过讨论训练用于图像分类的深度学习网络的四个步骤以及这四个步骤的管道与*传统、* *手工设计的*特征提取管道相比如何来结束这一章。

## **什么是图像分类？**

图像分类，就其核心而言，是从*预定义的* *类别集*中*给图像*分配标签的任务。

实际上，这意味着我们的任务是分析输入图像并返回一个对图像进行分类的标签。标签总是来自一组预定义的可能类别。

例如，让我们假设我们的可能类别集包括:

`categories = {cat, dog, panda}`

然后，我们将下面的图像(**图 1** )呈现给我们的分类系统:

我们在这里的目标是获取这个输入图像，并从我们的**类别**集合中给它分配一个标签——在本例中，是**狗**。

我们的分类系统还可以通过概率给图像分配多个标签，例如**dog:95%**；**猫:4%**；**熊猫:1%** 。

更正式的说法是，给定我们的输入图像为具有三个通道的 *W×H* 像素，分别为红色、绿色和蓝色，我们的目标是获取 *W×H×3 = N* 像素图像，并找出如何正确地对图像内容进行分类。

### **术语注释**

当执行机器学习和深度学习时，我们有一个 ***数据集*** 我们试图从中提取知识。数据集中的每个例子/项目(无论是图像数据、文本数据、音频数据等等。)是一个 ***数据点*** 。因此，数据集是数据点的集合(**图 2** )。

我们的目标是应用机器学习和深度学习算法来发现数据集中的潜在模式，使我们能够正确分类我们的算法尚未遇到的数据点。现在请花时间熟悉这些术语:

1.  在图像分类的背景下，我们的*数据集*是图像的*集合。*
2.  因此，每个*图像*是一个*数据点*。

在本书的其余部分，我将交替使用术语*图像*和*数据点*，所以请记住这一点。

### **语义鸿沟**

看看**图 3** 中的两张照片(*上*)。对我们来说，分辨两张照片之间的差异应该是相当微不足道的——左边明显是一只*猫*，右边是一只*狗*。但是计算机看到的只是两个大的像素矩阵(*底部*)。

假设计算机看到的只是一个大的像素矩阵，我们就有了语义差距的问题。语义鸿沟是人类*如何感知*图像内容与图像如何以计算机能够理解的方式*表示*之间的差异。

再一次，对上面两张照片的快速视觉检查可以揭示两种动物之间的差异。但事实上，计算机一开始就不知道图像中有动物。为了说明这一点，请看一下**图 4** ，其中有一张宁静海滩的照片。

我们可以这样描述这幅图像:

*   **空间:**天空在图像的顶部，沙子/海洋在底部。
*   **颜色:**天空是深蓝色，海水是比天空更淡的蓝色，而沙子是黄褐色。
*   **纹理:**天空有比较均匀的花纹，而沙子很粗糙。

我们如何以计算机能够理解的方式对所有这些信息进行编码？答案是应用*特征提取*来量化图像的内容。特征提取是获取输入图像、应用算法并获得量化我们的图像的特征向量(即，数字列表)的过程。

为了完成这个过程，我们可以考虑应用手工设计的特征，如 HOG、LBPs 或其他“传统的”方法来量化图像。另一种方法，也是这本书采用的方法，是应用深度学习来自动*学习一组特征*，这些特征可以用来量化并最终*标记*图像本身的内容。

然而，事情没那么简单。。。因为一旦我们开始检查现实世界中的图像，我们就会面临许多挑战。

### **挑战**

如果语义差距还不足以成为问题，我们还必须处理图像或物体如何出现的的变异因素。**图 5** 显示了这些变化因素的可视化。

首先，我们有 ***视点变化*** ，其中对象可以根据对象的拍摄和捕捉方式在多个维度上定向/旋转。无论我们从哪个角度捕捉这个树莓皮，它仍然是一个树莓皮。

我们还必须考虑到**的尺度变化。你曾经在星巴克点过一杯大杯或超大杯的咖啡吗？从技术上讲，它们都是一样的东西——一杯咖啡。但是它们都是一杯不同大小的咖啡。此外，同样的大杯咖啡，近距离拍摄和远距离拍摄看起来会有很大的不同。我们的图像分类方法必须能够容忍这些类型的尺度变化。**

 **最难解释的变化之一是*。对于那些熟悉电视连续剧*的人来说，我们可以在上图中看到主角。正如电视节目的名字所暗示的，这个角色富有弹性，可拉伸，能够以许多不同的姿势扭曲他的身体。我们可以将 *Gumby* 的这些图像视为一种*对象变形*——所有图像都包含 *Gumby* 角色；然而，它们彼此之间都有很大的不同。**

 *我们的图像分类还应该能够处理*，其中我们想要分类的对象的大部分在图像中隐藏起来(**图 5** )。在左边的*，*我们必须有一张狗的照片。在右边的*，*我们有一张同一只狗的照片，但是请注意这只狗是如何在被子下休息的，从我们的视角来看*被遮挡*。狗在两幅图像中都清晰可见——只是在一幅图像中她比另一幅更明显。图像分类算法应该仍然能够在两幅图像中检测和标记狗的存在。*

 *正如上面提到的*变形*和*遮挡*一样具有挑战性，我们还需要处理 ***光照*** 的变化。看看在标准和弱光下拍摄的咖啡杯(**图 5** )。左侧*的图像是在标准顶灯照明下拍摄的，而右侧*的图像是在非常微弱的照明下拍摄的。我们仍然在检查同一个杯子——但根据照明条件，杯子看起来有很大的不同(杯子的垂直纸板接缝在低照明条件下清晰可见，但不是标准照明条件下)。**

继续下去，我们还必须考虑 ***背景杂波*** 。玩过*游戏吗，沃尔多在哪里？沃利在哪里？为了我们的国际读者。)如果是这样，那么你知道游戏的目标是找到我们最喜欢的红白条纹衬衫朋友。然而，这些谜题不仅仅是一个有趣的儿童游戏——它们也是*背景混乱*的完美代表。这些图像非常“嘈杂”，里面有很多东西。我们只对图像中的*一个*特定对象感兴趣；然而，由于所有的“噪音”，不容易挑出瓦尔多/沃利。如果对我们来说都不容易做到，想象一下对图像没有语义理解的计算机有多难！*

最后，我们有 ***类内变异*** 。计算机视觉中类内变异的典型例子是展示椅子的多样化。从我们用来蜷缩着看书的舒适椅子，到家庭聚会餐桌上摆放的椅子，再到知名家庭中的超现代艺术装饰风格的椅子，椅子仍然是椅子——我们的图像分类算法必须能够正确地对所有这些变化进行分类。

您是否开始对构建图像分类器的复杂性感到有点不知所措？不幸的是，情况只会变得更糟——对于我们的图像分类系统来说，独立地对这些变化*鲁棒是不够的*，但是我们的系统还必须处理*多个变化组合* *在一起！*

那么，我们如何解释物体/图像中如此惊人数量的变化呢？总的来说，我们尽可能把问题框住。我们对图像的内容以及我们想要容忍的变化做出假设。我们还考虑项目的范围——最终目标是什么？我们想要建立什么？

在编写一行代码之前，部署到现实世界中的成功的计算机视觉、图像分类和深度学习系统会做出*仔细的假设和考虑*。

如果你采取的方法过于宽泛，比如*“我想分类并检测我厨房里的每一个物体**”*(可能有数百个可能的物体)，那么你的分类系统不太可能表现良好，除非你有多年构建图像分类器的经验——即使如此，也不能保证项目成功。

但是如果你**把你的问题**框起来，缩小范围，比如*“我只想识别炉子和冰箱”，*那么你的系统**更有可能**准确并正常工作，*尤其是*如果这是你第一次使用图像分类和深度学习。

这里的关键要点是 ***始终考虑你的图像分类器*** 的范围。虽然深度学习和卷积神经网络在各种挑战下表现出了显著的鲁棒性和分类能力，但你*仍然*应该尽可能保持你的项目范围紧凑和定义明确。

请记住，[ImageNet](https://link.springer.com/article/10.1007/s11263-015-0816-y),*事实上的*图像分类算法标准基准数据集，由我们日常生活中遇到的 1000 个对象组成——并且这个数据集*仍然*被试图推动深度学习发展的研究人员积极使用。

深度学习是*而不是*魔法。相反，深度学习就像你车库里的一把锯子——正确使用时强大而有用，但如果使用不当则很危险。在本书的其余部分，我将指导您进行深度学习，并帮助您指出何时应该使用这些强大的工具，何时应该参考更简单的方法(或者提及图像分类无法解决的问题)。

## **学习类型**

在你的机器学习和深度学习生涯中，你很可能会遇到三种类型的学习:监督学习、非监督学习和半监督学习。这本书主要关注深度学习背景下的监督学习。尽管如此，下面还是介绍了所有三种类型的学习。

### **监督学习**

想象一下:你刚刚大学毕业，获得了计算机科学的学士学位。你还年轻。破产了。在这个领域找工作——也许你甚至会在找工作的过程中感到迷茫。

但在你意识到之前，一位谷歌招聘人员在 LinkedIn 上找到了你，并给你提供了一个在他们的 Gmail 软件上工作的职位。你会接受吗？很有可能。

几个星期后，你来到谷歌位于加州山景城的壮观园区，被令人惊叹的景观、停车场的特斯拉车队和自助餐厅几乎永无止境的美食所淹没。

你终于在自己的办公桌前坐下来，置身于数百名员工之间的一个敞开的工作空间。。。然后你会发现你在公司中的角色。你受雇开发一款软件，让*自动将*电子邮件分类为*垃圾邮件*或*非垃圾邮件*。

你将如何完成这个目标？基于规则的方法可行吗？您能否编写一系列`if/else`语句来查找特定的单词，然后根据这些规则确定一封电子邮件是否是垃圾邮件？那可能有用。。。在某种程度上。但这种方法也很容易被击败，而且几乎不可能维持下去。

相反，你真正需要的是机器学习。你需要一个*训练集*，由电子邮件本身以及它们的*标签*组成，在本例中，是*垃圾邮件*或*非垃圾邮件*。有了这些数据，您可以分析电子邮件中的文本(即单词的分布)，并利用垃圾邮件/非垃圾邮件标签来教导机器学习分类器哪些单词出现在垃圾邮件中，哪些没有出现，而无需手动创建一系列复杂的`if/else`语句。

这个创建垃圾邮件过滤系统的例子是**监督学习**的例子。监督学习可以说是最知名和最受研究的机器学习类型。给定我们的训练数据，通过训练过程创建模型(或“分类器”)，其中对输入数据进行预测，然后在预测错误时进行纠正。这个训练过程一直持续到模型达到某个期望的停止标准，例如低错误率或训练迭代的最大次数。

常见的监督学习算法包括逻辑回归、支持向量机(SVMs) ( [Cortes 和 Vapnik，1995](http://dx.doi.org/10.1023/A:1022627411411) 、 [Boser 等人，1992](http://doi.acm.org/10.1145/130385.130401) )、[随机森林](http://dx.doi.org/10.1023/A:1010933404324)和人工神经网络。

在**图像分类**的背景下，我们假设我们的图像数据集由图像本身以及它们对应的*类别标签*组成，我们可以使用它们来教导我们的机器学习分类器每个类别“看起来像什么”如果我们的分类器做出了不正确的预测，我们可以应用方法来纠正它的错误。

通过查看**表 1** 中的示例，可以更好地理解监督学习、非监督学习和半监督学习之间的区别。表格的第一列是与特定图像相关联的标签。其余六列对应于每个数据点的特征向量，这里，我们选择通过计算每个 RGB 颜色通道的平均值和标准偏差来量化图像内容。

| **标签** |  |  | ***B*** | ***R[σ]*** | ***G[σ]*** | ***B[σ]*** |
| --- | --- | --- | --- | --- | --- | --- |
| 猫 | Fifty-seven point six one | Forty-one point three six | One hundred and twenty-three point four four | One hundred and fifty-eight point three three | One hundred and forty-nine point eight six | Ninety-three point three three |
| 猫 | One hundred and twenty point two three | One hundred and twenty-one point five nine | One hundred and eighty-one point four three | One hundred and forty-five point five eight | Sixty-nine point one three | One hundred and sixteen point nine one |
| 猫 | One hundred and twenty-four point one five | One hundred and ninety-three point three five | Sixty-five point seven seven | Twenty-three point six three | One hundred and ninety-three point seven four | One hundred and sixty-two point seven |
| 狗 | One hundred point two eight | One hundred and sixty-three point eight two | One hundred and four point eight one | Nineteen point six two | One hundred and seventeen point zero seven | Twenty-one point one one |
| 狗 | One hundred and seventy-seven point four three | Twenty-two point three one | One hundred and forty-nine point four nine | One hundred and ninety-seven point four one | Eighteen point nine nine | One hundred and eighty-seven point seven eight |
| 狗 | One hundred and forty-nine point seven three | Eighty-seven point one seven | One hundred and eighty-seven point nine seven | Fifty point two seven | Eighty-seven point one five | Thirty-six point six five |

**Table 1:** A table of data containing both the class labels (either *dog* or *cat*) and feature vectors for each data point (the mean and standard deviation of each Red, Green, and Blue color channel, respectively). This is an example of a ***supervised classification***task.

我们的监督学习算法将对这些特征向量中的每一个进行预测，如果它做出不正确的预测，我们将试图通过告诉它正确的标签实际上是什么来纠正它。然后，该过程将继续，直到满足期望的停止标准，例如精度、学习过程的迭代次数或者仅仅是任意数量的墙壁时间。

***备注:*** 为了解释监督、非监督和半监督学习之间的差异，我选择了使用基于特征的方法(即 RGB 颜色通道的均值和标准差)来量化图像的内容。当我们开始使用卷积神经网络时，我们实际上会**跳过**特征提取步骤，使用原始像素亮度本身。由于图像可能是很大的矩阵(因此不能很好地适应这个电子表格/表格示例)，我使用了特征提取过程来帮助可视化不同类型的学习之间的差异。

### **无监督学习**

与监督学习相反，**无监督学习**(有时称为**自学学习**)没有与输入数据相关联的标签，因此如果它做出不正确的预测，我们就无法纠正我们的模型。

回到电子表格的例子，将监督学习问题转换为非监督学习问题就像删除“标签”列一样简单(**表 2** )。

无监督学习有时被认为是机器学习和图像分类的“圣杯”。当我们考虑 Flickr 上的图片数量或 YouTube 上的视频数量时，我们很快意识到互联网上有大量未标记的数据。如果我们可以让我们的算法从*未标记数据*中学习模式，那么我们就不必花费大量的时间(和金钱)费力地为监督任务标记图像。

|  |  | ***B*** | ***R[σ]*** | ***G[σ]*** | ***B[σ]*** |
| --- | --- | --- | --- | --- | --- |
| Fifty-seven point six one | Forty-one point three six | One hundred and twenty-three point four four | One hundred and fifty-eight point three three | One hundred and forty-nine point eight six | Ninety-three point three three |
| One hundred and twenty point two three | One hundred and twenty-one point five nine | One hundred and eighty-one point four three | One hundred and forty-five point five eight | Sixty-nine point one three | One hundred and sixteen point nine one |
| One hundred and twenty-four point one five | One hundred and ninety-three point three five | Sixty-five point seven seven | Twenty-three point six three | One hundred and ninety-three point seven four | One hundred and sixty-two point seven |
| One hundred point two eight | One hundred and sixty-three point eight two | One hundred and four point eight one | Nineteen point six two | One hundred and seventeen point zero seven | Twenty-one point one one |
| One hundred and seventy-seven point four three | Twenty-two point three one | One hundred and forty-nine point four nine | One hundred and ninety-seven point four one | Eighteen point nine nine | One hundred and eighty-seven point seven eight |
| One hundred and forty-nine point seven three | Eighty-seven point one seven | One hundred and eighty-seven point nine seven | Fifty point two seven | Eighty-seven point one five | Thirty-six point six five |

**Table 2:** Unsupervised learning algorithms attempt to learn underlying patterns in a dataset *without* class labels. In this example we have removed the class label column, thus turning this task into an ***unsupervised learning*** problem.

当我们可以学习数据集的底层结构，然后反过来将我们学习到的特征应用于一个*监督*学习问题时，大多数无监督学习算法都是最成功的，在这种情况下，只有很少的标记数据可供使用。

用于无监督学习的经典机器学习算法包括主成分分析(PCA)和 k-means 聚类。具体到神经网络，我们看到自动编码器，自组织映射(SOMs)，以及应用于无监督学习的自适应共振理论。无监督学习是一个非常活跃的研究领域，也是一个尚未解决的问题。在本书中，我们不关注无监督学习。

### **半监督学习**

那么，如果我们只有*一些*与我们的数据相关联的标签，而*没有标签*与其他的相关联，会发生什么呢？有没有一种方法可以应用监督和非监督学习的混合，并且仍然能够对每个数据点进行分类？结果答案是*是的*——我们只需要应用半监督学习。

回到我们的电子表格示例，假设我们只有一小部分输入数据的标签(**表 3** )。我们的半监督学习算法将获取已知的数据，分析它们，并尝试标记每个未标记的数据点，以用作*额外的*训练数据。随着半监督算法学习数据的“结构”以做出更准确的预测并生成更可靠的训练数据，该过程可以重复多次。

| **标签** |  |  | ***B*** | ***R[σ]*** | ***G[σ]*** | ***B[σ]*** |
| --- | --- | --- | --- | --- | --- | --- |
| 猫 | Fifty-seven point six one | Forty-one point three six | One hundred and twenty-three point four four | One hundred and fifty-eight point three three | One hundred and forty-nine point eight six | Ninety-three point three three |
| ？ | One hundred and twenty point two three | One hundred and twenty-one point five nine | One hundred and eighty-one point four three | One hundred and forty-five point five eight | Sixty-nine point one three | One hundred and sixteen point nine one |
| ？ | One hundred and twenty-four point one five | One hundred and ninety-three point three five | Sixty-five point seven seven | Twenty-three point six three | One hundred and ninety-three point seven four | One hundred and sixty-two point seven |
| 狗 | One hundred point two eight | One hundred and sixty-three point eight two | One hundred and four point eight one | Nineteen point six two | One hundred and seventeen point zero seven | Twenty-one point one one |
| ？ | One hundred and seventy-seven point four three | Twenty-two point three one | One hundred and forty-nine point four nine | One hundred and ninety-seven point four one | Eighteen point nine nine | One hundred and eighty-seven point seven eight |
| 狗 | One hundred and forty-nine point seven three | Eighty-seven point one seven | One hundred and eighty-seven point nine seven | Fifty point two seven | Eighty-seven point one five | Thirty-six point six five |

**Table 3:** When performing ***semi-supervised learning*** we only have the labels for a subset of the images/feature vectors and must try to label the other data points to utilize them as extra training data.

半监督学习在计算机视觉中特别有用，在计算机视觉中，标记我们训练集中的每一幅图像通常是耗时、乏味和昂贵的(至少在工时方面)。在我们根本没有时间或资源来标记每张图像的情况下，我们可以只标记一小部分数据，并利用半监督学习来标记和分类其余的图像。

半监督学习算法通常用较小的标记输入数据集来换取分类精度的某种可容忍的降低。通常情况下，监督学习算法的标签训练越准确，它可以做出的预测就越准确(对于深度学习算法来说，*尤其是*是这样)。

随着训练数据量的减少，准确性不可避免地受到影响。半监督学习将准确性和数据量之间的这种关系考虑在内，并试图将分类准确性保持在可容忍的范围内，同时大幅减少构建模型所需的训练数据量-最终结果是一个准确的分类器(但通常不如监督分类器准确)，只需较少的工作和训练数据。半监督学习的流行选择包括[标签传播](https://www.google.com/books/edition/Advances_in_Neural_Information_Processin/0F-9C7K8fQ8C?hl=en&gbpv=1&dq=Learning+with+Local+and+Global+Consistency&pg=PA321&printsec=frontcover)、[标签传播](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf)、[梯形网络](http://arxiv.org/abs/1507.02672)和[共同学习/共同训练](http://doi.acm.org/10.1145/279943.279962)。

同样，我们将主要关注本书中的监督学习，因为在计算机视觉深度学习的背景下，无监督和半监督学习仍然是非常活跃的研究主题，没有关于使用哪种方法的明确指南。****