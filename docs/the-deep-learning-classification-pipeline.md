# 深度学习分类管道

> 原文：<https://pyimagesearch.com/2021/04/17/the-deep-learning-classification-pipeline/>

根据我们之前关于图像分类和学习算法类型的两节，您可能会开始对新术语、考虑事项以及构建图像分类器时看起来难以克服的大量变化感到有些不知所措，但事实是，一旦您理解了过程，构建图像分类器是相当简单的。

在这一部分，我们将回顾在使用机器学习时，你需要在思维模式上进行的一个重要转变。在那里，我将回顾构建基于深度学习的图像分类器的四个步骤，并比较和对比传统的基于特征的机器学习和端到端深度学习。

### **心态的转变**

在我们进入任何复杂的事情之前，让我们从我们都(很可能)熟悉的东西开始:*斐波那契数列*。

斐波纳契数列是一系列的数字，序列的下一个数字是通过对它前面的两个整数求和得到的。例如，给定序列`0, 1, 1`，通过添加`1 + 1 = 2`找到下一个数字。同样，给定`0, 1, 1, 2`，序列中的下一个整数是`1 + 2 = 3`。

按照这种模式，序列中的前几个数字如下:

`0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...`

当然，我们也可以使用递归在(极未优化的)Python 函数中定义这种模式:

```py
>>> def fib(n):
...     if n == 0:
...             return 0
...     elif n == 1:
...             return 1
...     else:
...             return fib(n-1) + fib(n-2)
...
>>>
```

使用这段代码，我们可以通过向`fib`函数提供一个值`n`来计算序列中的第 n 个*。例如，让我们计算斐波纳契数列中的第 7 个数字:*

```py
>>> fib(7)
13
```

第 13 个数字:

```py
>>> fib(13)
233
```

最后是第 35 个数字:

```py
>>> fib(35)
9227465
```

如您所见，斐波纳契数列非常简单，是一系列函数的示例:

1.  接受输入，返回输出。
2.  该过程被很好地定义。
3.  很容易验证输出的正确性。
4.  非常适合代码覆盖和测试套件。

一般来说，在你的一生中，你可能已经编写了成千上万个这样的过程函数。无论您是计算斐波纳契数列，从数据库中提取数据，还是从一系列数字中计算平均值和标准偏差，这些函数都是定义良好的，并且很容易验证其正确性。

***可惜深度学习和图像分类就不是这样了！*T3****

注意图 1 中一只猫和一只狗的图片。现在，想象一下，试着写一个程序函数，它不仅能区分*和*这两张照片，还能区分*任何一张猫和狗的*照片。你将如何着手完成这项任务？你会检查各个( *x，y* )坐标的像素值吗？写几百条`if/else`语句？你将如何维护和验证如此庞大的基于规则的系统的正确性？简而言之就是:**你不**。

与编写算法来计算斐波纳契数列或对一系列数字进行排序不同，如何创建算法来区分猫和狗的图片并不直观或显而易见。因此，与其试图构建一个基于规则的系统来描述每个类别“看起来像什么”，不如我们可以采取一种*数据驱动的方法*，通过提供每个类别看起来像什么的*例子*，然后*教*我们的算法使用这些例子来识别类别之间的差异。

我们将这些例子称为我们的标签图像的*训练数据集*，其中我们的训练数据集中的每个数据点包括:

1.  一幅图像
2.  标签/类别(如狗、猫、熊猫等。)的图像

同样，这些图像中的每一个都有与之相关联的标签是很重要的，因为我们的监督学习算法需要看到这些标签，以“教会自己”如何识别每个类别。记住这一点，让我们继续完成构建深度学习模型的四个步骤。

### **第一步:收集数据集**

构建深度学习网络的第一个组成部分是收集我们的初始数据集。我们需要*图像本身*以及与每个图像相关联的*标签*。这些标签应该来自一组有限的类别，比如:`categories = dog, cat, panda`。

此外，每个类别的图像数量应该大致一致(即，每个类别的示例数量相同)。如果我们的猫图像的数量是狗图像的两倍，熊猫图像的数量是猫图像的五倍，那么我们的分类器将自然地偏向于过度适合这些大量表示的类别。

类别不平衡是机器学习中的一个常见问题，有许多方法可以克服它。我们将在本书的后面讨论其中的一些方法，但是请记住，避免由于班级失衡而导致的学习问题的最好方法是完全避免班级失衡。

### **第二步:分割你的数据集**

现在我们有了初始数据集，我们需要将它分成两部分:

1.  一个*训练集*
2.  一个*测试装置*

我们的分类器使用*训练集*通过对输入数据进行预测来“学习”每个类别的样子，然后在预测错误时进行自我纠正。训练好分类器后，我们可以在*测试集*上评估性能。

**训练集和测试集相互独立********不要重叠，这一点非常重要！*** 如果您使用测试集作为训练数据的一部分，那么您的分类器就具有不公平的优势，因为它之前已经看过测试示例并从中“学习”。相反，你必须将这个测试集与你的训练过程完全分开，并且只使用它来评估你的网络。***

 ***训练集和测试集的常见拆分大小包括 66。 6 %/33。 3 %，75%/25%，90%/10%(**图 2** ):

这些数据分割是有意义的，**但是如果您有要调整的参数呢？**神经网络有许多旋钮和杠杆(例如，学习率、衰减、正则化等。)需要被调谐和拨号以获得最佳性能。我们将这些类型的参数称为*超参数*，它们的正确设置是*关键*。

在实践中，我们需要测试一堆这些超参数，并确定最有效的参数集。您可能会尝试使用您的测试数据来调整这些值， ***,但是同样，这是一个大禁忌！*** 测试集*仅*用于评估你的网络性能。

相反，你应该创建一个名为**验证集**的*第三个*数据分割。这组数据(通常)来自训练数据，并被用作“假测试数据”，因此我们可以调整我们的超参数。只有在我们使用验证集确定了超参数值之后，我们才能继续收集测试数据中的最终准确性结果。

我们通常分配大约 10-20%的训练数据进行验证。如果将数据分割成块听起来很复杂，实际上并不复杂。正如我们将在下一章看到的，这非常简单，多亏了 scikit-learn 库，只需一行代码就能完成**。**

### **第三步:训练你的人际网络**

给定我们的训练图像集，我们现在可以训练我们的网络。这里的目标是让我们的网络学会如何识别我们的标签数据中的每个类别。当模型犯错误时，它会从这个错误中学习并改进自己。

那么，实际的“学习”是如何进行的呢？总的来说，我们应用一种梯度下降的形式，我们将在另一篇文章中详述。

### **步骤#4:评估**

最后，我们需要评估我们训练有素的网络。对于我们测试集中的每一幅图像，我们将它们呈现给网络，并要求它*预测*它认为图像的标签是什么。然后，我们将测试集中图像的模型预测列表。

最后，将这些*模型预测*与来自我们测试集的*地面实况*标签进行比较。地面实况标签代表图像类别*实际上是什么*。从那里，我们可以计算我们的分类器获得正确预测的数量，并计算诸如精度、召回和 f-measure 之类的聚合报告，这些报告用于量化我们的网络作为一个整体的性能。

### 基于特征的学习与深度学习在图像分类中的比较

在传统的基于特征的图像分类方法中，实际上在步骤#2 和步骤#3 之间插入了一个步骤——这个步骤是 ***特征提取*** 。在这个阶段，我们应用手工设计的算法，如[猪](http://dx.doi.org/10.1109/CVPR.2005.177)、 [LBPs](https://doi.org/10.1109/TPAMI.2002.1017623) 等。，基于我们想要编码的图像的特定成分(即，形状、颜色、纹理)来量化图像的内容。给定这些特征，然后我们继续训练我们的分类器并评估它。

当构建卷积神经网络时，我们实际上可以*跳过*特征提取步骤。这样做的原因是因为 CNN 是*端到端*模型。我们将原始输入数据(像素)呈现给网络。然后网络*学习隐藏层内的*过滤器，可以用来区分物体类别。然后，网络的输出是类别标签上的概率分布。

使用 CNN 的一个令人兴奋的方面是，我们不再需要对手工设计的功能大惊小怪——我们可以让我们的网络学习这些功能。然而，这种权衡是有代价的。训练 CNN 可能是一个不平凡的过程，所以要准备好花相当多的时间来熟悉自己的经验，并进行许多实验来确定什么可行，什么不可行。

### **当我的预测不正确时会发生什么？**

不可避免地，你会在你的训练集上训练一个深度学习网络，在你的测试集上评估它(发现它获得了很高的准确率)，然后将它应用到你的训练集和测试集都在之外的*——*的图像上，却发现网络表现很差*。*

这个问题被称为**一般化**，网络能够*一般化*并正确预测不存在于其训练或测试数据中的图像的类别标签。网络的泛化能力实际上是深度学习研究最重要的方面——如果我们可以训练网络，使其能够泛化到外部数据集，而无需重新训练或微调，我们将在机器学习方面取得巨大进步，使网络能够在各种领域中重复使用。网络的泛化能力将在本书中多次讨论，但我现在想提出这个话题，因为你将不可避免地遇到泛化问题，特别是当你了解深度学习的诀窍时。

不要因为你的模型不能正确分类图像而沮丧，考虑一下上面提到的一系列变化因素。您的训练数据集是否准确地反映了这些变化因素的例子？如果没有，你需要收集更多的训练数据(并阅读本书的其余部分，学习其他技术，以减少偏差和打击过度拟合)。

## **总结**

我们了解了什么是图像分类，以及为什么它对计算机来说是一项如此具有挑战性的任务(尽管人类似乎毫不费力地凭直觉做到了)。然后，我们讨论了机器学习的三种主要类型:监督学习、非监督学习和半监督学习。

最后，我们回顾了深度学习分类管道中的四个步骤。这些步骤包括收集数据集、将数据分为训练、测试和验证步骤、训练网络以及最终评估模型。

与传统的基于特征的方法不同，传统的方法需要我们利用手工制作的算法从图像中提取特征，图像分类模型，如卷积神经网络，是端到端的分类器，它在内部学习可用于区分图像类别的特征。****