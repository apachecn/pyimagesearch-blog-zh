# PyTorch 是什么？

> 原文：<https://pyimagesearch.com/2021/07/05/what-is-pytorch/>

在本教程中，您将了解 PyTorch 深度学习库，包括:

*   PyTorch 是什么
*   如何在你的机器上安装 PyTorch
*   重要 PyTorch 功能，包括张量和亲笔签名
*   PyTorch 如何支持 GPU
*   PyTorch 为何如此受研究者欢迎
*   PyTorch 是否比 Keras/TensorFlow 更好
*   您是否应该在项目中使用 PyTorch 或 Keras/TensorFlow

此外，本教程是 PyTorch 基础知识五部分系列的第一部分:

1.  *py torch 是什么？*(今日教程)
2.  PyTorch 简介:使用 PyTorch 训练你的第一个神经网络(下周教程)
3.  *PyTorch:训练你的第一个卷积神经网络*
4.  *使用预训练网络的 PyTorch 图像分类*
5.  *使用预训练网络的 PyTorch 对象检测*

在本教程结束时，您将对 PyTorch 库有一个很好的介绍，并且能够与其他深度学习实践者讨论该库的优缺点。

**了解 PyTorch 深度学习库，** ***继续阅读。***

## **py torch 是什么？**

PyTorch 是一个开源的机器学习库，专门从事张量计算、自动微分和 GPU 加速。由于这些原因， **PyTorch 是*最受欢迎的*深度学习库**之一，与 Keras 和 TensorFlow 竞争“最常用”深度学习包的奖项:

PyTorch 趋向于*特别是*在研究社区中流行，因为它的 Pythonic 性质和易于扩展(例如，实现定制的层类型、网络架构等)。).

在本教程中，我们将讨论 PyTorch 深度学习库的基础知识。从下周开始，您将获得使用 PyTorch 训练神经网络、执行图像分类以及对图像和实时视频应用对象检测的实践经验。

让我们开始了解 PyTorch 吧！

### **PyTorch、深度学习和神经网络**

PyTorch 基于 Torch，一个用于 Lua 的科学计算框架。在 PyTorch 和 Keras/TensorFlow 之前，Caffe 和 Torch 等深度学习包往往最受欢迎。

然而，随着深度学习开始彻底改变计算机科学的几乎所有领域，开发人员和研究人员希望有一个高效、易用的库来用 Python 编程语言构建、训练和评估神经网络。

Python 和 R 是数据科学家和机器学习最受欢迎的两种编程语言，因此研究人员希望在他们的 Python 生态系统中使用深度学习算法是很自然的。

**谷歌人工智能研究员弗朗索瓦·乔莱(Franç ois Chollet)于 2015 年 3 月开发并发布了 Keras，这是一个开源库，提供了用于训练神经网络的 Python API。** Keras 因其易于使用的 API 而迅速受到欢迎，该 API 模仿了 scikit-learn 的大部分工作方式，*事实上的*Python 的标准机器学习库。

很快，谷歌在 2015 年 11 月发布了第一个 TensorFlow 版本。 TensorFlow 不仅成为 Keras 库的默认后端/引擎，还实现了高级深度学习实践者和研究人员创建最先进的网络和进行新颖研究所需的许多低级功能。

然而，有一个问题 TensorFlow v1.x API 不是很 Pythonic 化，也不是很直观和易于使用。为了解决这个问题，PyTorch 于 2016 年 9 月发布，由脸书赞助，Yann LeCun(现代神经网络复兴的创始人之一，脸书的人工智能研究员)支持。

PyTorch 解决了研究人员在 Keras 和 TensorFlow 中遇到的许多问题。虽然 Keras 非常容易使用，但就其本质和设计而言，Keras 并没有公开研究人员需要的一些低级功能和定制。

另一方面，TensorFlow *当然*提供了对这些类型函数的访问，但它们不是 Pythonic 式的，而且通常很难梳理 TensorFlow 文档来找出*到底需要什么函数。**简而言之，Keras 没有提供研究人员需要的低级 API，TensorFlow 的 API 也不那么友好。***

PyTorch 通过创建一个既 Pythonic 化又易于定制的 API 解决了这些问题，允许实现新的层类型、优化器和新颖的架构。研究小组慢慢开始接受 PyTorch，从 TensorFlow 转变过来。本质上，这就是为什么今天你会看到这么多研究人员在他们的实验室里使用 PyTorch。

**也就是说，自从 PyTorch 1.x 和 TensorFlow 2.x 发布以来，各自库的 API 基本上已经***(双关语)。PyTorch 和 TensorFlow 现在实现了本质上相同的功能，并提供 API 和函数调用来完成相同的事情*

 *这种说法甚至得到了伊莱·史蒂文斯、卢卡·安提卡和托马斯·维赫曼的支持，他们在《PyTorch》一书中写道:

> *有趣的是，随着 TorchScript 和 eager mode 的出现，****py torch 和 TensorFlow 都已经看到它们的功能集开始与对方的*** *融合，尽管两者之间这些功能的呈现和整体体验仍然有很大不同。*
> 
> *— [Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf) (Chapter 1, Section 1.3.1, Page 9)*

我在这里的观点是不要陷入 PyTorch 或 Keras/TensorFlow 哪个“更好”的争论中——这两个库实现了非常相似的特性，只是使用了不同的函数调用和不同的训练范式。

如果你是深度学习的初学者，不要陷入哪个库更好的争论(有时是敌意的)中是正确的。正如我在本教程后面讨论的，你最好*选择一个并学习它。不管你用 PyTorch 还是 Keras/TensorFlow，深度学习的基础都是一样的。*

### 如何安装 PyTorch？

PyTorch 库可以使用 Python 的包管理器 pip 安装:

```py
$ pip install torch torchvision
```

从那里，您应该启动一个 Python shell 并验证您可以导入`torch`和`torchvision`:

```py
$ python
>>> import torch
>>> torch.__version__
'1.8.1'
>>> 
```

恭喜你现在已经在你的机器上安装了 PyTorch！

***注:*** *需要帮忙安装 PyTorch？要开始，* [*一定要查阅 PyTorch 官方文档*](https://pytorch.org/get-started/locally/) *。不然你可能会对我预装 PyTorch 自带的* [*PyImageSearch 大学*](https://pyimagesearch.com/pyimagesearch-university/) *里面预配置的 Jupyter 笔记本感兴趣。*

### PyTorch 和 Tensors

PyTorch 将数据表示为多维的类似 NumPy 的数组，称为**张量。张量将输入存储到你的神经网络、隐藏层表示和输出中。**

下面是一个用 NumPy 初始化数组的例子:

```py
>>> import numpy as np
>>> np.array([[0.0, 1.3], [2.8, 3.3], [4.1, 5.2], [6.9, 7.0]])
array([[0\. , 1.3],
       [2.8, 3.3],
       [4.1, 5.2],
       [6.9, 7\. ]])
```

我们可以使用 PyTorch 初始化同一个数组，如下所示:

```py
>>> import torch
>>> torch.tensor([[0.0, 1.3], [2.8, 3.3], [4.1, 5.2], [6.9, 7.0]])
tensor([[0.0000, 1.3000],
        [2.8000, 3.3000],
        [4.1000, 5.2000],
        [6.9000, 7.0000]])
```

这看起来没什么大不了的，但是在幕后，PyTorch 可以从这些张量动态生成一个图形，然后在其上应用自动微分:

### **PyTorch 的亲笔签名特稿**

说到自动微分，PyTorch 让使用`torch.autograd`训练神经网络变得超级简单。

在引擎盖下，PyTorch 能够:

1.  组装一个神经网络图
2.  执行向前传递(即，进行预测)
3.  计算损失/误差
4.  向后遍历网络*(即反向传播)并调整网络参数，使其(理想情况下)基于计算的损耗/输出做出更准确的预测*

 *第 4 步总是手工实现的最乏味和耗时的步骤。幸运的是，PyTorch 会自动完成这一步。

***注意:*** *Keras 用户通常只需调用`model.fit`来训练网络，而 TensorFlow 用户则利用`GradientTape`类。PyTorch 要求我们手工实现我们的训练循环，所以`torch.autograd`在幕后为我们工作的事实是一个巨大的帮助。感谢 PyTorch 开发人员实现了自动微分，这样您就不必这么做了。*

### **PyTorch 和 GPU 支持**

PyTorch 库主要支持基于 NVIDIA CUDA 的 GPU。GPU 加速允许您在很短的时间内训练神经网络。

此外，PyTorch 支持*分布式训练*，可以让你更快地训练你的模型。

### **py torch 为什么受科研人员欢迎？**

PyTorch 在 2016 年(PyTorch 发布的时间)到 2019 年(TensorFlow 2.x 正式发布之前)之间在研究社区获得了立足点。

PyTorch 能够获得这个立足点的原因有很多，但最主要的原因是:

*   Keras 虽然非常容易使用，但并没有提供研究人员进行新颖的深度学习研究所需的低级功能
*   同样，Keras 使得研究人员很难实现他们自己的定制优化器、层类型和模型架构
*   TensorFlow 1.x *有没有*提供这种底层访问和自定义实现；然而，这个 API 很难使用，也不是很 Pythonic 化
*   PyTorch，特别是其亲笔签名的支持，帮助解决了 TensorFlow 1.x 的许多问题，使研究人员更容易实现他们自己的自定义方法
*   此外，PyTorch 让深度学习实践者*完全控制*训练循环

这两者之间当然有分歧。Keras 使得使用对`model.fit`的单个调用来训练神经网络变得微不足道，类似于我们如何在 scikit-learn 中训练标准机器学习模型。

缺点是研究人员无法(轻易地)修改这个`model.fit`调用，所以他们不得不使用 TensorFlow 的底层函数。但是这些方法并不容易让他们实施他们的训练程序。

PyTorch 解决了这个问题，从我们完全控制的意义上来说，这是好的(T1)，但坏的(T3)是*，因为我们可以*轻易地*用 PyTorch 搬起石头砸自己的脚(*每个* PyTorch 用户以前都忘了将渐变归零)。*

尽管如此，关于 PyTorch 和 TensorFlow 哪个更适合研究的争论开始平息。PyTorch 1.x 和 TensorFlow 2.x APIs 实现了非常相似的特性，它们只是以不同的方式实现，有点像学习一种编程语言而不是另一种。每种编程语言都有其优点，但两者都实现了相同类型的语句和控制(即“if”语句、“for”循环等)。).

### **py torch 比 TensorFlow 和 Keras 好吗？**

**这个问题问错了，** ***尤其是*** **如果你是深度学习的新手。谁也不比谁强。** Keras 和 TensorFlow 有特定的用途，就像 PyTorch 一样。

例如，你不会一概而论地说 Java 绝对比 Python 好。当处理机器学习和数据科学时，有一个强有力的论点是 Python 优于 Java。但是如果您打算开发运行在多种高可靠性架构上的企业应用程序，那么 Java 可能是更好的选择。

不幸的是，一旦我们对某个特定的阵营或团体变得忠诚，我们人类就会变得“根深蒂固”。**围绕 PyTorch 与 Keras/TensorFlow 的斗争有时会变得很难看**，这曾促使 Keras 的创始人弗朗索瓦·乔莱(Franç ois Chollet)要求 PyTorch 用户停止向他发送仇恨邮件:

仇恨邮件也不仅限于弗朗索瓦。我在 PyImageSearch 上的大量深度学习教程中使用了 Keras 和 TensorFlow，我很难过地报告，我收到了批评我使用 Keras/TensorFlow 的仇恨邮件，称我愚蠢/愚蠢，告诉我关闭 PyImageSearch，我不是“真正的”深度学习实践者(不管这是什么意思)。

我相信其他教育工作者也经历过类似的行为，不管他们是用 Keras/TensorFlow 还是 PyTorch 写的教程。双方都变得丑陋，这不仅限于 PyTorch 用户。

我在这里的观点是，你不应该变得如此根深蒂固，以至于你根据别人使用的深度学习库来攻击他们。说真的，*世界上有*更重要的问题值得你关注——你真的不需要使用你的电子邮件客户端或社交媒体平台上的回复按钮来煽动和催化更多的仇恨进入我们已经脆弱的世界。

**其次，如果你是深度学习的新手，那么从哪个库开始并不重要*****。PyTorch 1.x 和 TensorFlow 2.x 的 API 已经融合——两者都实现了相似的功能，只是实现方式不同。***

 *你在一个库中学到的东西会转移到另一个库中，就像学习一门新的编程语言一样。你学习的第一种语言通常是最难的，因为你不仅要学习该语言的*语法*，还要学习控制结构和程序设计。

你的第二编程语言通常容易学习一个数量级，因为到那时你已经理解了控制和程序设计的基础。

**深度学习库也是如此。**随便挑一个就学会了。如果你在挑选上有困难，抛硬币——这*真的*没关系，你的经验会转移。

### **是否应该用 PyTorch 代替 TensorFlow/Keras？**

正如我在这篇文章中多次提到的，在 Keras/TensorFlow 和 PyTorch 之间进行选择并不涉及做出笼统的陈述，例如:

*   “如果你在做研究，你绝对应该使用 PyTorch。”
*   如果你是初学者，你应该使用 Keras
*   “如果你正在开发一个行业应用，使用 TensorFlow 和 Keras。”

PyTorch/Keras 和 TensorFlow 之间的许多功能集是融合的——两者本质上包含相同的功能集，只是以不同的方式实现。

**如果你是全新的深度学习，** ***随便挑一个就学会了。*** 我个人*做*认为 Keras 是最适合教初露头角的深度学习从业者。我*也*认为 Keras 是快速原型化和部署深度学习模型的最佳选择。

也就是说，PyTorch *确实让更高级的实践者更容易实现定制的训练循环、层类型和架构。现在 TensorFlow 2.x API 出来了，这种争论有所减弱，但我相信它仍然值得一提。*

最重要的是，无论你使用或选择学习什么深度学习库，不要成为一个狂热分子，不要 troll 留言板，总的来说，不要造成问题。这个世界上的仇恨已经够多了——作为一个科学界，我们应该超越仇恨邮件和揪头发。

## **总结**

在本教程中，您了解了 PyTorch 深度学习库，包括:

*   PyTorch 是什么
*   如何在你的机器上安装 PyTorch
*   PyTorch GPU 支持
*   PyTorch 为什么在研究界受欢迎
*   在项目中是使用 PyTorch 还是 Keras/TensorFlow

下周，您将通过实现和训练您的第一个神经网络，获得一些使用 PyTorch 的实践经验。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******