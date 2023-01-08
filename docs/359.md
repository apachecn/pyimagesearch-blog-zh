# 我最喜欢的 9 个构建图像搜索引擎的 Python 库

> 原文：<https://pyimagesearch.com/2014/01/12/my-top-9-favorite-python-libraries-for-building-image-search-engines/>

八年前，当我第一次对计算机视觉和图像搜索引擎感兴趣时，我不知道从哪里开始。我不知道使用哪种语言，不知道安装哪些库，我找到的库也不知道如何使用。我希望有一个像这样的列表，详细列出用于图像处理、计算机视觉和图像搜索引擎的最佳库。

这份清单绝不是完整的或详尽的。这只是我最喜欢的 Python 库，我每天都用它来做计算机视觉和图像搜索引擎。如果你认为我漏掉了一个重要的，请在评论中给我留言或者给我发电子邮件。

## **对于新手:**

### 1. [NumPy](http://www.numpy.org/)

NumPy 是 Python 编程语言的一个库，它提供了对大型多维数组的支持。为什么这很重要？使用 NumPy，我们可以将图像表示为多维数组。例如，假设我们从 Google 下载了我们最喜欢的暴躁猫图片,现在我们想用 NumPy 把它表示成一个数组。这幅图像是 RGB 颜色空间中的 *452×589* 像素，其中每个像素由三个分量组成:红色值、绿色值和蓝色值。每个像素值 *p* 在范围 *0 < = p < = 255* 内。这样我们就有了一个 452×589 像素的矩阵。使用 NumPy，我们可以方便地将该图像存储在一个 *(452，593，3)* 数组中，该数组有 452 行、593 列和 3 个值，每个值对应 RGB 颜色空间中的一个像素。将图像表示为 NumPy 数组不仅计算和资源效率高，而且许多其他图像处理和机器学习库也使用 NumPy 数组表示。此外，通过使用 NumPy 内置的高级数学函数，我们可以快速地对图像进行数值分析。

### 2.[轨道](http://scipy.org/)轨道

与 NumPy 齐头并进，我们还有 SciPy。SciPy 增加了对科学和技术计算的进一步支持。我最喜欢的 SciPy 子包之一是[空间包](http://docs.scipy.org/doc/scipy/reference/spatial.html)，它包含了大量的距离函数和 kd-tree 实现。为什么距离函数很重要？当我们“描述”一幅图像时，我们执行特征提取。通常在特征提取之后，图像由一个向量(一列数字)来表示。为了比较两幅图像，我们依赖于距离函数，例如欧几里德距离。为了比较两个任意的特征向量，我们简单地计算它们的特征向量之间的距离。在欧几里德距离的情况下，距离越小，两幅图像越“相似”。

### 3. [matpotlib](http://matplotlib.org/)

简单来说，matplotlib 就是一个绘图库。如果你以前用过 MATLAB，你可能会觉得在 matplotlib 环境中很舒服。当分析图像时，我们将利用 matplotlib，无论是绘制搜索系统的整体准确性还是简单地查看图像本身，matplotlib 都是您工具箱中的一个伟大工具。

### 4. [PIL](http://www.pythonware.com/products/pil/) 和[枕头](http://pillow.readthedocs.org/en/latest/)

我并不反对 PIL 或枕头，不要误会我，他们非常擅长做什么:简单的图像操作，如调整大小，旋转等。总的来说，我只是觉得语法笨拙。也就是说，许多非科学的 Python 项目使用了 PIL 或 Pillow。例如，Python web 框架 [Django](https://www.djangoproject.com/) 使用 PIL 来表示数据库中的图像字段。如果你需要做一些快速和肮脏的图像操作，PIL 和 Pillow 有他们的位置，但是如果你认真学习图像处理、计算机视觉和图像搜索引擎，我会*高度*推荐你花时间玩 OpenCV 和 SimpleCV。

## **我的首选:**

### 5. [OpenCV](http://opencv.org/)

如果 NumPy 的主要目标是大型、高效、多维数组表示，那么，到目前为止，OpenCV 的主要目标是实时图像处理。这个库从 1999 年就存在了，但是直到 2009 年的 2.0 版本我们才看到令人难以置信的 NumPy 支持。库本身是用 C/C++编写的，但是 Python 绑定是在运行安装程序时提供的。OpenCV 无疑是我最喜欢的计算机视觉库，但它确实有一个学习曲线。准备好花大量的时间学习库的复杂性和浏览文档(现在已经增加了 NumPy 支持，这已经变得非常好了)。如果您仍然在测试计算机视觉 waters，您可能想要查看下面提到的 SimpleCV 库，它的学习曲线要小得多。

### 6\. [SimpleCV](http://simplecv.org/)

SimpleCV 的目标是让你尽快涉足图像处理和计算机视觉。他们在这方面做得很好。学习曲线比 OpenCV 小得多，正如他们的标语所说，“让计算机视觉变得简单”。也就是说，因为学习曲线更短，所以您无法接触到 OpenCV 提供的许多原始的、强大的技术。如果你只是在试水，一定要试试这个库。然而，作为一个公平的警告，我在这个博客中超过 95%的代码示例将使用 OpenCV。不过不要担心，我对文档绝对一丝不苟，我会为您提供完整而简洁的代码解释。

### 7.穆罕默德

Mahotas 就像 OpenCV 和 SimpleCV 一样，依赖于 NumPy 数组。在 Mahotas 中实现的许多功能可以在 OpenCV 和/或 SimpleCV 中找到，但在某些情况下，Mahotas 接口更容易使用，尤其是当涉及到他们的[特性](http://mahotas.readthedocs.org/en/latest/features.html)包时。

### 8\. [scikit-learn](http://scikit-learn.org/stable/)

好吧，我明白了，Scikit-learn 不是一个图像处理或计算机视觉库——它是一个机器学习库。也就是说，如果没有某种机器学习，你就不可能拥有先进的计算机视觉技术，无论是聚类、矢量量化、分类模型等等。Scikit-learn 还包括一些图像特征提取功能。

### 9.[统计数据](http://www.ilastik.org/)

我实话实说。我从来没用过 ilastik。但是通过我在计算机视觉会议上的经历，我遇到了相当多这样做的人，所以我觉得有必要把它放在这个列表中。Ilastik 主要用于图像分割和分类，特别面向科学界。

[![Practical Python and OpenCV](img/d1b26971128a0e32c30b738919f93c47.png)](https://pyimagesearch.com/practical-python-opencv/?src=in-post-top-9-favorite-libraries)

## **奖金:**

我不能在九点就停下来。这里是我一直在使用的另外三个额外的库。

### 10.[过程](https://pypi.python.org/pypi/pprocess)

从图像中提取特征本质上是一项可并行化的任务。通过使用多处理库，可以减少从整个数据集中提取要素所需的时间。我最喜欢的是 process，因为我需要它的简单性质，但是你可以使用你最喜欢的。

### 11. [h5py](http://www.h5py.org/)

h5py 库是 Python 中存储大型数值数据集的事实上的标准。最精彩的部分？它支持 NumPy 数组。因此，如果您有一个表示为 NumPy 数组的大型数据集，并且它不适合内存，或者如果您希望高效、持久地存储 NumPy 数组，那么 h5py 是一个不错的选择。我最喜欢的技术之一是将我提取的特征存储在 h5py 数据集中，然后应用 scikit-learn 的 [MiniBatchKMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html) 对这些特征进行聚类。整个数据集不必一次全部从磁盘上加载，内存占用也非常小，即使对于成千上万的特征向量也是如此。

### 12. [scikit-image](http://scikit-image.org/) [](http://www.h5py.org/) 

在这篇博文的初稿上，我完全忘记了 scikit-image。我真傻。无论如何，scikit-image 很棒，但是你必须知道你在做什么来有效地使用这个库——我不是指“有一个陡峭的学习曲线”类型的方式。学习曲线实际上相当低，尤其是如果你查看他们的[画廊](http://scikit-image.org/docs/dev/auto_examples/)。scikit-image 中包含的算法(我认为)更接近计算机视觉的最先进水平。在 scikit-image 中可以找到来自学术论文的新算法，但是为了(有效地)使用这些算法，您需要在计算机视觉领域有一些严谨和理解。如果你已经有一些计算机视觉和图像处理的经验，一定要看看 scikit-image；否则，我会继续使用 OpenCV 和 SimpleCV。

## **总之:**

NumPy 为您提供了一种将图像表示为多维数组的方法。许多其他图像处理、计算机视觉和机器学习库都使用 NumPy，因此安装它(和 SciPy)非常重要。虽然 PIL 和 Pillow 非常适合简单的图像处理任务，但如果你认真测试计算机视觉水域，你的时间最好花在玩 SimpleCV 上。一旦你确信计算机视觉很棒，安装 OpenCV 并重新学习你在 SimpleCV 中所做的。我将在博客中展示的 95%以上的代码都是 OpenCV。最后安装 scikit-learn 和 h5py。你现在还不需要它们。但是一旦我向你展示了他们的能力，你就会爱上他们。