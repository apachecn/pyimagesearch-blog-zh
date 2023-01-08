# 构建图像搜索引擎:定义相似性度量(第 3 步，共 4 步)

> 原文：<https://pyimagesearch.com/2014/02/17/building-an-image-search-engine-defining-your-similarity-metric-step-3-of-4/>

首先，让我们快速回顾一下。

两周前，我们探索了构建图像搜索引擎的第一步:定义你的图像描述符。我们探索了图像的三个容易描述的方面:颜色、纹理和形状。

从那里，我们继续到[步骤 2:索引你的数据集](https://pyimagesearch.com/2014/02/10/building-an-image-search-engine-indexing-your-dataset-step-2-of-4/ "Building an Image Search Engine: Indexing Your Dataset (Step 2 of 4)")。索引是通过应用图像描述符从数据集中的每个图像提取特征来量化数据集的过程。

索引也是一项可以轻松实现并行的任务——如果我们的数据集很大，我们可以通过在机器上使用多个内核/处理器来轻松加速索引过程。

最后，不管我们是使用串行还是并行处理，我们都需要将得到的特征向量写入磁盘以备后用。

现在，是时候进入构建图像搜索引擎的第三步了:**定义你的相似性度量**

# 定义您的相似性度量

今天我们将对不同种类的距离和相似性度量进行粗略的回顾，我们可以用它们来比较两个特征向量。

***注意:**根据你使用的距离函数，在这个过程中你需要注意很多很多的“陷阱”。我将回顾每一个距离函数，并在本博客的后面提供如何正确使用它们的例子，但不要在没有首先理解特征向量应该如何缩放、归一化等的情况下盲目地将距离函数应用于特征向量。，否则您可能会得到奇怪的结果。*

那么距离度量和相似性度量之间有什么区别呢？

为了回答这个问题，我们首先需要定义一些变量。

设`d`是我们的距离函数，并且`x`、`y`和`z`是实值特征向量，那么必须满足以下条件:

1.  **非否定性:** `d(x, y) >= 0`。这仅仅意味着我们的距离必须是非负的。
2.  **重合公理:** `d(x, y) = 0`当且仅当`x = y`。只有当两个向量具有相同的值时，距离为零(意味着向量相同)才是可能的。
3.  **对称:** `d(x, y) = d(y, x)`。为了使我们的距离函数被认为是一个距离度量，距离中参数的顺序应该无关紧要。指定`d(x, y)`而不是`d(y, x)`对我们的距离度量没有影响，两个函数调用应该返回相同的值。
4.  **三角形不等式:**:T0。你还记得你高中的三角函数课吗？所有这些条件表明，任意两条边的长度之和必须大于剩余的一条边。

如果所有四个条件都成立，那么我们的距离函数可以被认为是距离度量。

那么这是否意味着我们应该*只*使用距离度量，而忽略其他类型的相似性度量呢？

当然不是！

但是理解术语是很重要的，尤其是当你开始自己构建图像搜索引擎的时候。

让我们来看看五个比较流行的距离度量和相似性函数。我包含了该函数对应的 SciPy 文档的链接，以防您想自己使用这些函数。

*   **[欧几里得:](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html#scipy.spatial.distance.euclidean)** 可以说是最广为人知且必须使用的距离度量。欧几里得距离通常被描述为两点之间的“直线”距离。
*   [**曼哈顿:**](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cityblock.html#scipy.spatial.distance.cityblock) 也称“城市街区”。想象你自己在一辆出租车里，沿着城市街区转来转去，直到你到达目的地。
*   [**切比雪夫:**](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.chebyshev.html#scipy.spatial.distance.chebyshev) 任意单一维度上的点之间的最大距离。
*   [**余弦:**](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html#scipy.spatial.distance.cosine) 在进入向量空间模型、tf-idf 加权、高维正空间之前，我们不会太多地使用这个相似度函数，但是余弦相似度函数极其重要。值得注意的是，余弦相似性函数不是一个合适的距离度量——它违反了三角形不等式和重合公理。
*   [**汉明:**](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.hamming.html#scipy.spatial.distance.hamming) 给定两个(通常为二进制)向量，汉明距离衡量两个向量之间“不一致”的次数。两个相同的向量将没有分歧，因此完全相似。

这份清单绝非详尽无遗。有大量的距离函数和相似性度量。但是在我们开始深入细节和探索何时何地使用每个距离函数之前，我只想提供一个关于构建图像搜索引擎的 10，000 英尺的概述。

现在，让我们计算一些距离:

```py
>>> from scipy.spatial import distance as dist
>>> import numpy as np
>>> np.random.seed(42)
>>> x = np.random.rand(4)
>>> y = np.random.rand(4)
>>> x
array([ 0.37454012,  0.95071431,  0.73199394,  0.59865848])
>>> y
array([ 0.15601864,  0.15599452,  0.05808361,  0.86617615])

```

我们要做的第一件事是导入我们需要的包:SciPy 的`distance`模块和 NumPy 本身。从这里开始，我们需要用一个显式值作为伪随机数生成器的种子。通过提供一个显式值(在本例中为 42)，它确保了如果您自己执行这段代码，您会得到与我相同的结果。

最后，我们生成我们的“特征向量”。这些是长度为 4 的实值列表，值在范围[0，1]内。

是时候计算一些实际距离了:

```py
>>> dist.euclidean(x, y)
1.0977486080871359
>>> dist.cityblock(x, y)
1.9546692556997436
>>> dist.chebyshev(x, y)
0.79471978607371352

```

那么这告诉我们什么呢？

欧几里得距离小于曼哈顿距离。直觉上，这是有道理的。欧几里得距离是“直线距离”(意思是可以走两点之间最短的路径，就像飞机从一个机场飞到另一个机场)。相反，曼哈顿距离更类似于驾车穿过城市街区——我们正在进行急转弯，就像在一张网格纸上行驶，因此曼哈顿距离更大，因为我们在两点之间行驶的时间更长。

最后，切比雪夫距离是矢量中任意两个分量之间的最大距离。在这种情况下，从*| 0.95071431–0.15599452 |*可以找到 *0.794* 的最大距离。

现在，让我们来看看海明距离:

```py
>>> x = np.random.random_integers(0, high = 1, size =(4,))
>>> y = np.random.random_integers(0, high = 1, size =(4,))
>>> x
array([1, 1, 1, 0])
>>> y
array([1, 0, 1, 1])
>>> dist.hamming(x, y)
0.5

```

在前面的例子中，我们有*个实值的*个特征向量。现在我们有了*二进制* 特征向量。汉明距离比较了`x`和`y`特征向量之间不匹配的数量。在这种情况下，它发现了两个不匹配—第一个是在`x[1]`和`y[1]`，第二个是在`x[3]`和`y[3]`。

假设我们有两个不匹配，并且向量的长度是 4，那么不匹配与向量长度的比率是 2 / 4 = 0.5，因此我们的汉明距离。

# 摘要

构建图像搜索引擎的第一步是[决定图像描述符](https://pyimagesearch.com/2014/02/03/building-an-image-search-engine-defining-your-image-descriptor-step-1-of-4/ "Building an Image Search Engine: Defining Your Image Descriptor (Step 1 of 4)")。从那里，可以将图像描述符应用于数据集中的每个图像，并提取一组特征。[这个过程被称为“索引数据集”](https://pyimagesearch.com/2014/02/10/building-an-image-search-engine-indexing-your-dataset-step-2-of-4/ "Building an Image Search Engine: Indexing Your Dataset (Step 2 of 4)")。为了比较两个特征向量并确定它们有多“相似”，需要相似性度量。

在这篇博文中，我们粗略地探索了距离和相似度函数，它们可以用来衡量两个特征向量有多“相似”。

流行的距离函数和相似性度量包括(但当然不限于):欧几里德距离、曼哈顿(城市街区)、切比雪夫、余弦距离和汉明。

在这篇博客的后面，我们不仅会更详细地探讨这些距离函数，我还会介绍更多，包括专门用于比较直方图的方法，如相关法、交集法、卡方法和土方距离法。

在这一点上，你应该有一个基本的想法是什么需要建立一个图像搜索引擎。开始使用简单的[颜色直方图](https://pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/)从图像中提取特征。然后使用上面讨论的距离函数对它们进行比较。记下你的发现。

最后，在下面注册我的时事通讯，以便在我发布新的图片搜索引擎文章时得到更新。作为回报，我会给你一份关于计算机视觉和图像搜索引擎的 11 页的资源指南。