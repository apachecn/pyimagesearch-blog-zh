# 在 Python 和 OpenCV 中实现 RootSIFT

> 原文：<https://pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/>

[![rootsift_extracted_example](img/ea1c224daee4ec4e19ed1bcc10f04922.png)](https://pyimagesearch.com/wp-content/uploads/2015/01/rootsift_extracted_example.jpg)

还在使用大卫·劳的原始、简单的 ole 实现吗？

好吧，根据 Arandjelovic 和 Zisserman 在他们 2012 年的论文 [*中所说的，每个人都应该知道的三件事来改善对象检索*](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf) ，你正在通过使用最初的实现来低估你自己(和你的准确性)。

相反，您应该利用 SIFT 的一个简单扩展，称为 RootSIFT，它可以用来显著提高对象识别的准确性、量化和检索的准确性。

无论是匹配关键点周围区域的描述符，使用 k-means 对 SIFT 描述符进行聚类，还是构建一个视觉单词包模型，RootSIFT 扩展都可以用来改善结果。

最棒的是，RootSIFT 扩展 ***位于原始 SIFT 实现****之上，不需要修改原始 SIFT 源代码。*

 *您不必重新编译或修改您最喜欢的 SIFT 实现来利用 RootSIFT 的优势。

因此，如果你经常在你的计算机视觉应用中使用 SIFT，但还没有升级到 RootSIFT，请继续阅读。

这篇博文将向您展示如何在 Python 和 OpenCV 中实现 RootSIFT 而无需(1)更改 OpenCV SIFT 原始实现中的一行代码,( 2)无需编译整个库。

听起来有趣吗？查看这篇博文的其余部分，了解如何在 Python 和 OpenCV 中实现 RootSIFT。

**OpenCV 和 Python 版本:**
为了运行这个例子，你需要 **Python 2.7** 和 **OpenCV 2.4.X** 。

# Why RootSIFT?

众所周知，在比较直方图时，欧几里德距离通常比使用卡方距离或 Hellinger 核产生的性能差[Arandjelovic 等人，2012]。

如果是这样的话，为什么我们在匹配关键点时经常使用欧几里德距离来比较 SIFT 描述符呢？或者聚类 SIFT 描述符形成码本？或者量化 SIFT 描述符形成视觉单词包？

请记住，虽然最初的 SIFT 论文讨论了使用欧几里德距离比较描述符，但 SIFT 本身仍然是一个直方图——难道其他距离度量标准不会提供更高的准确性吗？

原来，答案是**是**。代替使用不同的*度量*来比较 SIFT 描述符，我们可以代之以*直接修改*从 SIFT 返回的 128-dim 描述符。

你看，Arandjelovic 等人建议对 SIFT 描述符本身进行简单的代数扩展，称为 RootSIFT，它允许使用 Hellinger 核来“比较”SIFT 描述符，但仍然利用欧几里德距离。

下面是将 SIFT 扩展到 RootSIFT 的简单算法:

*   **步骤 1:** 使用你最喜欢的 SIFT 库计算 SIFT 描述符。
*   步骤 2: 对每个 SIFT 向量进行 L1 归一化。
*   **第三步:**取 SIFT 向量中每个元素的平方根。那么向量是 L2 归一化的。

就是这样！

这是一个简单的扩展。但这个小小的修改可以极大地改善结果，无论你是匹配关键点，聚集 SIFT 描述符，还是量化以形成一袋视觉单词，Arandjelovic 等人已经表明，RootSIFT 可以很容易地用于 SIFT 的所有场景，同时改善结果。

在这篇博文的剩余部分，我将展示如何使用 Python 和 OpenCV 实现 RootSIFT。使用这个实现，您将能够将 RootSIFT 整合到您自己的应用程序中——并改进您的结果！

# 在 Python 和 OpenCV 中实现 RootSIFT

打开您最喜欢的编辑器，创建一个新文件并命名为`rootsift.py`，让我们开始吧:

```py
# import the necessary packages
import numpy as np
import cv2

class RootSIFT:
	def __init__(self):
		# initialize the SIFT feature extractor
		self.extractor = cv2.DescriptorExtractor_create("SIFT")

	def compute(self, image, kps, eps=1e-7):
		# compute SIFT descriptors
		(kps, descs) = self.extractor.compute(image, kps)

		# if there are no keypoints or descriptors, return an empty tuple
		if len(kps) == 0:
			return ([], None)

		# apply the Hellinger kernel by first L1-normalizing and taking the
		# square-root
		descs /= (descs.sum(axis=1, keepdims=True) + eps)
		descs = np.sqrt(descs)
		#descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)

		# return a tuple of the keypoints and descriptors
		return (kps, descs)

```

我们要做的第一件事是导入我们需要的包。我们将使用 NumPy 进行数值处理，使用`cv2`进行 OpenCV 绑定。

然后我们在第 5 行的**上定义我们的`RootSIFT`类，在第 6-8 行**的**上定义构造器。构造函数简单地初始化 OpenCV SIFT 描述符提取器。**

第 10 行**上的`compute`函数处理 RootSIFT 描述符的计算。该函数需要两个参数和一个可选的第三个参数。**

`compute`函数的第一个参数是我们想要从中提取 RootSIFT 描述符的`image`。第二个参数是关键点或局部区域的列表，将从这些区域中提取 RootSIFT 描述符。最后，提供一个ε变量`eps`，以防止任何被零除的错误。

从那里，我们提取第 12 行上的原始 SIFT 描述符。

我们检查第 15 行**和第 16 行**——如果没有关键点或描述符，我们简单地返回一个空元组。

将原始 SIFT 描述符转换成 RootSIFT 描述符发生在第 20-22 行的**处。**

我们首先对`descs`数组中的每个向量进行 L1 归一化(**第 20 行**)。

从那里，我们得到 SIFT 向量中每个元素的平方根(**第 21 行**)。

最后，我们要做的就是将关键点元组和 RootSIFT 描述符返回给第 25 行的调用函数。

# 运行 RootSIFT

要真正看到 RootSIFT 的运行，打开一个新文件，命名为`driver.py`，我们将探索如何从图像中提取 SIFT 和 RootSIFT 描述符:

```py
# import the necessary packages
from rootsift import RootSIFT
import cv2

# load the image we are going to extract descriptors from and convert
# it to grayscale
image = cv2.imread("example.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect Difference of Gaussian keypoints in the image
detector = cv2.FeatureDetector_create("SIFT")
kps = detector.detect(gray)

# extract normal SIFT descriptors
extractor = cv2.DescriptorExtractor_create("SIFT")
(kps, descs) = extractor.compute(gray, kps)
print "SIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape)

# extract RootSIFT descriptors
rs = RootSIFT()
(kps, descs) = rs.compute(gray, kps)
print "RootSIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape)

```

在**的第 1 行和第 2 行**,我们导入了我们的`RootSIFT`描述符和 OpenCV 绑定。

然后，我们加载我们的示例图像，将其转换为灰度，并检测第 7-12 行上的高斯关键点的差异。

从那里，我们提取第 15-17 行上的原始 SIFT 描述符。

我们在第 20-22 行提取 RootSIFT 描述符。

要执行我们的脚本，只需发出以下命令:

```py
$ python driver.py

```

您的输出应该如下所示:

```py
SIFT: kps=1006, descriptors=(1006, 128) 
RootSIFT: kps=1006, descriptors=(1006, 128)

```

[![rootsift_extracted_example](img/ea1c224daee4ec4e19ed1bcc10f04922.png)](https://pyimagesearch.com/wp-content/uploads/2015/01/rootsift_extracted_example.jpg)

如你所见，我们已经提取了 1006 个狗关键点。对于每个关键点，我们提取了 128 维 SIFT 和 RootSIFT 描述符。

从这里开始，您可以将这个 RootSIFT 实现应用到您自己的应用程序中，包括关键点和描述符匹配、对描述符进行聚类以形成质心，以及量化以创建一个视觉单词包模型——所有这些我们将在以后的帖子中介绍。

# 摘要

在这篇博文中，我向您展示了如何扩展 David Lowe 最初的 OpenCV SIFT 实现来创建 RootSIFT 描述符，这是 Arandjelovic 和 Zisserman 在他们 2012 年的论文 *[中建议的一个简单扩展，这是每个人都应该知道的改进对象检索的三件事](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)。*

RootSIFT 扩展*不需要*修改您最喜欢的 SIFT 实现的源代码——它只是位于原始实现之上。

计算 RootSIFT 的简单的 ~~4 步~~ 3 步流程是:

*   **步骤 1:** 使用你最喜欢的 SIFT 库计算 SIFT 描述符。
*   步骤 2: 对每个 SIFT 向量进行 L1 归一化。
*   **第三步:**取 SIFT 向量中每个元素的平方根。然后向量被 L2 归一化

无论您是使用 SIFT 来匹配关键点，使用 k-means 形成聚类中心，还是量化 SIFT 描述符来形成一个视觉单词包，您都应该明确考虑使用 RootSIFT 而不是原始 SIFT 来提高对象检索的准确性。*