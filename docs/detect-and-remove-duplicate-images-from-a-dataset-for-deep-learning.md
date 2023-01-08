# 从用于深度学习的数据集中检测并移除重复图像

> 原文：<https://pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/>

在本教程中，您将学习如何从深度学习的数据集中检测和删除重复图像。

在过去的几周里，我一直在和 Victor Gevers 合作一个项目，他是 GDI 受人尊敬的道德黑客。该组织以负责任地披露数据泄露和报告安全漏洞而闻名。

我不能进入项目的细节(还不能)，但其中一项任务要求我训练一个定制的深度神经网络来检测图像中的特定模式。

我所使用的数据集是由多个来源的图像组合而成的。我*知道*数据集中会有重复的图像——**因此我需要一种方法来*检测*和*从我的数据集中移除*这些重复的图像。**

当我在做这个项目时，我碰巧收到了一封来自 Dahlia 的电子邮件，她是一名大学生，也有一个关于图像重复以及如何处理它们的问题:

> 嗨，阿德里安，我的名字是大丽花。我是一名本科生，正在进行最后一年的毕业设计，任务是通过抓取谷歌图片、必应等建立一个图像数据集。然后在数据集上训练深度神经网络。
> 
> 我的教授告诉我在构建数据集时要小心，说我需要移除重复的图像。
> 
> 这引起了我的一些怀疑:
> 
> **为什么数据集中的重复图像是一个问题？其次，我如何检测图像副本？**
> 
> 试图手动这样做听起来像是一个容易出错的过程。我不想犯任何错误。
> 
> **有什么办法可以让我自动*从我的图像数据集中检测并删除重复的图像吗？***
> 
> *谢谢你。*

 *大丽花问了一些很棒的问题。

数据集中存在重复影像会产生问题，原因有两个:

1.  它将*偏差*引入到你的数据集中，给你的深度神经网络额外的机会来学习模式*特定的*到副本
2.  这损害了你的模型对超出其训练范围的新图像进行概括的能力

虽然我们经常假设数据集中的数据点是独立且同分布的，但在处理真实世界的数据集时，这种情况*很少*(如果有的话)发生。当训练卷积神经网络时，我们通常希望在训练模型之前*移除*那些重复的图像。

其次，试图手动*检测数据集中的重复图像是极其*耗时且容易出错的——它也不能扩展到大型图像数据集。因此，我们需要一种方法来自动*检测并从我们的深度学习数据集中删除重复的图像。***

 ***这样的方法可能吗？

的确如此——我将在今天教程的剩余部分讲述这一点。

**要了解如何从深度学习数据集中检测和删除重复图像，*继续阅读！***

## 从用于深度学习的数据集中检测并移除重复图像

在本教程的第一部分中，您将了解为什么在您尝试在数据基础上训练深度神经网络之前，通常需要从数据集中检测和移除重复图像。

从那里，我们将回顾我创建的示例数据集，以便我们可以练习检测数据集中的重复图像。

然后，我们将使用一种叫做[图像散列](https://pyimagesearch.com/2017/11/27/image-hashing-opencv-python/)的方法来实现我们的图像重复检测器。

最后，我们将回顾我们的工作结果，并:

1.  执行模拟运行，以验证我们的图像重复检测器工作正常
2.  再次运行我们的重复检测器，这一次从数据集中删除实际的重复项

## 在训练深度神经网络时，为什么要费心从数据集中删除重复的图像？

如果您曾经尝试过手动构建自己的影像数据集，那么您知道很有可能(如果不是不可避免的话)您的数据集中会有重复的影像。

通常，由于以下原因，您最终会在数据集中得到重复的图像:

1.  从多个来源抓取图像(例如，Google、Bing 等。)
2.  组合现有数据集(例如将 ImageNet 与 Sun397 和室内场景相结合)

发生这种情况时，您需要一种方法来:

1.  **检测**数据集中有重复的图像
2.  **删除**重复项

**但这也引出了一个问题— *为什么一开始就要费心去关心副本呢？***

监督机器学习方法的通常假设是:

1.  数据点是独立的
2.  它们是同分布的
3.  训练和测试数据是从同一个分布中抽取的

问题是这些假设在实践中很少(如果有的话)成立。

你真正需要害怕的是你的模型的概括能力。

如果您的数据集中包含多个相同的图像，您的神经网络将被允许在每个时期多次从该图像中查看和学习模式。

你的网络可能会变得**偏向**那些重复图像中的模式，**使得它不太可能推广到*新的*图像。**

偏见和归纳能力在机器学习中是一件大事——当处理“理想”数据集时，它们可能很难克服。

花些时间从影像数据集中移除重复项，以免意外引入偏差或损害模型的概化能力。

## 我们的示例重复图像数据集

为了帮助我们学习如何从深度学习数据集中检测和删除重复的图像，我基于[斯坦福狗数据集](http://vision.stanford.edu/aditya86/ImageNetDogs/)创建了一个我们可以使用的“练习”数据集。

这个数据集由 20，580 张狗品种的图片组成，包括比格犬、纽芬兰犬和博美犬等等。

**为了创建我们的复制图像数据集，我:**

1.  下载了斯坦福狗的数据集
2.  我会复制三个图像样本
3.  将这三个图像中的每一个总共复制了 *N* 次
4.  然后进一步随机取样斯坦福狗数据集，直到我总共获得了 1000 张图像

下图显示了每个图像的副本数量:

我们的目标是创建一个 Python 脚本，它可以在训练深度学习模型之前检测并删除这些重复。

## 项目结构

我已经在本教程的 ***“下载”*** 部分包含了复制的图像数据集和代码。

一旦您提取了`.zip`，您将看到下面的目录结构:

```py
$ tree --dirsfirst --filelimit 10
.
├── dataset [1000 entries]
└── detect_and_remove.py

1 directory, 1 file

```

如您所见，我们的项目结构非常简单。我们有 1000 张图片(包括重复的)。此外，我们还有我们的`detect_and_remove.py` Python 脚本，它是今天教程的基础。

## 实现我们的图像重复检测器

我们现在准备实施我们的图像重复检测器。

打开项目目录中的`detect_and_remove.py`脚本，让我们开始工作:

```py
# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2
import os

```

我们脚本的导入包括来自 [imutils](https://github.com/jrosebr1/imutils) 的我的`paths`实现，因此我们可以获取我们数据集中所有图像的文件路径，NumPy 用于图像堆叠，OpenCV 用于图像 I/O、操作和显示。`os`和`argparse`都是 Python 内置的。

如果你的机器上没有安装 OpenCV 或 imutils，我建议你遵循我的 *[pip 安装 opencv](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)* 指南，它会告诉你如何安装两者。

本教程的主要组件是`dhash`函数:

```py
def dhash(image, hashSize=8):
	# convert the image to grayscale and resize the grayscale image,
	# adding a single column (width) so we can compute the horizontal
	# gradient
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hashSize + 1, hashSize))

	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]

	# convert the difference image to a hash and return it
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

```

*   将图像转换为单通道灰度图像(**第 12 行**)
*   根据`hashSize` ( **第 13 行**)调整图像大小。该算法要求图像的*宽度*恰好比*高度*多`1`列，如维度元组所示。
*   计算相邻列像素之间的相对水平梯度(**行 17** )。这就是现在众所周知的“差异图像”
*   应用我们的散列计算并返回结果(**第 20 行**)。

我已经在之前的文章的[中介绍过图像哈希技术。特别是，你应该阅读我的](https://pyimagesearch.com/tag/image-hashing/)*[使用 OpenCV 和 Python](https://pyimagesearch.com/2017/11/27/image-hashing-opencv-python/)* 进行图像哈希的指南，来理解使用我的`dhash`函数进行图像哈希的概念。

定义好散列函数后，我们现在准备好[定义和解析命令行参数](https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/):

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-r", "--remove", type=int, default=-1,
	help="whether or not duplicates should be removed (i.e., dry run)")
args = vars(ap.parse_args())
```

我们的脚本处理两个命令行参数，您可以通过终端或命令提示符传递这两个参数:

*   `--dataset`:输入数据集的路径，其中包含您希望从数据集中删除的重复项
*   ``--remove``

我们现在准备开始计算哈希值:

```py
# grab the paths to all images in our input dataset directory and
# then initialize our hashes dictionary
print("[INFO] computing image hashes...")
imagePaths = list(paths.list_images(args["dataset"]))
hashes = {}

# loop over our image paths
for imagePath in imagePaths:
	# load the input image and compute the hash
	image = cv2.imread(imagePath)
	h = dhash(image)

	# grab all image paths with that hash, add the current image
	# path to it, and store the list back in the hashes dictionary
	p = hashes.get(h, [])
	p.append(imagePath)
	hashes[h] = p
```

*   加载图像(**第 39 行**)
*   使用`dhash`便利函数(**第 40 行**)计算散列值`h`
*   抓取所有图像路径，`p`，用相同的 hash，`h` ( **第 44 行**)。
*   将最新的`imagePath`追加到`p` ( **第 45 行**)。此时，`p`代表我们的一组重复图像(即，具有相同散列值的图像)
*   将所有这些重复项添加到我们的`hashes`字典中(**第 46 行**)

```py
{
	...
	7054210665732718398: ['dataset/00000005.jpg', 'dataset/00000071.jpg', 'dataset/00000869.jpg'],
	8687501631902372966: ['dataset/00000011.jpg'],
	1321903443018050217: ['dataset/00000777.jpg'],
	...
}

```

请注意第一个散列关键字示例如何具有*三个*关联的图像路径(表示重复)，而接下来的两个散列关键字只有*一个*路径条目(表示没有重复)。

此时，计算完所有的散列后，我们需要循环遍历散列并处理重复项:

```py
# loop over the image hashes
for (h, hashedPaths) in hashes.items():
	# check to see if there is more than one image with the same hash
	if len(hashedPaths) > 1:
		# check to see if this is a dry run
		if args["remove"] <= 0:
			# initialize a montage to store all images with the same
			# hash
			montage = None

			# loop over all image paths with the same hash
			for p in hashedPaths:
				# load the input image and resize it to a fixed width
				# and heightG
				image = cv2.imread(p)
				image = cv2.resize(image, (150, 150))

				# if our montage is None, initialize it
				if montage is None:
					montage = image

				# otherwise, horizontally stack the images
				else:
					montage = np.hstack([montage, image])

			# show the montage for the hash
			print("[INFO] hash: {}".format(h))
			cv2.imshow("Montage", montage)
			cv2.waitKey(0)
```

如果不是，我们忽略这个散列并继续检查下一个散列。

**另一方面，如果事实上有两个或更多的`hashedPaths`，它们就是重复的！**

因此，我们启动一个`if` / `else`块来检查这是否是一次“预演”;如果`--remove`标志不是正值，我们将进行一次试运行(**第 53 行**)。

预演意味着我们还没有准备好*删除*副本。相反，我们只是想让*检查*，看看是否存在任何重复项。

```py
# otherwise, we'll be removing the duplicate images
		else:
			# loop over all image paths with the same hash *except*
			# for the first image in the list (since we want to keep
			# one, and only one, of the duplicate images)
			for p in hashedPaths[1:]:
				os.remove(p)

```

在这种情况下，我们实际上是*从我们的系统中删除*重复的图像。

在这个场景中，除了列表中的第一个图像的之外，我们简单地循环所有具有相同散列值*的图像路径——我们希望**保留一个**，并且只保留一个示例图像，而**删除**所有其他相同的图像。*

伟大的工作实现自己的重复图像检测和删除系统。

## 为我们的深度学习数据集运行图像重复检测器

让我们把我们的图像重复检测工作。

首先确保您已经使用了本教程的 ***【下载】*** 部分来下载源代码和示例数据集。

从那里，打开一个终端，执行以下命令来验证在我们的`dataset/`目录中有 1，000 个图像:

```py
$ ls -l dataset/*.jpg | wc -l
    1000

```

现在，让我们执行一次预演，这将允许我们可视化数据集中的重复项:

```py
$ python detect_and_remove.py --dataset dataset
[INFO] computing image hashes...
[INFO] hash: 7054210665732718398
[INFO] hash: 15443501585133582635
[INFO] hash: 13344784005636363614

```

下图显示了我们脚本的输出，表明我们已经能够成功地找到重复项，如上面的*“我们的示例重复图像数据集”*部分所述。

为了从我们的系统中真正删除重复项，我们需要再次执行`detect_and_remove.py`，这次提供`--remove 1`命令行参数:

```py
$ python detect_and_remove.py --dataset dataset --remove 1
[INFO] computing image hashes...

```

我们可以通过计算`dataset`目录中 JPEG 图像的数量来验证重复图像是否已被删除:

```py
$ ls -l dataset/*.jpg | wc -l
     993

```

最初`dataset`中有 1000 张图片，现在有 993 张，暗示我们去掉了 7 张重复的图片。

在这一点上，你可以继续在这个数据集上训练一个深度神经网络。

## 如何创建自己的影像数据集？

我为今天的教程创建了一个示例数据集，它包含在 ***【下载】*** 中，以便您可以立即开始学习重复数据删除的概念。

但是，您可能想知道:

*“首先，我如何创建数据集？”*

创建数据集没有“一刀切”的方法。相反，你需要考虑这个问题并围绕它设计你的数据收集。

您可能会确定需要自动化和摄像机来收集数据。或者您可能决定需要合并现有的数据集来节省大量时间。

让我们首先考虑以人脸应用为目的的数据集。如果您想[创建一个自定义的人脸数据集](https://pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/)，您可以使用三种方法中的任何一种:

1.  通过 OpenCV 和网络摄像头注册人脸
2.  以编程方式下载人脸图像
3.  手动采集人脸图像

从那里，你可以应用[面部应用](https://pyimagesearch.com/category/faces/)，包括面部识别、面部标志等。

但是如果你想利用互联网和现有的搜索引擎或抓取工具呢？有希望吗？

事实上是有的。我写了三个教程来帮助你入门。

1.  *[如何使用谷歌图片创建深度学习数据集](https://pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/)*
2.  *[如何(快速)建立深度学习图像数据集](https://pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/)(使用 Bing)*
3.  *[用 Python 和刺儿头刮图像](https://pyimagesearch.com/2015/10/12/scraping-images-with-python-and-scrapy/)*

使用这些博客帖子来帮助创建您的数据集，请记住图像所有者的版权。一般来说，你只能出于教育目的使用受版权保护的图片。出于商业目的，您需要联系每个图像的所有者以获得许可。

在网上收集图片几乎总会导致重复——一定要快速检查。创建数据集后，请按照今天的重复数据删除教程计算哈希值并自动删除重复项。

回想一下从机器学习和深度学习数据集中删除重复项的两个重要原因:

1.  你的数据集中的重复图像将*偏差*引入到你的数据集中，给你的深度神经网络额外的机会来学习重复图像的*特定*模式。
2.  此外，副本会影响您的模型将归纳为超出其训练范围的新图像的能力。

从那里，您可以在新形成的数据集上训练您自己的深度学习模型，并部署它！

## 摘要

在本教程中，您学习了如何从深度学习数据集中检测和移除重复图像。

通常，您会希望从数据集中移除重复的图像，以确保每个数据点(即图像)仅表示一次*-如果数据集中有多个相同的图像，您的卷积神经网络可能会学习偏向这些图像，从而使您的模型不太可能推广到新图像。*

 *为了帮助防止这种类型的偏差，我们使用一种叫做[图像哈希](https://pyimagesearch.com/2017/11/27/image-hashing-opencv-python/)的方法实现了我们的重复检测器。

图像哈希的工作原理是:

1.  检查图像的内容
2.  构造散列值(即整数)，该散列值*仅基于图像的内容*唯一地量化输入图像*的*

使用我们的散列函数，我们可以:

1.  循环浏览我们图像数据集中的所有图像
2.  为每个图像计算图像哈希
3.  检查“哈希冲突”，这意味着如果两个图像有相同的哈希，它们一定是重复的
4.  从我们的数据集中删除重复的图像

使用本教程中介绍的技术，您可以从自己的数据集中检测并删除重复的图像——从那里，您将能够在新删除重复数据的数据集上训练一个深度神经网络！

我希望你喜欢这个教程。

**要下载这篇文章的源代码和示例数据集(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！********