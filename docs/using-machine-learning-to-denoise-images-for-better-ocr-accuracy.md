# 使用机器学习对图像去噪以获得更好的 OCR 准确度

> 原文：<https://pyimagesearch.com/2021/10/20/using-machine-learning-to-denoise-images-for-better-ocr-accuracy/>

应用光学字符识别(OCR)最具挑战性的方面之一不是 *OCR 本身*。相反，它是预处理、去噪和清理图像的过程，这样它们*就可以*被 OCR 识别。

**要学习如何对你的图像去噪以获得更好的光学字符识别，** ***继续阅读。***

## **使用机器学习对图像进行去噪以获得更好的 OCR 准确度**

当处理由计算机生成的文档、屏幕截图或基本上任何从未接触过打印机然后扫描的文本时，OCR 变得容易得多。文字干净利落。背景和前景之间有足够的对比。而且大多数时候，文字并不存在于复杂的背景上。

一旦一段文字被打印和扫描，一切都变了。从那里开始，OCR 变得*更具挑战性。*

*   打印机可能碳粉或墨水不足，导致文本褪色且难以阅读。
*   扫描文档时可能使用了旧扫描仪，导致图像分辨率低和文本对比度差。
*   一个手机扫描仪应用程序可能在光线不好的情况下使用，这使得人眼阅读文本变得非常困难，更不用说计算机了。
*   太常见的是真实的人类触摸过纸张的清晰迹象，包括角落上的咖啡杯污渍、纸张起皱、破裂、撕裂等。

对于人类思维所能做的所有惊人的事情，当涉及到印刷材料时，我们似乎都只是等待发生的行走事故。给我们一张纸和足够的时间，我保证即使我们中最有条理的人也会从原始状态中取出文件，并最终在其上引入一些污点、裂口、褶皱和皱纹。

这些问题不可避免地会出现，当这些问题出现时，我们需要利用我们的计算机视觉、图像处理和 OCR 技能来预处理和提高这些受损文档的质量。从那里，我们将能够获得更高的 OCR 准确性。

在本教程的剩余部分，您将了解即使是以新颖方式构建的简单机器学习算法也可以帮助您在应用 OCR 之前对图像进行降噪。

## **学习目标**

在本教程中，您将:

*   获得处理嘈杂、损坏的文档数据集的经验
*   了解机器学习如何用于对这些损坏的文档进行降噪
*   使用 [Kaggle 的去噪脏文档](https://www.kaggle.com/c/denoising-dirty-documents/data)数据集
*   从该数据集中提取要素
*   根据我们提取的特征训练一个随机森林回归器(RFR)
*   用这个模型去噪我们测试集中的图像(然后也能去噪你的数据集)

## **利用机器学习进行图像去噪**

在本教程的第一部分，我们将回顾数据集，我们将使用去噪文件。从那里，我们将回顾我们的项目结构，包括我们将使用的五个单独的 Python 脚本，包括:

*   存储跨多个 Python 脚本使用的变量的配置文件
*   一个助手功能，用来模糊和限制我们的文件
*   用于从数据集中提取要素和目标值的脚本
*   用于训练 RFR 的另一个脚本
*   以及用于将我们的训练模型应用到我们的测试集中的图像的最终脚本

这是我的一个较长的教程，虽然它很简单，并遵循线性进展，这里也有许多微妙的细节。因此，我建议你将本教程*复习两遍*，一遍从高层次理解我们在做什么，然后再从低层次理解实现。

说完了，我们开始吧！

### **我们嘈杂的文档数据集**

在本教程中，我们将使用 Kaggle 的[去噪脏文档](https://www.kaggle.com/c/denoising-dirty-documents/data)数据集。该数据集是 [UCI 机器学习库](http://archive.ics.uci.edu/ml)的一部分，但被转换成了 Kaggle 竞赛。在本教程中，我们将使用三个文件。这些文件是 Kaggle 竞赛数据的一部分，并被命名为:`test.zip`、`train.zip`和`train_cleaned.zip`。

该数据集相对较小，只有 144 个训练样本，因此很容易处理并用作教育工具。但是，不要让小数据集欺骗了你！我们要对这个数据集做的是*远离*基础的或介绍性的。

**图** **1** 显示了脏文档数据集的示例。对于样本文档，*顶部*显示文档的噪声版本，包括污点、褶皱、折叠等。然后,*底部的*显示了我们希望生成的文档的目标原始版本。

**我们的目标是在** ***顶部*** **输入图像，并训练机器学习模型在** ***底部产生干净的输出。现在看来这似乎是不可能的，但是一旦你看到我们将要使用的一些技巧和技术，这将比你想象的要简单得多。***

### **文档去噪算法**

我们的去噪算法依赖于训练 RFR 接受有噪声的图像，并自动预测输出像素值。这种算法的灵感来自科林·普里斯特介绍的[去噪技术。](https://colinpriest.com/2015/09/07/denoising-dirty-documents-part-6/)

这些算法通过应用一个从*从左到右*和*从上到下*滑动的`5 x 5` 窗口来工作，一次一个像素(**图 2** )穿过*噪声图像(即，我们想要自动预处理和清理的图像)和目标输出图像(即，图像在清理后应该出现的“黄金标准”)。*

 *在每个滑动窗口停止时，我们提取:

1.  **噪声输入图像**的`5 x 5`区域。然后，我们将`5 x 5` 区域展平成一个`25-d` 列表，并将其视为一个特征向量。
2.  同样的`5 x 5` 区域的**被清理过的图像**，但这次我们只取中心`(x, y)`-坐标，用位置`(2, 2)` *表示。*

给定来自噪声输入图像的`25-d`(维度)特征向量，这个单个像素值就是我们想要我们的 RFR 预测的。

为了使这个例子更具体，再次考虑图 2 中的**、**，这里我们有下面的`5 x 5`、*、*来自噪声图像的网格像素值:

```py
[[247 227 242 253 237]
 [244 228 225 212 219]
 [223 218 252 222 221]
 [242 244 228 240 230]
 [217 233 237 243 252]]
```

然后，我们将其展平成一个由`5 x 5 = 25-d` 值组成的列表:

```py
[247 227 242 253 237 244 228 225 212 219 223 218 252 222 221 242 244 228
 240 230 217 233 237 243 252]
```

**这个`25-d`向量是我们的特征向量，我们的 RFR 将在其上被训练。**

但是，我们仍然需要定义 RFR 的目标产值。**我们的回归模型应该接受输入的`25-d`向量，并输出干净的、去噪的像素。**

现在，让我们假设我们的黄金标准/目标图像中有以下`5 x 5` 窗口:

```py
[[0 0 0 0 0]
 [0 0 0 0 1]
 [0 0 1 1 1]
 [0 0 1 1 1]
 [0 0 0 1 1]]
```

我们只对这个`5 x 5` *区域*的*中心*感兴趣，记为位置`x = 2`、`y = 2` *。* **因此，我们提取`1`(前景，相对于`0`，是背景)的这个值，并把它作为我们的 RFR 应该预测的目标值。**

将整个示例放在一起，我们可以将以下视为样本训练数据点:

```py
trainX = [[247 227 242 253 237 244 228 225 212 219 223 218 252 222 221 242 244 228
 240 230 217 233 237 243 252]]
trainY = [[1]]
```

给定我们的`trainX`变量(我们的原始像素强度)，我们想要预测`trainY`中相应的净化/去噪像素值。

**我们将以这种方式训练我们的 RFR，最终得到一个模型，该模型可以接受有噪声的文档输入，并通过检查局部`5 x 5` 区域，然后预测中心(干净的)像素值，自动去噪。**

### **配置您的开发环境**

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果你需要帮助为 OpenCV 配置开发环境，我*强烈推荐*阅读我的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让你启动并运行。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

本教程的项目目录结构比其他教程稍微复杂一点，因为有五个 Python 脚本需要查看(三个脚本、一个助手函数和一个配置文件)。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

在我们继续之前，让我们熟悉一下这些文件:

```py
|-- pyimagesearch
|   |-- __init__.py
|   |-- denoising
|   |   |-- __init__.py
|   |   |-- helpers.py
|-- config
|   |-- __init__.py
|   |-- denoise_config.py
|-- build_features.py
|-- denoise_document.py
|-- denoiser.pickle
|-- denoising-dirty-documents
|   |-- test
|   |   |-- 1.png
|   |   |-- 10.png
|   |   |-- ...
|   |   |-- 94.png
|   |   |-- 97.png
|   |-- train
|   |   |-- 101.png
|   |   |-- 102.png
|   |   |-- ...
|   |   |-- 98.png
|   |   |-- 99.png
|   |-- train_cleaned
|   |   |-- 101.png
|   |   |-- 102.png
|   |   |-- ...
|   |   |-- 98.png
|   |   |-- 99.png
|-- train_denoiser.py
```

`denoising-dirty-documents directory`包含来自 [Kaggle 去噪脏文档](https://www.kaggle.com/c/denoising-dirty-documents/data)数据集的所有图像。

在`pyimagesearch`的`denoising`子模块中，有一个`helpers.py`文件。该文件包含一个函数`blur_and_threshold`，顾名思义，该函数用于将平滑和阈值处理相结合，作为我们文档的预处理步骤。

然后我们有了`denoise_config.py`文件，它存储了一些指定训练数据文件路径、输出特征 CSV 文件和最终序列化 RFR 模型的配置。

我们将完整回顾三个 Python 脚本:

1.  `build_features.py`:接受我们的输入数据集并创建一个 CSV 文件，我们将使用它来训练我们的 RFR。
2.  `train_denoiser.py`:训练实际的 RFR 模型，并将其序列化到磁盘中作为`denoiser.pickle`。
3.  `denoise_document.py`:从磁盘接受输入图像，加载训练好的 RFR，然后对输入图像去噪。

在本教程中，我们需要回顾几个 Python 脚本。因此，我建议你将本教程*复习两遍*，以便更好地理解我们正在实现什么，然后在更深层次上掌握实现。

### **实现我们的配置文件**

实现降噪文档的第一步是创建配置文件。打开项目目录结构的`config`子目录下的`denoise_config.py`文件，插入以下代码:

```py
# import the necessary packages
import os

# initialize the base path to the input documents dataset
BASE_PATH = "denoising-dirty-documents"

# define the path to the training directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "train"])
CLEANED_PATH = os.path.sep.join([BASE_PATH, "train_cleaned"])
```

**第 5 行**定义了我们的`denoising-dirty-documents`数据集的基本路径。如果您从 Kaggle 下载该数据集，请确保解压缩该目录下的所有*`.zip`文件，以使数据集中的所有图像解压缩并驻留在磁盘上。*

 *然后我们分别定义到*原始噪声*图像目录和相应的*清洁图像*目录的路径(**第 8 行和第 9 行**)。

`TRAIN_PATH`图像包含有噪声的文档，而`CLEANED_PATH`图像包含我们的“黄金标准”,即在通过我们训练的模型应用文档去噪后，我们的输出图像应该是什么样子。我们将在我们的`train_denoiser.py`脚本中构建测试集。

让我们继续定义配置文件:

```py
# define the path to our output features CSV file then initialize
# the sampling probability for a given row
FEATURES_PATH = "features.csv"
SAMPLE_PROB = 0.02

# define the path to our document denoiser model
MODEL_PATH = "denoiser.pickle"
```

**第 13 行**定义了输出`features.csv`文件的路径。我们的特色包括:

1.  通过滑动窗口从有噪声的输入图像中采样的局部`5 x 5`区域
2.  `5 x 5` 区域的中心，记为`(x, y)`-坐标`(2, 2)`，对应于清洗后的图像

然而，如果我们将每个特性/目标组合的*写到磁盘上，我们最终会得到*数百万*行和一个几千兆字节大小的 CSV。因此，我们不是穷尽地计算所有的滑动窗口和目标组合，而是以`SAMPLES_PROB`的概率将它们写入磁盘。*

最后，**第 17 行**指定了到`MODEL_PATH`的路径，我们的输出序列化模型。

### **创建我们的模糊和阈值辅助函数**

为了帮助我们的 RFR 从前景(即文本)像素中预测背景(即噪声)，我们需要定义一个助手函数，该函数将在我们训练模型并使用它进行预测之前预处理我们的图像。

我们的图像处理操作流程可以在**图** **4** 中看到。首先，我们将输入图像模糊化*(左上)*，然后从输入图像中减去模糊的图像*(右上)*。**我们这样做是为了逼近图像的前景，因为本质上，模糊会模糊聚焦的特征，并显示图像的更多“结构”成分。**

接下来，我们通过将任何大于零的像素值设置为零来对近似的前景区域进行阈值化(**图** **4** ，*左下方*)。

最后一步是执行最小-最大缩放*(右下角)*，这将使像素亮度回到范围`[0, 1]`(或`[0, 255]`，取决于您的数据类型)。**当我们执行滑动窗口采样时，这个最终图像将作为噪声输入。**

现在我们已经了解了一般的预处理步骤，让我们用 Python 代码来实现它们。

在`pyimagesearch`的`denoising`子模块中打开`helpers.py`文件，让我们开始定义我们的`blur_and_threshold`功能:

```py
# import the necessary packages
import numpy as np
import cv2

def blur_and_threshold(image, eps=1e-7):
	# apply a median blur to the image and then subtract the blurred
	# image from the original image to approximate the foreground
	blur = cv2.medianBlur(image, 5)
	foreground = image.astype("float") - blur

	# threshold the foreground image by setting any pixels with a
	# value greater than zero to zero
	foreground[foreground > 0] = 0
```

`blur_and_threshold`函数接受两个参数:

1.  `image`:我们将要预处理的输入图像。
2.  `eps`:用于防止被零除的ε值。

然后，我们对图像应用中值模糊以减少噪声，并从原始的`image`中减去`blur`，得到一个`foreground`近似值(**第 8 行和第 9 行**)。

从那里，我们通过将任何大于零的像素强度设置为零来对`foreground`图像进行阈值处理(**行 13** )。

这里的最后一步是执行最小-最大缩放:

```py
	# apply min/max scaling to bring the pixel intensities to the
	# range [0, 1]
	minVal = np.min(foreground)
	maxVal = np.max(foreground)
	foreground = (foreground - minVal) / (maxVal - minVal + eps)

	# return the foreground-approximated image
	return foreground
```

这里，我们找到了`foreground`图像中的最小值和最大值。我们使用这些值将`foreground`图像中的像素强度缩放到范围`[0, 1]` *。*

这个前景近似的图像然后被返回给调用函数。

### **实现特征提取脚本**

定义了我们的`blur_and_threshold`函数后，我们可以继续我们的`build_features.py`脚本。

顾名思义，这个脚本负责从*有噪*图像中创建我们的`5 x 5 - 25-d` 特征向量，然后从相应的黄金标准图像中提取*目标*(即清理后的)像素值。

我们将以 CSV 格式将这些特征保存到磁盘，然后在“实现我们的去噪训练脚本”一节中对它们训练一个随机森林回归模型

现在让我们开始实施:

```py
# import the necessary packages
from config import denoise_config as config
from pyimagesearch.denoising import blur_and_threshold
from imutils import paths
import progressbar
import random
import cv2
```

**第 2 行**导入我们的`config`来访问我们的数据集文件路径并输出 CSV 文件路径。注意，我们在这里使用了`blur_and_threshold`函数。

下面的代码块获取我们的`TRAIN_PATH`(噪声图像)和`CLEANED_PATH`(我们的 RFR 将学习预测的干净图像)中所有图像的路径:

```py
# grab the paths to our training images
trainPaths = sorted(list(paths.list_images(config.TRAIN_PATH)))
cleanedPaths = sorted(list(paths.list_images(config.CLEANED_PATH)))

# initialize the progress bar
widgets = ["Creating Features: ", progressbar.Percentage(), " ",
	progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(trainPaths),
	widgets=widgets).start()
```

注意`trainPaths`包含了我们所有的*噪音*图像。`cleanedPaths`包含相应的*清洗过的*图像。

**图 5** 显示了一个例子。在*顶部*是我们的输入训练图像。在*底部*，我们有相应的图像清理版本。我们将从`trainPaths`和`cleanedPaths`中提取`5 x 5` 区域——目标是使用有噪声的`5 x 5` 区域来预测干净的版本。

现在让我们开始循环这些图像组合:

```py
# zip our training paths together, then open the output CSV file for
# writing
imagePaths = zip(trainPaths, cleanedPaths)
csv = open(config.FEATURES_PATH, "w")

# loop over the training images together
for (i, (trainPath, cleanedPath)) in enumerate(imagePaths):
	# load the noisy and corresponding gold-standard cleaned images
	# and convert them to grayscale
	trainImage = cv2.imread(trainPath)
	cleanImage = cv2.imread(cleanedPath)
	trainImage = cv2.cvtColor(trainImage, cv2.COLOR_BGR2GRAY)
	cleanImage = cv2.cvtColor(cleanImage, cv2.COLOR_BGR2GRAY)
```

在**第 21 行，**我们使用 Python 的`zip`函数将`trainPaths`和`cleanedPaths`组合在一起。然后，我们打开我们的输出`csv`文件，写入第**行 22。**

**第 25 行**在我们的`imagePaths`组合上开始一个循环。对于每个`trainPath`，我们也有相应的`cleanedPath`。

我们从磁盘中加载我们的`trainImage`和`cleanImage`，并将它们转换成灰度(**第 28-31 行**)。

接下来，我们需要在每个方向用 2 像素的边框填充`trainImage`和`cleanImage`:

```py
	# apply 2x2 padding to both images, replicating the pixels along
	# the border/boundary
	trainImage = cv2.copyMakeBorder(trainImage, 2, 2, 2, 2,
		cv2.BORDER_REPLICATE)
	cleanImage = cv2.copyMakeBorder(cleanImage, 2, 2, 2, 2,
		cv2.BORDER_REPLICATE)

	# blur and threshold the noisy image
	trainImage = blur_and_threshold(trainImage)

	# scale the pixel intensities in the cleaned image from the range
	# [0, 255] to [0, 1] (the noisy image is already in the range
	# [0, 1])
	cleanImage = cleanImage.astype("float") / 255.0
```

为什么我们要为填充物费心呢？我们从输入图像的*从左到右*和*从上到下*滑动一个窗口，并使用窗口内的像素来预测位于`x = 2`、`y = 2`的输出*中心*像素，这与卷积运算没有什么不同(只有卷积我们的滤波器是固定和定义的)。

像卷积一样，您需要填充输入图像，以便输出图像的大小不会变小。如果你不熟悉这个概念，请参考我的关于 OpenCV 和 Python 卷积的指南。

填充完成后，我们对`trainImage`进行模糊和阈值处理，并手动将`cleanImage`缩放到范围`[0, 1]`。由于`blur_and_threshold`内的最小-最大缩放，`trainImage`已经*缩放到`[0, 1]` 范围。*

对我们的图像进行预处理后，我们现在可以在图像上滑动一个`5 x 5` 窗口:

```py
	# slide a 5x5 window across the images
	for y in range(0, trainImage.shape[0]):
		for x in range(0, trainImage.shape[1]):
			# extract the window ROIs for both the train image and
			# clean image, then grab the spatial dimensions of the
			# ROI
			trainROI = trainImage[y:y + 5, x:x + 5]
			cleanROI = cleanImage[y:y + 5, x:x + 5]
			(rH, rW) = trainROI.shape[:2]

			# if the ROI is not 5x5, throw it out
			if rW != 5 or rH != 5:
				continue
```

**49 线和 50 线**从*左右*和*上下*滑动一个`5 x 5` 窗口穿过`trainImage`和`cleanImage`。在每次滑动窗口停止时，我们提取训练图像和干净图像的`5 x 5`*ROI(**行 54 和 55** )。*

 *我们获取第 56**行**上`trainROI`的宽度和高度，如果宽度或高度不是五个像素(由于我们在图像的边界上)，我们丢弃 ROI(因为我们只关心`5 x 5` 区域)。

接下来，我们构建我们的特征向量，并将该行保存到 CSV 文件中:

```py
			# our features will be the flattened 5x5=25 raw pixels
			# from the noisy ROI while the target prediction will
			# be the center pixel in the 5x5 window
			features = trainROI.flatten()
			target = cleanROI[2, 2]

			# if we wrote *every* feature/target combination to disk
			# we would end up with millions of rows -- let's only
			# write rows to disk with probability N, thereby reducing
			# the total number of rows in the file
			if random.random() <= config.SAMPLE_PROB:
				# write the target and features to our CSV file
				features = [str(x) for x in features]
				row = [str(target)] + features
				row = ",".join(row)
				csv.write("{}\n".format(row))

	# update the progress bar
	pbar.update(i)

# close the CSV file
pbar.finish()
csv.close()
```

**第 65 行**从`trainROI`中取出`5 x 5` 像素区域，将其展平成一个`5 x 5 = 25-d` 列表— **这个列表作为我们的特征向量。**

**行 66** 然后从`cleanROI`的中心提取*清洁/黄金标准*像素值。**这个像素值就是我们希望 RFR 预测的值。**

此时，我们可以将特征向量和目标值的组合写入磁盘；然而，如果我们将每个特性/目标组合的*写到 CSV 文件中，我们最终会得到一个几千兆字节大小的文件。*

为了避免产生大量的 CSV 文件，我们需要在下一步中处理它。因此，我们改为只允许将`SAMPLE_PROB`(在本例中，2%)行写入磁盘(**行 72** )。进行这种采样可以减小生成的 CSV 文件的大小，并使其更易于管理。

**第 74 行**构建了我们的`features`行，并在前面加上了`target`像素值。然后，我们将该行写入 CSV 文件。我们对所有`imagePaths`重复这个过程。

### **运行特征提取脚本**

我们现在准备运行我们的特征提取器。首先，打开一个终端，然后执行`build_features.py`脚本:

```py
$ python build_features.py
Creating Features: 100% |#########################| Time:  0:01:05
```

在我的 3 GHz 英特尔至强 W 处理器上，整个特征提取过程只花了一分多钟。

检查我的项目目录结构，您现在可以看到结果 CSV 文件的特性:

```py
$ ls -l *.csv
adrianrosebrock  staff  273968497 Oct 23 06:21 features.csv
```

如果您打开系统中的`features.csv`文件，您会看到每行包含 26 个条目。

行中的第一个条目是目标输出像素。我们将尝试根据该行剩余部分的内容预测输出像素值，这些内容是输入 ROI 像素的`5 x 5 = 25` 。

下一节将介绍如何训练一个 RFR 模型来做到这一点。

### **实施我们的去噪训练脚本**

现在我们的`features.csv`文件已经生成，我们可以继续学习训练脚本了。这个脚本负责加载我们的`features.csv`文件，训练一个 RFR 接受一个*有噪*图像的`5 x 5` 区域，然后预测*清理后的*中心像素值。

让我们开始检查代码:

```py
# import the necessary packages
from config import denoise_config as config
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
```

**第 2-7 行**处理我们需要的 Python 包，包括:

*   `config`:我们的项目配置保存了我们的输出文件路径和训练变量
*   `RandomForestRegressor`:我们将用来预测像素值的回归模型的 scikit-learn 实现
*   我们的误差/损失函数——这个值越低，我们在图像去噪方面做得越好
*   `train_test_split`:用于从我们的`features.csv`文件创建培训/测试分割
*   `pickle`:用于将我们训练过的 RFR 序列化到磁盘

让我们继续从磁盘加载我们的 CSV 文件:

```py
# initialize lists to hold our features and target predicted values
print("[INFO] loading dataset...")
features = []
targets = []

# loop over the rows in our features CSV file
for row in open(config.FEATURES_PATH):
	# parse the row and extract (1) the target pixel value to predict
	# along with (2) the 5x5=25 pixels which will serve as our feature
	# vector
	row = row.strip().split(",")
	row = [float(x) for x in row]
	target = row[0]
	pixels = row[1:]

	# update our features and targets lists, respectively
	features.append(pixels)
	targets.append(target)
```

**第 11 行和第 12 行**初始化我们的`features` ( `5 x 5` 像素区域)和目标(我们要预测的目标输出像素值)。

我们在第 15 行**开始循环 CSV 文件的所有行。对于每个`row`，我们提取`target`和`pixel`值(**第 19-22 行**)。然后我们分别更新我们的`features`和`targets`列表。**

将 CSV 文件加载到内存中后，我们可以构建我们的训练和测试分割:

```py
# convert the features and targets to NumPy arrays
features = np.array(features, dtype="float")
target = np.array(targets, dtype="float")

# construct our training and testing split, using 75% of the data for
# training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(features, target,
	test_size=0.25, random_state=42)
```

这里，我们使用 75%的数据进行训练，剩下的 25%用于测试。这种类型的分割在机器学习领域是相当标准的。

最后，我们可以训练我们的 RFR:

```py
# train a random forest regressor on our data
print("[INFO] training model...")
model = RandomForestRegressor(n_estimators=10)
model.fit(trainX, trainY)

# compute the root mean squared error on the testing set
print("[INFO] evaluating model...")
preds = model.predict(testX)
rmse = np.sqrt(mean_squared_error(testY, preds))
print("[INFO] rmse: {}".format(rmse))

# serialize our random forest regressor to disk
f = open(config.MODEL_PATH, "wb")
f.write(pickle.dumps(model))
f.close()
```

**第 39 行**初始化我们的`RandomForestRegressor`，指示它训练`10`独立的回归树。然后在**线 40** 上训练该模型。

训练完成后，我们计算均方根误差(RMSE)来衡量我们在预测干净、去噪的图像方面做得有多好。*****降低*** **的误差值，把*****的工作做好。*****

 ***最后，我们将训练好的 RFR 模型序列化到磁盘上，这样我们就可以用它来预测我们的噪声图像。

### **训练我们的文档去噪模型**

随着我们的`train_denoiser.py`脚本的实现，我们现在准备训练我们的自动图像降噪器！首先，打开一个 shell，然后执行`train_denoiser.py`脚本:

```py
$ time python train_denoiser.py
[INFO] loading dataset...
[INFO] training model...
[INFO] evaluating model...
[INFO] rmse: 0.04990744293857625

real	1m18.708s
user	1m19.361s
sys     0m0.894s
```

训练我们的脚本只需要一分多钟，产生了一个`≈0.05`的 RMSE。这是一个非常低的损失值，表明我们的模型成功地接受了有噪声的输入像素 ROI，并正确地预测了目标输出值。

检查我们的项目目录结构，您会看到 RFR 模型已经被序列化到磁盘上，名为`denoiser.pickle`:

```py
$ ls -l *.pickle
adrianrosebrock  staff  77733392 Oct 23 denoiser.pickle
```

在下一节中，我们将从磁盘中加载经过训练的`denoiser.pickle`模型，然后使用它来自动清理和预处理我们的输入文档。

### **创建文档降噪脚本**

这个项目的最后一步是采用我们训练过的 denoiser 模型来自动清理我们的输入图像。

现在打开`denoise_document.py`，我们将看到这个过程是如何完成的:

```py
# import the necessary packages
from config import denoise_config as config
from pyimagesearch.denoising import blur_and_threshold
from imutils import paths
import argparse
import pickle
import random
import cv2
```

**第 2-8 行**处理导入我们需要的 Python 包。然后我们继续解析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--testing", required=True,
	help="path to directory of testing images")
ap.add_argument("-s", "--sample", type=int, default=10,
	help="sample size for testing images")
args = vars(ap.parse_args())
```

我们的`denoise_document.py`脚本接受两个命令行参数:

1.  `--testing`:包含 Kaggle 的 [*去噪脏文档*](https://www.kaggle.com/c/denoising-dirty-documents/data) 数据集测试图像的目录路径
2.  当应用我们的去噪模型时，我们将采样的测试图像的数量

说到我们的去噪模型，让我们从磁盘加载序列化模型:

```py
# load our document denoiser from disk
model = pickle.loads(open(config.MODEL_PATH, "rb").read())

# grab the paths to all images in the testing directory and then
# randomly sample them
imagePaths = list(paths.list_images(args["testing"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:args["sample"]]
```

我们还抓取测试集的所有`imagePaths`部分，随机洗牌，然后选择总共`--sample`个图像，在这些图像中应用我们的自动降噪模型。

让我们循环一遍`imagePaths`的样本:

```py
# loop over the sampled image paths
for imagePath in imagePaths:
	# load the image, convert it to grayscale, and clone it
	print("[INFO] processing {}".format(imagePath))
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	orig = image.copy()

	# pad the image followed by blurring/thresholding it
	image = cv2.copyMakeBorder(image, 2, 2, 2, 2,
		cv2.BORDER_REPLICATE)
	image = blur_and_threshold(image)
```

这里，我们执行的预处理步骤与我们在培训阶段使用的步骤相同:

*   我们从磁盘加载输入图像
*   将其转换为灰度
*   在每个方向用两个像素填充图像
*   应用`blur_and_threshold`功能

现在我们需要循环处理过的`image`并提取每个 `5 x 5` 像素邻域的 ***:***

```py
	# initialize a list to store our ROI features (i.e., 5x5 pixel
	# neighborhoods)
	roiFeatures = []

	# slide a 5x5 window across the image
	for y in range(0, image.shape[0]):
		for x in range(0, image.shape[1]):
			# extract the window ROI and grab the spatial dimensions
			roi = image[y:y + 5, x:x + 5]
			(rH, rW) = roi.shape[:2]

			# if the ROI is not 5x5, throw it out
			if rW != 5 or rH != 5:
				continue

			# our features will be the flattened 5x5=25 pixels from
			# the training ROI
			features = roi.flatten()
			roiFeatures.append(features)
```

**第 42 行**初始化一个列表`roiFeatures`，以存储每个`5 x 5` 邻域。

然后我们将一个`5 x 5` 窗口从*左右*和*上下*滑过`image`。在窗口的每一步，我们提取`roi` ( **第 48 行**)，抓取它的空间维度(**第 49 行**)，如果 ROI 大小不是`5 x 5` ( **第 52 行和第 53 行**)。

然后我们取我们的`5 x 5` 像素邻域，展平成一个`features`的列表，更新我们的`roiFeatures`列表(**第 57 行和第 58 行**)。

现在，在我们的滑动窗口`for`循环之外，我们用每一个可能的`5 x 5` 像素邻域填充了我们的`roiFeatures`。

然后我们可以对这些`roiFeatures`进行预测，得到最终的清洁图像:

```py
	# use the ROI features to predict the pixels of our new denoised
	# image
	pixels = model.predict(roiFeatures)

	# the pixels list is currently a 1D array so we need to reshape
	# it to a 2D array (based on the original input image dimensions)
	# and then scale the pixels from the range [0, 1] to [0, 255]
	pixels = pixels.reshape(orig.shape)
	output = (pixels * 255).astype("uint8")

	# show the original and output images
	cv2.imshow("Original", orig)
	cv2.imshow("Output", output)
	cv2.waitKey(0)
```

**第 62 行**调用`.predict`方法作为我们的 RFR，产生`pixels`，我们的前景对背景预测。

然而，我们的`pixels`列表目前是一个 1D 数组，所以我们必须注意将`reshape`数组转换成 2D 图像，然后将像素亮度缩放回`[0, 255]` ( **第 67 行和第 68 行**)。

最后，我们可以在屏幕上显示原始图像(有噪声的图像)和输出图像(干净的图像)。

### **运行我们的文档降噪器**

你成功了！这是一个很长的章节，但是我们终于准备好将我们的文档 denoiser 应用于我们的测试数据。

要查看我们的`denoise_document.py`脚本的运行情况，请打开一个终端并执行以下命令:

```py
$ python denoise_document.py --testing denoising-dirty-documents/test
[INFO] processing denoising-dirty-documents/test/133.png
[INFO] processing denoising-dirty-documents/test/160.png
[INFO] processing denoising-dirty-documents/test/40.png
[INFO] processing denoising-dirty-documents/test/28.png
[INFO] processing denoising-dirty-documents/test/157.png
[INFO] processing denoising-dirty-documents/test/190.png
[INFO] processing denoising-dirty-documents/test/100.png
[INFO] processing denoising-dirty-documents/test/49.png
[INFO] processing denoising-dirty-documents/test/58.png
[INFO] processing denoising-dirty-documents/test/10.png
```

我们的结果可以在**图** **6** 中看到。每个样本的*左侧*图像显示有噪声的输入文档，包括污点、褶皱、折叠等。右边的*显示了我们的 RFR 生成的清晰图像。*

**如你所见，我们的 RFR 在自动清理这些图像方面做得非常好！**

## **总结**

在本教程中，您学习了如何使用计算机视觉和机器学习对脏文档进行降噪。

使用这种方法，我们可以接受已经“损坏”的文档图像，包括裂口、撕裂、污点、起皱、折叠等。然后，通过以一种新的方式应用机器学习，我们可以将这些图像清理到接近原始状态，使 OCR 引擎更容易检测文本，提取文本，并正确地进行 OCR。

当你发现自己将 OCR 应用于真实世界的图像时，*尤其是*扫描的文档，你将不可避免地遇到质量差的文档。不幸的是，当这种情况发生时，您的 OCR 准确性可能会受到影响。

与其认输，不如考虑一下本教程中使用的技术会有什么帮助。是否可以手动预处理这些图像的子集，然后将它们用作训练数据？在此基础上，您可以训练一个模型，该模型可以接受有噪声的像素 ROI，然后生成原始、干净的输出。

通常，我们不使用原始像素作为机器学习模型的输入(当然，卷积神经网络除外)。通常，我们会使用一些特征检测器或描述符提取器来量化输入图像。从那里，产生的特征向量被交给机器学习模型。

很少有人看到标准的机器学习模型对原始像素强度进行操作。这是一个巧妙的技巧，但感觉在实践中并不可行。然而，正如你在这里看到的，这个方法是有效的！

我希望在实现文档去噪管道时，您可以将本教程作为一个起点。

更深入地说，您可以使用去噪自动编码器来提高去噪质量。在这一章中，我们使用了一个随机森林回归器，一个不同决策树的集合。你可能想探索的另一个系综是极端梯度增强，简称 XGBoost。

### **引用信息**

**A. Rosebrock** ，“使用机器学习对图像去噪以获得更好的 OCR 准确度”， *PyImageSearch* ，2021，[https://PyImageSearch . com/2021/10/20/Using-Machine-Learning-to-de-noise-Images-for-Better-OCR-Accuracy/](https://pyimagesearch.com/2021/10/20/using-machine-learning-to-denoise-images-for-better-ocr-accuracy/)

`@article{Rosebrock_2021_Denoise, author = {Adrian Rosebrock}, title = {Using Machine Learning to Denoise Images for Better {OCR} Accuracy}, journal = {PyImageSearch}, year = {2021}, note = {https://pyimagesearch.com/2021/10/20/using-machine-learning-to-denoise-images-for-better-ocr-accuracy/}, }`

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！**********