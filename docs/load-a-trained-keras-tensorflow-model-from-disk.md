# 从磁盘加载经过训练的 Keras/TensorFlow 模型

> 原文：<https://pyimagesearch.com/2021/05/22/load-a-trained-keras-tensorflow-model-from-disk/>

既然我们已经训练了模型并序列化了它，我们需要从磁盘加载它。作为模型序列化的一个实际应用，我将演示如何对动物数据集中的*张图片*进行分类，然后将分类后的图片显示在屏幕上。

**要了解如何从磁盘加载经过训练的 Keras/TensFlow 模型，*继续阅读。***

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

## 从磁盘加载经过训练的 Keras/TensorFlow 模型

打开一个新的文件，命名为`shallownet_load.py`，我们来动手做:

```py
# import the necessary packages
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2
```

我们从导入所需的 Python 包开始。**第 2-4 行**导入用于构建我们的标准管道的类，该管道将图像的大小调整为固定大小，将其转换为 Keras 兼容的数组，然后使用这些预处理程序将整个图像数据集加载到内存中。

用于从磁盘加载我们训练好的模型的实际函数是第 5 行的**上的`load_model`。该函数负责接受到我们训练好的网络(HDF5 文件)的路径，解码 HDF5 文件内的权重和优化器，并在我们的架构内设置权重，以便我们可以(1)继续训练或(2)使用网络对新图像进行分类。**

我们还将在第 9 行导入 OpenCV 绑定，这样我们就可以在图像上绘制分类标签，并将它们显示在屏幕上。

接下来，让我们解析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
args = vars(ap.parse_args())

# initialize the class labels
classLabels = ["cat", "dog", "panda"]
```

就像在`shallownet_save.py`中一样，我们需要两个命令行参数:

1.  `--dataset`:包含我们想要分类的图像的目录的路径(在这个例子中，是动物数据集)。
2.  `--model`:磁盘上序列化的*训练好的网络*的路径。

然后，第 20 行初始化动物数据集的类标签列表。

我们的下一个代码块处理从动物数据集中随机采样十个图像路径进行分类:

```py
# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]
```

这十个图像中的每一个都需要预处理，所以让我们初始化我们的预处理器，并从磁盘加载这十个图像:

```py
# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0
```

请注意我们是如何以与训练中预处理图像完全相同的方式预处理图像的。不执行此程序会导致不正确的分类，因为网络将呈现其无法识别的模式。始终特别注意确保您的*测试图像*以与您的*训练图像*相同的方式进行预处理。

接下来，让我们从磁盘加载保存的网络:

```py
# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])
```

加载我们的序列化网络就像调用`load_model`并提供驻留在磁盘上的模型的 HDF5 文件的路径一样简单。

一旦加载了模型，我们就可以对我们的 10 幅图像进行预测:

```py
# make predictions on the images
print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)
```

请记住，`model`的`.predict`方法将为`data`中的每个图像返回一个概率列表——分别为每个类别标签返回一个概率。在`axis=1`上取`argmax`为每幅图像找到具有*最大概率*的类别标签的索引。

现在我们有了我们的预测，让我们来想象一下结果:

```py
# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
	# load the example image, draw the prediction, and display it
	# to our screen
	image = cv2.imread(imagePath)
	cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
```

在第**行第 48** 处，我们开始循环十个随机采样的图像路径。对于每个图像，我们从磁盘中加载它(**第 51 行**)，并在图像本身上绘制分类标签预测(**第 52 行和第 53 行**)。然后输出图像通过第 54 和 55 行显示在我们的屏幕上。

要尝试一下`shallownet_load.py`,请执行以下命令:

```py
$ python shallownet_load.py --dataset ../datasets/animals \
	--model shallownet_weights.hdf5
[INFO] sampling images...
[INFO] loading pre-trained network...
[INFO] predicting...
```

根据输出，您可以看到我们的图像已被采样，预训练的浅网权重已从磁盘加载，浅网已对我们的图像进行了预测。我在**图 2** 中包含了一个来自图像本身上绘制的浅水网的预测样本。

请记住，ShallowNet 在动物数据集上获得了 *≈* 70%的分类准确率，这意味着近三分之一的示例图像将被错误分类。此外，根据早期教程中的[中的`classification_report`，我们知道网络仍然在努力坚持区分狗和猫。随着我们继续将深度学习应用于计算机视觉分类任务，我们将研究帮助我们提高分类准确性的方法。](https://pyimagesearch.com/2021/05/22/a-gentle-guide-to-training-your-first-cnn-with-keras-and-tensorflow/)

## **总结**

在本教程中，我们学习了如何:

1.  训练一个网络。
2.  将网络权重和优化器状态序列化到磁盘。
3.  加载训练好的网络，对图像进行分类。

在单独的教程**、**中，我们将发现如何在每个时期的*之后将模型的权重保存到磁盘，从而允许我们“检查”我们的网络并选择性能最佳的网络。在实际训练过程中保存模型权重也使我们能够*从一个特定点*重新开始训练，如果我们的网络开始出现过度拟合的迹象。*

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****