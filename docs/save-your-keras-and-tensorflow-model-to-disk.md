# 将 Keras 和 TensorFlow 模型保存到磁盘

> 原文：<https://pyimagesearch.com/2021/05/22/save-your-keras-and-tensorflow-model-to-disk/>

在我们的[之前的教程](https://pyimagesearch.com/2021/05/22/a-gentle-guide-to-training-your-first-cnn-with-keras-and-tensorflow/)中，你学习了如何使用 Keras 库训练你的第一个卷积神经网络。但是，您可能已经注意到，每次您想要评估您的网络或在一组图像上测试它时，您首先需要在进行任何类型的评估之前训练它。这个要求可能会很麻烦。

我们只是在小数据集上使用浅层网络，可以相对快速地进行训练，但如果我们的网络很深，我们需要在更大的数据集上进行训练，因此需要花费许多小时甚至几天来训练，那该怎么办？我们每次都必须投入这么多的时间和资源来训练我们的网络吗？或者，有没有一种方法可以在训练完成后将我们的模型保存到磁盘，然后在我们想要对新图像进行分类时从磁盘加载它？

你肯定有办法。保存和加载已训练模型的过程称为**模型序列化**，是本教程的主要主题。

**要学习如何序列化一个模型，*继续阅读。***

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

## **将模型序列化到磁盘**

使用 Keras 库，模型序列化就像在训练好的模型上调用`model.save`一样简单，然后通过`load_model`函数加载它。在本教程的第一部分，我们将修改来自[之前教程](https://pyimagesearch.com/2021/05/22/a-gentle-guide-to-training-your-first-cnn-with-keras-and-tensorflow/)的一个 ShallowNet 训练脚本，在动物数据集上训练后序列化网络。然后，我们将创建第二个 Python 脚本，演示如何从磁盘加载我们的序列化模型。

让我们从培训部分开始—打开一个新文件，将其命名为`shallownet_train.py`，并插入以下代码:

```py
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
```

****第 2-13 行**** 导入我们需要的 Python 包。这个例子中的大部分代码与[“用 Keras 和 TensorFlow 训练你的第一个 CNN 的温和指南”中的`shallownet_animals.py`相同](https://pyimagesearch.com/2021/05/22/a-gentle-guide-to-training-your-first-cnn-with-keras-and-tensorflow/)为了完整起见，我们将回顾整个文件，我一定会指出为完成模型序列化所做的重要更改，但关于如何在 Animals 数据集上训练 ShallowNet 的详细回顾，请参考[之前的教程](https://pyimagesearch.com/2021/05/22/a-gentle-guide-to-training-your-first-cnn-with-keras-and-tensorflow/)。

接下来，让我们解析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
args = vars(ap.parse_args())
```

我们之前的脚本只需要一个*单个*开关`--dataset`，这是输入动物数据集的路径。但是，如您所见，我们在这里添加了另一个交换机— `--model`，这是训练完成后我们希望*保存网络的路径*。

我们现在可以在`--dataset`中获取图像的路径，初始化我们的预处理器，并从磁盘加载我们的图像数据集:

```py
# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0
```

下一步是将我们的数据划分为训练和测试部分，同时将我们的标签编码为向量:

```py
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
```

通过下面的代码块处理 ShallowNet 训练:

```py
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=32, epochs=100, verbose=1)
```

现在我们的网络已经训练好了，我们需要把它保存到磁盘上。这个过程就像调用`model.save`并提供我们的输出网络保存到磁盘的路径一样简单:

```py
# save the network to disk
print("[INFO] serializing network...")
model.save(args["model"])
```

`.save`方法获取优化器的权重和状态，并以 HDF5 格式将它们序列化到磁盘中。从磁盘加载这些权重就像保存它们一样简单。

在这里，我们评估我们的网络:

```py
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=["cat", "dog", "panda"]))
```

以及绘制我们的损失和准确性:

```py
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
```

要运行我们的脚本，只需执行以下命令:

```py
$ python shallownet_train.py --dataset ../datasets/animals \
	--model shallownet_weights.hdf5
```

网络完成训练后，列出您的目录内容:

```py
$ ls
shallownet_load.py  shallownet_train.py  shallownet_weights.hdf5
```

并且你会看到一个名为`shallownet_weights.hdf5`的文件——这个文件就是我们的连载网络。下一步是获取这个保存的网络并从磁盘加载它。

### 摘要

恭喜你，现在你知道如何将你的 Keras 和 TensorFlow 模型保存到磁盘了。如您所见，这就像调用`model.save`并提供保存输出网络的路径一样简单。现在，您知道了如何训练模型并保存它，这样您就不需要每次训练模型时都从头开始。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****