# 如何在 Keras 和 TensorFlow 中使用 ModelCheckpoint 回调

> 原文：<https://pyimagesearch.com/2021/06/30/how-to-use-the-modelcheckpoint-callback-with-keras-and-tensorflow/>

之前，我们讨论了如何在训练完成后将您的模型保存并序列化到磁盘上。我们还学习了如何发现正在发生的欠拟合和过拟合*，使您能够取消表现不佳的实验，同时保留在训练中表现出希望的模型。*

 *然而，你可能想知道是否有可能将这两种策略结合起来。每当我们的损失/准确性提高时，我们可以序列化模型吗？还是在训练过程中可以只连载*最好的*型号(即损耗最低或精度最高的型号)？你打赌。幸运的是，我们也不需要构建一个定制的回调函数——这个功能已经内置到 Keras 中了。

****要学习如何用 Keras 和 TensorFlow 使用 ModelCheckpoint 回调，*继续阅读。*****

## **如何通过 Keras 和 TensorFlow 使用 ModelCheckpoint 回调**

检查点的一个很好的应用是，每当在训练期间有改进时，将您的网络序列化到磁盘。我们将“改进”定义为损失的*减少*或准确度的*增加*——我们将在实际的 Keras 回调中设置该参数。

在本例中，我们将在 CIFAR-10 数据集上训练 MiniVGGNet 架构，然后在每次模型性能提高时将我们的网络权重序列化到磁盘。首先，打开一个新文件，将其命名为`cifar10_checkpoint_improvements.py`，并插入以下代码:

```py
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import argparse
import os
```

**第 2-8 行**导入我们需要的 Python 包。请注意在**行 4** 上导入的`ModelCheckpoint`类——这个类将使我们能够在发现模型性能有所提高时检查点并序列化我们的网络到磁盘。

接下来，让我们解析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
	help="path to weights directory")
args = vars(ap.parse_args())
```

我们需要的唯一命令行参数是`--weights`，它是输出目录的路径，该目录将在训练过程中存储我们的序列化模型。然后，我们执行从磁盘加载 CIFAR-10 数据集的标准例程，将像素强度缩放到范围`[0, 1]`，然后对标签进行一次性编码:

```py
# load the training and testing data, then scale it into the
# range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
```

给定我们的数据，我们现在准备初始化我们的 SGD 优化器以及 MiniVGGNet 架构:

```py
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
```

我们将使用初始学习率为 *α* = 0 *的 SGD 优化器。然后在 40 个时期的过程中缓慢衰减。我们也将应用一个动量 *γ* = 0 *。* 9，表示也应使用内斯特罗夫加速度。*

MiniVGGNet 架构被实例化为接受宽度为 32 像素、高度为 32 像素、深度为 3(通道数)的输入图像。我们设置`classes=10`,因为 CIFAR-10 数据集有十个可能的类标签。

对我们的网络进行检查点操作的关键步骤可以在下面的代码块中找到:

```py
# construct the callback to save only the *best* model to disk
# based on the validation loss
fname = os.path.sep.join([args["weights"],
	"weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
	save_best_only=True, verbose=1)
callbacks = [checkpoint]
```

在**的第 37 行和第 38 行**，我们构造了一个特殊的文件名(`fname`)模板字符串，Keras 在将我们的模型写到磁盘时使用它。模板中的第一个变量`{epoch:03d}`是我们的纪元编号，写成三位数。

第二个变量是我们想要监控改进的度量，`{val_loss:.4f}`，即在当前时期设置的验证损失本身。当然，如果我们想监控验证的准确性，我们可以用`val_acc`代替`val_loss`。相反，如果我们想要监控*训练*损失和准确性，变量将分别变成`train_loss`和`train_acc`(尽管我会建议*监控您的验证度量*，因为它们会让您更好地了解您的模型将如何概括)。

一旦定义了输出文件名模板，我们就在第 39 行和第 40 行的**上实例化`ModelCheckpoint`类。`ModelCheckpoint`的第一个参数是代表文件名模板的字符串。然后我们把我们想要的传递给`monitor`。在这种情况下，我们希望监控验证损失(`val_loss`)。**

`mode`参数控制`ModelCheckpoint`是否应该寻找使*最小化*我们的度量或者*最大化*的值。既然我们正在处理损失，越低越好，所以我们设置`mode="min"`。如果我们改为使用`val_acc`，我们将设置`mode="max"`(因为精度越高越好)。

设置`save_best_only=True`确保最新的最佳模型(根据监控的指标)不会被覆盖。最后，当一个模型在训练期间被序列化到磁盘时，`verbose=1`设置简单地记录一个通知到我们的终端。

**第 41 行**然后构造一个`callbacks`列表——我们唯一需要的回调是我们的`checkpoint`。

最后一步是简单地训练网络，让我们的`checkpoint`去处理剩下的事情:

```py
# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=64, epochs=40, callbacks=callbacks, verbose=2)
```

要执行我们的脚本，只需打开一个终端并执行以下命令:

```py
$ python cifar10_checkpoint_improvements.py --weights weights/improvements
[INFO] loading CIFAR-10 data...
[INFO] compiling model...
[INFO] training network...
Train on 50000 samples, validate on 10000 samples
Epoch 1/40
171s - loss: 1.6700 - acc: 0.4375 - val_loss: 1.2697 - val_acc: 0.5425
Epoch 2/40
Epoch 00001: val_loss improved from 1.26973 to 0.98481, saving model to test/
	weights-001-0.9848.hdf5
...
Epoch 40/40
Epoch 00039: val_loss did not improve
315s - loss: 0.2594 - acc: 0.9075 - val_loss: 0.5707 - val_acc: 0.8190
```

正如我们可以从我的终端输出和**图 1** 中看到的，每次验证损失减少时，我们都会将新的序列化模型保存到磁盘上。

在培训过程结束时，我们有 18 个单独的文件，每个文件代表一个增量改进:

```py
$ find ./  -printf "%f\n" | sort
./
weights-000-1.2697.hdf5
weights-001-0.9848.hdf5
weights-003-0.8176.hdf5
weights-004-0.7987.hdf5
weights-005-0.7722.hdf5
weights-006-0.6925.hdf5
weights-007-0.6846.hdf5
weights-008-0.6771.hdf5
weights-009-0.6212.hdf5
weights-012-0.6121.hdf5
weights-013-0.6101.hdf5
weights-014-0.5899.hdf5
weights-015-0.5811.hdf5
weights-017-0.5774.hdf5
weights-019-0.5740.hdf5
weights-022-0.5724.hdf5
weights-024-0.5628.hdf5
weights-033-0.5546.hdf5
```

如您所见，每个文件名都有三个组成部分。第一个是静态字符串， *weights* 。然后我们有了*纪元编号*。文件名的最后一部分是我们衡量改进的指标，在本例中是*验证损失*。

我们最好的验证损失是在第 33 个时段获得的，值为 0 *。* 5546。然后我们可以把这个模型从磁盘上载入。

请记住，你的结果将不会匹配我的，因为网络是随机的，并用随机变量初始化。根据初始值的不同，您可能会有显著不同的模型检查点，但在训练过程结束时，我们的网络应该会获得类似的准确性(几个百分点)。

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

### **只检查最佳神经网络**

也许检查点增量改进的最大缺点是，我们最终会得到一堆我们(不太可能)感兴趣的额外文件，如果我们的验证损失在训练时期上下波动，这种情况尤其如此——这些增量改进中的每一个都将被捕获并序列化到磁盘上。在这种情况下，最好只保存*一个*模型，并在训练期间每次我们的指标提高时，简单地*覆盖它*。

幸运的是，完成这个动作很简单，只需更新`ModelCheckpoint`类来接受一个*简单字符串*(即，一个没有任何模板变量的文件路径*)。然后，每当我们的指标提高时，该文件就会被简单地覆盖。为了理解这个过程，让我们创建第二个名为`cifar10_checkpoint_best.py`的 Python 文件，并回顾一下其中的区别。*

首先，我们需要导入所需的 Python 包:

```py
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import argparse
```

然后解析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
	help="path to best model weights file")
args = vars(ap.parse_args())
```

命令行参数本身的*名称*是相同的(`--weights`)，但是开关的*描述*现在是不同的:“路径到*最佳*模型权重*文件*”因此，这个命令行参数将是一个输出路径的简单字符串——没有模板应用于这个字符串。

从那里，我们可以加载我们的 CIFAR-10 数据集，并为训练做准备:

```py
# load the training and testing data, then scale it into the
# range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
```

以及初始化我们的 SGD 优化器和 MiniVGGNet 架构:

```py
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
```

我们现在准备更新`ModelCheckpoint`代码:

```py
# construct the callback to save only the *best* model to disk
# based on the validation loss
checkpoint = ModelCheckpoint(args["weights"], monitor="val_loss",
	save_best_only=True, verbose=1)
callbacks = [checkpoint]
```

注意`fname`模板字符串是如何消失的——我们所做的就是将`--weights`的值提供给`ModelCheckpoint`。由于没有需要填写的模板值，每当我们的监控指标提高时(在这种情况下，验证损失)，Keras 将简单地*覆盖*现有的序列化权重文件。

最后，我们在网络上训练下面的代码块:

```py
# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=64, epochs=40, callbacks=callbacks, verbose=2)
```

要执行我们的脚本，发出以下命令:

```py
$ python cifar10_checkpoint_best.py \
	--weights weights/best/cifar10_best_weights.hdf5
[INFO] loading CIFAR-10 data...
[INFO] compiling model...
[INFO] training network...
Train on 50000 samples, validate on 10000 samples
Epoch 1/40
Epoch 00000: val_loss improved from inf to 1.26677, saving model to
	test_best/cifar10_best_weights.hdf5
305s - loss: 1.6657 - acc: 0.4441 - val_loss: 1.2668 - val_acc: 0.5584
Epoch 2/40
Epoch 00001: val_loss improved from 1.26677 to 1.21923, saving model to
	test_best/cifar10_best_weights.hdf5
309s - loss: 1.1996 - acc: 0.5828 - val_loss: 1.2192 - val_acc: 0.5798
...
Epoch 40/40
Epoch 00039: val_loss did not improve
173s - loss: 0.2615 - acc: 0.9079 - val_loss: 0.5511 - val_acc: 0.8250
```

在这里，您可以看到，如果我们的验证损失减少，我们仅使用更新的网络*覆盖我们的`cifar10_best_weights.hdf5`文件*。这有两个主要好处:

1.  训练过程的最后只有*一个*序列化文件——获得最低损失的模型历元。
2.  我们没有捕捉到亏损上下波动的“增量改善”。相反，如果我们的度量获得的损失低于*所有之前的*时期，我们只保存和覆盖*现有的*最佳模型。

为了证实这一点，请看一下我的`weights/best`目录，在那里您可以看到只有一个输出文件:

```py
$ ls -l weights/best/
total 17024
-rw-rw-r-- 1 adrian adrian 17431968 Apr 28 09:47 cifar10_best_weights.hdf5
```

然后，您可以获取这个序列化的 MiniVGGNet，并根据测试数据对其进行进一步评估，或者将其应用到您自己的映像中。

## **总结**

在本教程中，我们回顾了如何监控给定的指标(例如，验证损失、验证准确性等。)然后将高性能网络保存到磁盘。在 Keras 中有两种方法可以实现这一点:

1.  检查点*增量*改进。
2.  检查点*仅*流程中找到的最佳模型。

就个人而言，我更喜欢后者而不是前者，因为它产生更少的文件和一个表示在训练过程中找到的最佳时期的输出文件。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****