# 使用 Keras 调谐器和 TensorFlow 轻松调整超参数

> 原文：<https://pyimagesearch.com/2021/06/07/easy-hyperparameter-tuning-with-keras-tuner-and-tensorflow/>

在本教程中，您将学习如何使用 Keras Tuner 软件包通过 Keras 和 TensorFlow 轻松进行超参数调谐。

本教程是我们关于超参数调优的四部分系列的第四部分:

1.  [*使用 scikit-learn 和 Python*](https://pyimagesearch.com/2021/05/17/introduction-to-hyperparameter-tuning-with-scikit-learn-and-python/) 进行超参数调优的介绍(本系列的第一篇教程)
2.  [*【网格搜索超参数调优】用 scikit-learn(GridSearchCV)*](https://pyimagesearch.com/2021/05/24/grid-search-hyperparameter-tuning-with-scikit-learn-gridsearchcv/)(教程来自两周前)
3.  [*用 scikit-learn、Keras 和 TensorFlow*](https://pyimagesearch.com/2021/05/31/hyperparameter-tuning-for-deep-learning-with-scikit-learn-keras-and-tensorflow/) 进行深度学习的超参数调优(上周的帖子)
4.  *使用 Keras 调谐器和 TensorFlow 轻松调整超参数*(今天的帖子)

上周我们学习了如何使用 scikit-learn 与 Keras 和 TensorFlow 进行交互，以执行随机交叉验证的超参数搜索。

然而，还有更先进的超参数调整算法，包括**贝叶斯超参数优化**和**超带**，这是对传统随机超参数搜索的适应和改进。

贝叶斯优化和 Hyperband 都在 [keras 调谐器包](https://keras-team.github.io/keras-tuner/)中实现。正如我们将看到的，在您自己的深度学习脚本中使用 Keras Tuner 就像单个导入和单个类实例化一样简单——从那里开始，它就像训练您的神经网络一样简单！

除了易于使用，你会发现 Keras 调谐器:

1.  集成到您现有的深度学习培训管道中，只需最少的代码更改
2.  实现新的超参数调整算法
3.  可以用最少的努力提高准确性

**要了解如何使用 Keras Tuner 调整超参数，** ***继续阅读。***

## **使用 Keras 调谐器和 TensorFlow 轻松调谐超参数**

在本教程的第一部分，我们将讨论 Keras Tuner 包，包括它如何帮助用最少的代码自动调整模型的超参数。

然后，我们将配置我们的开发环境，并检查我们的项目目录结构。

我们今天要回顾几个 Python 脚本，包括:

1.  我们的配置文件
2.  模型架构定义(我们将调整超参数，包括 CONV 层中的过滤器数量、学习速率等。)
3.  绘制我们训练历史的实用程序
4.  一个驱动程序脚本，它将所有部分粘合在一起，并允许我们测试各种超参数优化算法，包括贝叶斯优化、超波段和传统的随机搜索

我们将讨论我们的结果来结束本教程。

### **什么是 Keras Tuner，它如何帮助我们自动调整超参数？**

[上周，您学习了如何使用 scikit-learn 的超参数搜索功能](https://pyimagesearch.com/2021/05/31/hyperparameter-tuning-for-deep-learning-with-scikit-learn-keras-and-tensorflow/)来调整基本前馈神经网络的超参数(包括批量大小、要训练的时期数、学习速率和给定层中的节点数)。

虽然这种方法工作得很好(并且给了我们一个很好的准确性提升)，但是代码并不一定“漂亮”

更重要的是，*它使我们不容易调整模型架构的“内部”参数*(例如，CONV 层中过滤器的数量、步幅大小、池的大小、辍学率等。).

像 [keras tuner](https://keras-team.github.io/keras-tuner/) 这样的库使得以一种*有机的方式将超参数优化实现到我们的训练脚本中变得非常简单*:

*   当我们实现我们的模型架构时，我们定义我们想要为给定的参数搜索什么范围(例如，我们的第一 CONV 层中的过滤器的数量，第二 CONV 层中的过滤器的数量，等等)。)
*   然后我们定义一个`Hyperband`、`RandomSearch`或`BayesianOptimization`的实例
*   keras tuner 包负责剩下的工作，运行多次试验，直到我们收敛到最佳的超参数集

这听起来可能很复杂，但是一旦你深入研究了代码，这就很容易了。

此外，如果您有兴趣了解更多关于 Hyperband 算法的信息，请务必阅读李等人的 2018 年出版物， [*Hyperband:一种基于 Bandit 的超参数优化新方法*](https://arxiv.org/abs/1603.06560) 。

要了解关于贝叶斯超参数优化的更多信息，请参考多伦多大学教授兼研究员[罗杰·格罗斯](https://www.cs.toronto.edu/~rgrosse/)的[幻灯片](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/slides/lec21.pdf)。

### **配置您的开发环境**

要遵循本指南，您需要安装 TensorFlow、OpenCV、scikit-learn 和 Keras Tuner。

所有这些软件包都是 pip 可安装的:

```py
$ pip install tensorflow # use "tensorflow-gpu" if you have a GPU
$ pip install opencv-contrib-python
$ pip install scikit-learn
$ pip install keras-tuner
```

此外，这两个指南提供了在您的计算机上安装 Keras 和 TensorFlow 的更多详细信息、帮助和提示:

*   [*如何在 Ubuntu 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   [*如何在 macOS 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)

这两个教程都有助于在一个方便的 Python 虚拟环境中为您的系统配置这篇博客文章所需的所有软件。

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

在我们可以使用 Keras Tuner 来调优我们的 Keras/TensorFlow 模型的超参数之前，让我们首先回顾一下我们的项目目录结构。

通过访问本教程的 ***【下载】*** 部分来检索源代码。

从那里，您将看到以下目录结构:

```py
$ tree . --dirsfirst --filelimit 10
.
├── output
│   ├── bayesian [12 entries exceeds filelimit, not opening dir]
│   ├── hyperband [79 entries exceeds filelimit, not opening dir]
│   └── random [12 entries exceeds filelimit, not opening dir]
│   ├── bayesian_plot.png
│   ├── hyperband_plot.png
│   └── random_plot.png
├── pyimagesearch
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   └── utils.py
└── train.py

2 directories, 8 files
```

在`pyimagesearch`模块中，我们有三个 Python 脚本:

1.  `config.py`:包含重要的配置选项，例如输出路径目录、输入图像尺寸以及数据集中唯一类标签的数量
2.  `model.py`:包含`build_model`函数，负责实例化我们模型架构的一个实例；该函数设置将要调整的超参数以及每个超参数的适当取值范围
3.  `utils.py`:实现`save_plot`，一个助手/便利函数，生成训练历史图

`train.py`脚本使用`pyimagesearch`模块中的每个实现来执行三种类型的超参数搜索:

1.  超波段
2.  随意
3.  贝叶斯优化

每个实验的结果都保存在`output`目录中。为每个实验使用专用输出目录的主要好处是，您可以启动、停止和恢复超参数调整实验。这一点*尤其重要*，因为超参数调整可能需要相当长的时间。

### **创建我们的配置文件**

在使用 Keras Tuner 优化超参数之前，我们首先需要创建一个配置文件来存储重要的变量。

打开项目目录结构中的`config.py`文件，插入以下代码:

```py
# define the path to our output directory
OUTPUT_PATH = "output"

# initialize the input shape and number of classes
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
```

**第 2 行**定义了我们的输出目录路径(即存储训练历史图和超参数调整实验日志的位置)。

在这里，我们定义了数据集中图像的输入空间维度以及唯一类标签的总数(**第 5 行和第 6 行**)。

下面我们定义我们的训练变量:

```py
# define the total number of epochs to train, batch size, and the
# early stopping patience
EPOCHS = 50
BS = 32
EARLY_STOPPING_PATIENCE = 5
```

对于每个实验，我们将允许我们的模型训练*最大*的`50`个时期。我们将在每个实验中使用批量`32`。

对于没有显示有希望迹象的短路实验，我们定义了一个早期停止耐心`5`，这意味着如果我们的精度在`5`个时期后没有提高，我们将终止训练过程，并继续进行下一组超参数。

调整超参数是一个计算量非常大的过程。如果我们可以通过取消表现不佳的实验来减少需要进行的实验数量，我们就可以节省大量的时间。

### **实现我们的绘图助手功能**

在为我们的模型找到最佳超参数后，我们将希望在这些超参数上训练模型，并绘制我们的训练历史(包括训练集和验证集的损失和准确性)。

为了简化这个过程，我们可以在`utils.py`文件中定义一个`save_plot`助手函数。

现在打开这个文件，让我们来看看:

```py
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary package
import matplotlib.pyplot as plt

def save_plot(H, path):
	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig(path)
```

`save_plot`函数需要我们传入两个变量:通过调用`model.fit`获得的训练历史`H`和输出图的`path`。

然后，我们绘制训练损失、验证损失、训练准确度和验证准确度。

结果图保存到输出`path`。

### **创建我们的 CNN**

可以说，本教程最重要的部分是定义我们的 CNN 架构，也就是说，因为这是我们设置想要调谐的超参数的地方。

打开`pyimagesearch`模块内的`model.py`文件，让我们看看发生了什么:

```py
# import the necessary packages
from . import config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
```

**2-11 行**导入我们需要的包。请注意我们是如何导入我们在本指南前面创建的`config`文件的。

如果您以前使用 Keras 和 TensorFlow 创建过 CNN，那么这些导入的其余部分对您来说应该很熟悉。如果没有，建议你看我的 [Keras 教程](https://pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/)，连同我的书 [*用 Python 进行计算机视觉的深度学习*](https://pyimagesearch.com/deep-learning-computer-vision-python-book/) *。*

现在让我们构建我们的模型:

```py
def build_model(hp):
	# initialize the model along with the input shape and channel
	# dimension
	model = Sequential()
	inputShape = config.INPUT_SHAPE
	chanDim = -1

	# first CONV => RELU => POOL layer set
	model.add(Conv2D(
		hp.Int("conv_1", min_value=32, max_value=96, step=32),
		(3, 3), padding="same", input_shape=inputShape))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))
```

`build_model`函数接受单个对象`hp`，这是我们从 Keras Tuner 得到的超参数调整对象。在本教程的后面，我们将在我们的驱动脚本`train.py`中创建`hp`。

**第 16-18 行**初始化我们的`model`，获取我们数据集中输入图像的空间维度，并设置通道排序(假设“通道最后”)。

从那里，**行 21-26** 定义了我们的第一个 conv =>=>池层集，最重要的一行是**行 22。**

这里，我们定义了第一个要搜索的超参数——conv 层中过滤器的数量。

由于 CONV 层中滤镜的数量是一个整数，我们使用`hp.Int`创建一个整数超参数对象。

超参数被命名为`conv_1`，可以接受范围*【32，96】*内的值，步长为`32`。这意味着`conv_1`的有效值是`32, 64, 96`。

我们的超参数调谐器将*自动*为这个 CONV 层选择最大化精度的最佳值。

同样，我们对第二个 CONV => RELU = >池层集做同样的事情:

```py
	# second CONV => RELU => POOL layer set
	model.add(Conv2D(
		hp.Int("conv_2", min_value=64, max_value=128, step=32),
		(3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))
```

对于我们的第二个 CONV 层，我们允许在*【64，128】范围内学习更多的过滤器。*步长为`32`，这意味着我们将测试`64, 96, 128`的值。

我们将对完全连接的节点数量做类似的事情:

```py
	# first (and only) set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(hp.Int("dense_units", min_value=256,
		max_value=768, step=256)))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	# softmax classifier
	model.add(Dense(config.NUM_CLASSES))
	model.add(Activation("softmax"))
```

**第 38 行和第 39 行**定义了我们的 FC 层。我们希望调整这一层中的节点数量。我们指定最少`256`和最多`768`个节点，允许一步`256`。

我们的下一个代码块使用了`hp.Choice`函数:

```py
	# initialize the learning rate choices and optimizer
	lr = hp.Choice("learning_rate",
		values=[1e-1, 1e-2, 1e-3])
	opt = Adam(learning_rate=lr)

	# compile the model
	model.compile(optimizer=opt, loss="categorical_crossentropy",
		metrics=["accuracy"])

	# return the model
	return model
```

对于我们的学习率，我们希望看到`1e-1`、`1e-2`和`1e-3`中哪一个表现最好。使用`hp.Choice`将允许我们的超参数调谐器选择最佳学习率。

最后，我们编译模型并将其返回给调用函数。

### **使用 Keras 调谐器实现超参数调谐**

让我们把所有的部分放在一起，学习如何使用 Keras Tuner 库来调整 Keras/TensorFlow 超参数。

打开项目目录结构中的`train.py`文件，让我们开始吧:

```py
# import the necessary packages
from pyimagesearch import config
from pyimagesearch.model import build_model
from pyimagesearch import utils
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report
import kerastuner as kt
import numpy as np
import argparse
import cv2
```

**第 2-13 行**导入我们需要的 Python 包。值得注意的进口包括:

*   `config`:我们的配置文件
*   `build_model`:接受一个超参数调整对象，该对象选择各种值来测试 CONV 滤波器、FC 节点和学习率——生成的模型被构建并返回给调用函数
*   `utils`:用于绘制我们的训练历史
*   `EarlyStopping`:一个 Keras/TensorFlow 回调，用于缩短运行不佳的超参数调整实验
*   `fashion_mnist`:我们将在[时尚 MNIST 数据集](https://pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/)上训练我们的模特
*   `kerastuner`:用于实现超参数调谐的 Keras 调谐器包

接下来是我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tuner", required=True, type=str,
	choices=["hyperband", "random", "bayesian"],
	help="type of hyperparameter tuner we'll be using")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())
```

我们有两个命令行参数要解析:

1.  我们将使用的超参数优化器的类型
2.  输出训练历史图的路径

在那里，从磁盘加载时尚 MNIST 数据集:

```py
# load the Fashion MNIST dataset
print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

# add a channel dimension to the dataset
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# one-hot encode the training and testing labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# initialize the label names
labelNames = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]
```

**第 26 行**装载时装 MNIST，预分割成训练和测试集。

然后，我们向数据集添加一个通道维度(**行 29 和 30** )，将像素强度从范围*【0，255】*缩放到*【0，1】*(**行 33 和 34** )，然后对标签进行一次性编码(**行 37 和 38** )。

正如在这个脚本的导入部分提到的，我们将使用`EarlyStopping`来缩短运行不佳的超参数试验:

```py
# initialize an early stopping callback to prevent the model from
# overfitting/spending too much time training with minimal gains
es = EarlyStopping(
	monitor="val_loss",
	patience=config.EARLY_STOPPING_PATIENCE,
	restore_best_weights=True)
```

我们会监控验证损失。如果验证损失在`EARLY_STOPPING_PATIENCE`总时期后未能显著改善，那么我们将终止试验并继续下一个试验。

请记住，调整超参数是一个*极其*计算量巨大的过程，因此，如果我们能够取消表现不佳的试验，我们就可以节省大量*时间。*

下一步是初始化我们的超参数优化器:

```py
# check if we will be using the hyperband tuner
if args["tuner"] == "hyperband":
	# instantiate the hyperband tuner object
	print("[INFO] instantiating a hyperband tuner object...")
	tuner = kt.Hyperband(
		build_model,
		objective="val_accuracy",
		max_epochs=config.EPOCHS,
		factor=3,
		seed=42,
		directory=config.OUTPUT_PATH,
		project_name=args["tuner"])
```

**第 52-62 行**处理我们是否希望使用 Hyperband 调谐器。Hyperband 调谐器是随机搜索与“自适应资源分配和提前停止”的结合它实质上是李等人的论文， *Hyperband:一种新的基于 Bandit 的超参数优化方法的实现。*

如果我们提供一个值`random`作为我们的`--tuner`命令行参数，那么我们将使用一个基本的随机超参数搜索:

```py
# check if we will be using the random search tuner
elif args["tuner"] == "random":
	# instantiate the random search tuner object
	print("[INFO] instantiating a random search tuner object...")
	tuner = kt.RandomSearch(
		build_model,
		objective="val_accuracy",
		max_trials=10,
		seed=42,
		directory=config.OUTPUT_PATH,
		project_name=args["tuner"])
```

否则，我们将假设我们正在使用[贝叶斯优化](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Bayesian_optimization):

```py
# otherwise, we will be using the bayesian optimization tuner
else:
	# instantiate the bayesian optimization tuner object
	print("[INFO] instantiating a bayesian optimization tuner object...")
	tuner = kt.BayesianOptimization(
		build_model,
		objective="val_accuracy",
		max_trials=10,
		seed=42,
		directory=config.OUTPUT_PATH,
		project_name=args["tuner"])
```

一旦我们的超参数调谐器被实例化，我们可以搜索空间:

```py
# perform the hyperparameter search
print("[INFO] performing hyperparameter search...")
tuner.search(
	x=trainX, y=trainY,
	validation_data=(testX, testY),
	batch_size=config.BS,
	callbacks=[es],
	epochs=config.EPOCHS
)

# grab the best hyperparameters
bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
print("[INFO] optimal number of filters in conv_1 layer: {}".format(
	bestHP.get("conv_1")))
print("[INFO] optimal number of filters in conv_2 layer: {}".format(
	bestHP.get("conv_2")))
print("[INFO] optimal number of units in dense layer: {}".format(
	bestHP.get("dense_units")))
print("[INFO] optimal learning rate: {:.4f}".format(
	bestHP.get("learning_rate")))
```

**第 90-96 行**开始超参数调整过程。

调谐过程完成后，我们获得最佳超参数(**行 99** )并在终端上显示最佳:

*   第一个 CONV 图层中的滤镜数量
*   第二 CONV 层中的过滤器数量
*   全连接层中的节点数
*   最佳学习率

一旦我们有了最好的超参数，我们需要基于它们实例化一个新的`model`:

```py
# build the best model and train it
print("[INFO] training the best model...")
model = tuner.hypermodel.build(bestHP)
H = model.fit(x=trainX, y=trainY,
	validation_data=(testX, testY), batch_size=config.BS,
	epochs=config.EPOCHS, callbacks=[es], verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

# generate the training loss/accuracy plot
utils.save_plot(H, args["plot"])
```

**第 111 行**负责用我们最好的超参数建立模型。

在**行 112-114** 上对`model.fit`的调用在最佳超参数上训练我们的模型。

训练完成后，我们对测试集进行全面评估(**第 118-120 行**)。

最后，使用我们的`save_plot`实用函数将生成的训练历史图保存到磁盘。

### **超波段超参数调谐**

让我们看看应用 Keras 调谐器的超波段优化器的结果。

通过访问本教程的 ***【下载】*** 部分来检索源代码。

从那里，打开一个终端并执行以下命令:

```py
$ time python train.py --tuner hyperband --plot output/hyperband_plot.png
[INFO] loading Fashion MNIST...
[INFO] instantiating a hyperband tuner object..."
[INFO] performing hyperparameter search...

Search: Running Trial #1

Hyperparameter    |Value             |Best Value So Far
conv_1            |96                |?
conv_2            |96                |?
dense_units       |512               |?
learning_rate     |0.1               |?

Epoch 1/2
1875/1875 [==============================] - 119s 63ms/step - loss: 3.2580 - accuracy: 0.6568 - val_loss: 3.9679 - val_accuracy: 0.7852
Epoch 2/2
1875/1875 [==============================] - 79s 42ms/step - loss: 3.5280 - accuracy: 0.7710 - val_loss: 2.5392 - val_accuracy: 0.8167

Trial 1 Complete [00h 03m 18s]
val_accuracy: 0.8166999816894531

Best val_accuracy So Far: 0.8285999894142151
Total elapsed time: 00h 03m 18s
```

Keras 调谐器包通过运行几个“试验”来工作在这里，我们可以看到，在第一次试验中，我们将对第一个 CONV 层使用`96`过滤器，对第二个 CONV 层使用`96`过滤器，对全连接层使用总共`512`个节点，学习速率为`0.1`。

随着我们试验的结束，`Best Value So Far`栏将会更新，以反映找到的最佳超参数。

但是，请注意，我们仅针对总共两个时期训练该模型——这是由于我们的`EarlyStopping`停止标准。**如果我们的验证准确性没有提高一定的量，我们将缩短训练过程，以避免花费太多时间探索不会显著提高我们准确性的超参数。**

由此可见，在第一次试验结束时，我们坐在 **![\pmb\approx](img/e36579093a0fcb827143090e30c2b0c2.png "\pmb\approx")的准确率为 82%。**

现在让我们跳到最后的审判:

```py
Search: Running Trial #76

Hyperparameter    |Value             |Best Value So Far   
conv_1            |32                |64
conv_2            |64                |128
dense_units       |768               |512
learning_rate     |0.01              |0.001

Epoch 1/17
1875/1875 [==============================] - 41s 22ms/step - loss: 0.8586 - accuracy: 0.7624 - val_loss: 0.4307 - val_accuracy: 0.8587
...
Epoch 17/17
1875/1875 [==============================] - 40s 21ms/step - loss: 0.2248 - accuracy: 0.9220 - val_loss: 0.3391 - val_accuracy: 0.9089

Trial 76 Complete [00h 11m 29s]
val_accuracy: 0.9146000146865845

Best val_accuracy So Far: 0.9289000034332275
Total elapsed time: 06h 34m 56s
```

**目前发现的最好的验证准确率是![\pmb\approx](img/e36579093a0fcb827143090e30c2b0c2.png "\pmb\approx")92%。**

Hyperband 完成运行后，我们会看到终端上显示的最佳参数:

```py
[INFO] optimal number of filters in conv_1 layer: 64
[INFO] optimal number of filters in conv_2 layer: 128
[INFO] optimal number of units in dense layer: 512
[INFO] optimal learning rate: 0.0010
```

对于我们的第一个 CONV 层，我们看到`64`过滤器是最好的。网络中的下一个 CONV 层喜欢`128`层——这并不是一个完全令人惊讶的发现。通常，随着我们深入 CNN，随着体积大小*的空间维度减少*，过滤器的数量*增加。*

AlexNet、VGGNet、ResNet 和几乎所有其他流行的 CNN 架构都有这种类型的模式。

最终的 FC 层有`512`个节点，而我们的最优学习速率是`1e-3`。

现在让我们用这些超参数来训练 CNN:

```py
[INFO] training the best model...
Epoch 1/50
1875/1875 [==============================] - 69s 36ms/step - loss: 0.5655 - accuracy: 0.8089 - val_loss: 0.3147 - val_accuracy: 0.8873
...
Epoch 11/50
1875/1875 [==============================] - 67s 36ms/step - loss: 0.1163 - accuracy: 0.9578 - val_loss: 0.3201 - val_accuracy: 0.9088
[INFO] evaluating network...
              precision    recall  f1-score   support

         top       0.83      0.92      0.87      1000
     trouser       0.99      0.99      0.99      1000
    pullover       0.83      0.92      0.87      1000
       dress       0.93      0.93      0.93      1000
        coat       0.90      0.83      0.87      1000
      sandal       0.99      0.98      0.99      1000
       shirt       0.82      0.70      0.76      1000
     sneaker       0.94      0.99      0.96      1000
         bag       0.99      0.98      0.99      1000
  ankle boot       0.99      0.95      0.97      1000

    accuracy                           0.92     10000
   macro avg       0.92      0.92      0.92     10000
weighted avg       0.92      0.92      0.92     10000

real    407m28.169s
user    2617m43.104s
sys     51m46.604s
```

在我们最好的超参数上训练 50 个时期后，我们在我们的验证集上获得了 **![\pmb\approx](img/e36579093a0fcb827143090e30c2b0c2.png "\pmb\approx")92%的准确度**。

在我的 3 GHz 英特尔至强 W 处理器上，总的超参数搜索和训练时间为![\approx](img/efc44d3feeb891408f5038a6a5a7c926.png "\approx")

6.7 hours. Using a GPU would reduce the training time *considerably.*

### **随机搜索超参数调谐**

现在让我们来看一个普通的随机搜索。

同样，请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

在那里，您可以执行以下命令:

```py
$ time python train.py --tuner random --plot output/random_plot.png
[INFO] loading Fashion MNIST...
[INFO] instantiating a random search tuner object...
[INFO] performing hyperparameter search...

Search: Running Trial #1

Hyperparameter    |Value             |Best Value So Far
conv_1            |64                |?
conv_2            |64                |?
dense_units       |512               |?
learning_rate     |0.01              |?

Epoch 1/50
1875/1875 [==============================] - 51s 27ms/step - loss: 0.7210 - accuracy: 0.7758 - val_loss: 0.4748 - val_accuracy: 0.8668
...
Epoch 14/50
1875/1875 [==============================] - 49s 26ms/step - loss: 0.2180 - accuracy: 0.9254 - val_loss: 0.3021 - val_accuracy: 0.9037

Trial 1 Complete [00h 12m 08s]
val_accuracy: 0.9139999747276306

Best val_accuracy So Far: 0.9139999747276306
Total elapsed time: 00h 12m 08s
```

在第一次试验结束时，我们获得了![\approx](img/efc44d3feeb891408f5038a6a5a7c926.png "\approx")

91% accuracy on our validation set with `64` filters for the first CONV layer, `64` filters for the second CONV layer, a total of `512` nodes in the FC layer, and a learning rate of `1e-2`.

到第 10 次试验时，我们的准确率有所提高，但没有使用 Hyperband 时的进步大:

```py
Search: Running Trial #10

Hyperparameter    |Value             |Best Value So Far   
conv_1            |96                |96
conv_2            |64                |64
dense_units       |512               |512
learning_rate     |0.1               |0.001

Epoch 1/50
1875/1875 [==============================] - 64s 34ms/step - loss: 3.8573 - accuracy: 0.6515 - val_loss: 1.3178 - val_accuracy: 0.7907
...
Epoch 6/50
1875/1875 [==============================] - 63s 34ms/step - loss: 4.2424 - accuracy: 0.8176 - val_loss: 622.4448 - val_accuracy: 0.8295

Trial 10 Complete [00h 06m 20s]
val_accuracy: 0.8640999794006348
Total elapsed time: 01h 47m 02s

Best val_accuracy So Far: 0.9240000247955322
Total elapsed time: 01h 47m 02s
```

我们现在到了![\approx](img/efc44d3feeb891408f5038a6a5a7c926.png "\approx")

92% accuracy. Still, the good news is that we’ve only spent 1h47m exploring the hyperparameter space (as opposed to ![\approx](img/efc44d3feeb891408f5038a6a5a7c926.png "\approx")6h30m from the Hyperband trials).

下面我们可以看到随机搜索找到的最佳超参数:

```py
[INFO] optimal number of filters in conv_1 layer: 96
[INFO] optimal number of filters in conv_2 layer: 64
[INFO] optimal number of units in dense layer: 512
[INFO] optimal learning rate: 0.0010
```

我们随机搜索的输出与超波段调谐的输出略有不同。第一层 CONV 有`96`滤镜，第二层有`64` (Hyperband 分别有`64`和`128`)。

也就是说，随机搜索和 Hyperband 都同意 FC 层中的`512`节点和`1e-3`的学习速率。

经过培训后，我们达到了与 Hyperband 大致相同的验证准确度:

```py
[INFO] training the best model...
Epoch 1/50
1875/1875 [==============================] - 64s 34ms/step - loss: 0.5682 - accuracy: 0.8157 - val_loss: 0.3227 - val_accuracy: 0.8861
...
Epoch 13/50
1875/1875 [==============================] - 63s 34ms/step - loss: 0.1066 - accuracy: 0.9611 - val_loss: 0.2636 - val_accuracy: 0.9251
[INFO] evaluating network...
              precision    recall  f1-score   support

         top       0.85      0.91      0.88      1000
     trouser       0.99      0.98      0.99      1000
    pullover       0.88      0.89      0.88      1000
       dress       0.94      0.90      0.92      1000
        coat       0.82      0.93      0.87      1000
      sandal       0.97      0.99      0.98      1000
       shirt       0.82      0.69      0.75      1000
     sneaker       0.96      0.95      0.96      1000
         bag       0.99      0.99      0.99      1000
  ankle boot       0.97      0.96      0.97      1000

    accuracy                           0.92     10000
   macro avg       0.92      0.92      0.92     10000
weighted avg       0.92      0.92      0.92     10000

real    120m52.354s
user    771m17.324s
sys     15m10.248s
```

虽然 **![\pmb\approx](img/e36579093a0fcb827143090e30c2b0c2.png "\pmb\approx")92%的准确率**与 Hyperband 基本相同，**一次随机搜索将我们的超参数搜索时间缩短了 3 倍，这本身就是一个** ***巨大的*** **改进。**

### **使用贝叶斯优化的超参数调整**

让我们看看贝叶斯优化性能如何与超波段和随机搜索进行比较。

请务必访问本教程的 ***“下载”*** 部分来检索源代码。

从这里开始，让我们尝试一下贝叶斯超参数优化:

```py
$ time python train.py --tuner bayesian --plot output/bayesian_plot.png
[INFO] loading Fashion MNIST...
[INFO] instantiating a bayesian optimization tuner object...
[INFO] performing hyperparameter search...

Search: Running Trial #1

Hyperparameter    |Value             |Best Value So Far
conv_1            |64                |?
conv_2            |64                |?
dense_units       |512               |?
learning_rate     |0.01              |?

Epoch 1/50
1875/1875 [==============================] - 143s 76ms/step - loss: 0.7434 - accuracy: 0.7723 - val_loss: 0.5290 - val_accuracy: 0.8095
...
Epoch 12/50
1875/1875 [==============================] - 50s 27ms/step - loss: 0.2210 - accuracy: 0.9223 - val_loss: 0.4138 - val_accuracy: 0.8693

Trial 1 Complete [00h 11m 45s]
val_accuracy: 0.9136999845504761

Best val_accuracy So Far: 0.9136999845504761
Total elapsed time: 00h 11m 45s
```

在我们的第一次试验中，我们击中了![\approx](img/efc44d3feeb891408f5038a6a5a7c926.png "\approx")

91% accuracy.

在最后的试验中，我们略微提高了精确度:

```py
Search: Running Trial #10

Hyperparameter    |Value             |Best Value So Far   
conv_1            |64                |32
conv_2            |96                |96
dense_units       |768               |768
learning_rate     |0.001             |0.001

Epoch 1/50
1875/1875 [==============================] - 64s 34ms/step - loss: 0.5743 - accuracy: 0.8140 - val_loss: 0.3341 - val_accuracy: 0.8791
...
Epoch 16/50
1875/1875 [==============================] - 62s 33ms/step - loss: 0.0757 - accuracy: 0.9721 - val_loss: 0.3104 - val_accuracy: 0.9211

Trial 10 Complete [00h 16m 41s]
val_accuracy: 0.9251999855041504

Best val_accuracy So Far: 0.9283000230789185
Total elapsed time: 01h 47m 01s
```

我们现在获得![\approx](img/efc44d3feeb891408f5038a6a5a7c926.png "\approx")

92% accuracy.

通过贝叶斯优化找到的最佳超参数如下:

```py
[INFO] optimal number of filters in conv_1 layer: 32
[INFO] optimal number of filters in conv_2 layer: 96
[INFO] optimal number of units in dense layer: 768
[INFO] optimal learning rate: 0.0010
```

以下列表对超参数进行了细分:

*   我们的第一个 CONV 层有`32`个节点(相对于超波段的`64`和随机的`96`)
*   第二 CONV 层有`96`个节点(超波段选择`128`和随机搜索`64`
*   完全连接的层有`768`个节点(超波段和随机搜索都选择了`512`
*   我们的学习率是`1e-3`(这三个超参数优化器都同意)

现在让我们在这些超参数上训练我们的网络:

```py
[INFO] training the best model...
Epoch 1/50
1875/1875 [==============================] - 49s 26ms/step - loss: 0.5764 - accuracy: 0.8164 - val_loss: 0.3823 - val_accuracy: 0.8779
...
Epoch 14/50
1875/1875 [==============================] - 47s 25ms/step - loss: 0.0915 - accuracy: 0.9665 - val_loss: 0.2669 - val_accuracy: 0.9214
[INFO] evaluating network...
              precision    recall  f1-score   support

         top       0.82      0.93      0.87      1000
     trouser       1.00      0.99      0.99      1000
    pullover       0.86      0.92      0.89      1000
       dress       0.93      0.91      0.92      1000
        coat       0.90      0.86      0.88      1000
      sandal       0.99      0.99      0.99      1000
       shirt       0.81      0.72      0.77      1000
     sneaker       0.96      0.98      0.97      1000
         bag       0.99      0.98      0.99      1000
  ankle boot       0.98      0.96      0.97      1000

    accuracy                           0.92     10000
   macro avg       0.93      0.92      0.92     10000
weighted avg       0.93      0.92      0.92     10000

real    118m11.916s
user    740m56.388s
sys     18m2.676s
```

这里的精确度有所提高。我们现在处于 **![\pmb\approx](img/e36579093a0fcb827143090e30c2b0c2.png "\pmb\approx")93%的准确率**使用贝叶斯优化(超波段和随机搜索都有报道![\approx](img/efc44d3feeb891408f5038a6a5a7c926.png "\approx")

92% accuracy).

### 我们如何解释这些结果？

现在让我们花点时间来讨论这些结果。既然贝叶斯优化返回了最高的精度，这是否意味着您应该*总是*使用贝叶斯超参数优化？

不，不一定。

相反，我建议对每个超参数优化器进行一些试验，这样您就可以了解跨几种算法的超参数的“一致程度”。如果所有三个超参数调谐器都报告了相似的超参数，那么你可以有理由相信你找到了最佳的。

说到这里，下表列出了每个优化器的超参数结果:

**虽然在 CONV 滤波器的数量和 FC 节点的数量上有一些分歧，** ***三者都同意 1e-3 是最佳学习速率。***

这告诉我们什么？

假设其他超参数有变化，但所有三个优化器的学习率是相同的，**我们可以得出结论，学习率对准确性有最大的影响**。其他参数没有简单地获得正确的学习率重要。

## **总结**

在本教程中，您学习了如何使用 Keras Tuner 和 TensorFlow 轻松调整神经网络超参数。

Keras Tuner 软件包通过以下方式使您的模型超参数的调整变得非常简单:

*   只需要一次进口
*   允许您在您的模型架构中定义值和范围
*   *直接与 Keras 和 TensorFlow 接口*
*   *实施最先进的超参数优化器*

 *当训练你自己的神经网络时，我建议你至少花一些时间来调整你的超参数，因为你很可能能够在精度上从 1-2%的提升(低端)到 25%的提升(高端)。同样，这取决于你项目的具体情况。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，** ***只需在下面的表格中输入您的电子邮件地址！*****