# 使用 scikit-learn、Keras 和 TensorFlow 进行深度学习的超参数调整

> 原文：<https://pyimagesearch.com/2021/05/31/hyperparameter-tuning-for-deep-learning-with-scikit-learn-keras-and-tensorflow/>

在本教程中，您将学习如何使用 scikit-learn、Keras 和 TensorFlow 来调整深度神经网络的超参数。

本教程是我们关于超参数调优的四部分系列的第三部分:

1.  [*使用 scikit-learn 和 Python*](https://pyimagesearch.com/2021/05/17/introduction-to-hyperparameter-tuning-with-scikit-learn-and-python/) 进行超参数调优的介绍(本系列的第一篇教程)
2.  [*【网格搜索超参数调优用 scikit-learn(GridSearchCV)*](https://pyimagesearch.com/2021/05/24/grid-search-hyperparameter-tuning-with-scikit-learn-gridsearchcv/)(上周教程)
3.  *使用 scikit-learn、Keras 和 TensorFlow 进行深度学习的超参数调优*(今天的帖子)
4.  *使用 Keras 调谐器和 TensorFlow 进行简单的超参数调谐*(下周发布)

当训练一个深度神经网络时，优化你的超参数是非常关键的。一个网络有许多旋钮、转盘和参数——更糟糕的是，网络本身不仅对训练构成挑战*，而且对训练*也很慢*(即使有 GPU 加速)。*

**未能正确优化深度神经网络的超参数可能会导致性能不佳。**幸运的是，有一种方法可以让我们搜索超参数搜索空间，并自动找到最优值*—我们今天将介绍这些方法。*

 ***要了解如何用 scikit-learn、Keras、TensorFlow、** ***将超参数调至深度学习模型，只需继续阅读。***

## **使用 scikit-learn、Keras 和 TensorFlow 进行深度学习的超参数调整**

在本教程的第一部分，我们将讨论深度学习和超参数调整的重要性。我还将向您展示 scikit-learn 的超参数调节功能如何与 Keras 和 TensorFlow 接口。

然后，我们将配置我们的开发环境，并检查我们的项目目录结构。

从这里，我们将实现两个 Python 脚本:

1.  一种是通过用 *no* 超参数调谐来训练基本多层感知器(MLP)来建立基线
2.  另一个搜索超参数空间，导致更精确的模型

我们将讨论我们的结果来结束本教程。

### **如何用 scikit-learn 调优深度学习超参数模型？**

这篇关于神经网络超参数调整的教程是从 PyImageSearch 读者 Abigail 给我的一个问题中得到启发的:

> *嗨，阿德里安，*
> 
> 感谢所有关于神经网络的教程。我有一些关于选择/构建架构的问题:
> 
> *   *您如何“知道”给定层中要使用的节点数量？*
> *   如何选择学习率？
> *   *最佳批量是多少？*
> *   *你怎么知道网络要训练多少个纪元？*
> 
> *If you could shed some light on that, I would really appreciate it.”*

通常，有三种方法可以设置这些值:

1.  **什么都不做(只是猜测):**这是很多初学机器学习的从业者都会做的事情。他们阅读书籍、教程或指南，了解其他架构使用的内容，然后简单地复制并粘贴到自己的代码中。有时这行得通，有时行不通— *但是在几乎所有的情况下，不调整超参数会留下一些误差。*
2.  **依靠你的经验:**训练一个深度神经网络，一部分是艺术，一部分是科学。一旦你训练了 100 或 1000 个神经网络，你就开始发展一种第六感，知道什么可行，什么不可行。问题是，达到这一水平需要很长时间*(当然，会有你的直觉不正确的情况)。*
3.  ***用算法调整你的超参数:**这是你找到最佳超参数的简单方法。是的，由于需要运行 100 次甚至 1000 次的试验，这有点耗时，但您肯定会得到一些改进。*

 *今天，我们将学习如何调整神经网络的以下超参数:

*   层中的节点数
*   学习率
*   辍学率
*   批量
*   为之训练的时代

我们将通过以下方式完成这项任务:

1.  实现基本的神经网络架构
2.  定义要搜索的超参数空间
3.  从`tensorflow.keras.wrappers.scikit_learn`子模块实例化`KerasClassifier`的实例
4.  通过 scikit-learn 的`RandomizedSearchCV`类在超参数和模型架构上运行随机搜索

到本指南结束时，我们将把精确度从 **78.59%** (无超参数调整)提高到 **98.28%** (有超参数调整的*)。*

### **配置您的开发环境**

这个超参数调优教程需要 Keras 和 TensorFlow。如果你打算遵循这个教程，我建议你花时间配置你的深度学习开发环境。

您可以利用这两个指南中的任何一个在您的系统上安装 TensorFlow 和 Keras:

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

在我们实现任何代码之前，让我们首先确保理解我们的项目目录结构。

请务必访问本指南的 ***“下载”*** 部分以检索源代码。

然后，您将看到以下目录结构:

```py
$ tree . --dirsfirst
.
├── pyimagesearch
│   └── mlp.py
├── random_search_mlp.py
└── train.py

1 directory, 3 files
```

在`pyimagesearch`模块中，我们有一个单独的文件`mlp.py`。这个脚本包含`get_mlp_model`，它接受几个参数，然后构建一个多层感知器(MLP)架构。**它接受的参数将由我们的超参数调整算法设置，从而允许我们以编程方式调整网络的内部参数。**

为了建立一个没有*超参数调整*的基线，我们将使用`train.py`脚本创建我们的 MLP 的一个实例，然后在 MNIST 数字数据集上训练它。

一旦我们的基线建立，我们将通过`random_search_mlp.py`执行随机超参数搜索。

**如本教程的结果部分所示，超参数搜索会导致*大规模*的准确性********—********![\pmb\approx](img/e36579093a0fcb827143090e30c2b0c2.png "\pmb\approx")增加 20%！****

 *### **实现我们的基本前馈神经网络**

为了调整神经网络的超参数，我们首先需要定义模型架构。在模型架构中，我们将包含给定层中节点数量和辍学率的变量。

我们还将包括优化器本身的学习率。

该模型一旦构建，将被返回到超参数调谐器。然后，调谐器将根据我们的训练数据拟合神经网络，对其进行评估，并返回分数。

所有试验完成后，超参数调谐器将告诉我们哪些超参数提供了最佳精度。

但是这一切都是从实现模型架构本身开始的。在`pyimagesearch`模块中打开`mlp.py`文件，让我们开始工作:

```py
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
```

**第 2-6 行**导入我们需要的 Python 包。这些都是构建基本前馈神经网络的相当标准的输入。如果你需要学习用 Keras/TensorFlow 构建神经网络的基础知识，我推荐阅读[这个教程。](https://pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/)

现在让我们定义我们的`get_mlp_model`函数，它负责接受我们想要测试的超参数，构建神经网络，并返回它:

```py
def get_mlp_model(hiddenLayerOne=784, hiddenLayerTwo=256,
	dropout=0.2, learnRate=0.01):
	# initialize a sequential model and add layer to flatten the
	# input data
	model = Sequential()
	model.add(Flatten())
```

我们的`get_mlp_model`接受四个参数，包括:

*   `hiddenLayerOne`:第*个*全连通层的节点数
*   `hiddenLayerTwo`:第*第二*全连通层的节点数
*   `dropout`:全连接层之间的脱落率(有助于减少过度拟合)
*   `learnRate`:Adam 优化器的学习率

**12-13 线**开始建造`model`建筑。

让我们继续在下面的代码块中构建架构:

```py
	# add two stacks of FC => RELU => DROPOUT
	model.add(Dense(hiddenLayerOne, activation="relu",
		input_shape=(784,)))
	model.add(Dropout(dropout))
	model.add(Dense(hiddenLayerTwo, activation="relu"))
	model.add(Dropout(dropout))

	# add a softmax layer on top
	model.add(Dense(10, activation="softmax"))

	# compile the model
	model.compile(
		optimizer=Adam(learning_rate=learnRate),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"])

	# return compiled model
	return model
```

**第 16-20 行**定义了两堆`FC => RELU => DROPOUT`层。

**注意，我们在构建架构时使用了`hiddenLayerOne`、`hiddenLayerTwo`和`dropout`层。**将这些值中的每一个编码为变量允许我们在执行超参数搜索时向`get_mlp_model`提供*不同的值*。**这样做是 scikit-learn 如何将超参数调整到 Keras/TensorFlow 模型的“魔法”。**

第 23 行在我们最终的 FC 层上添加了一个 softmax 分类器。

然后，我们使用 Adam 优化器和指定的`learnRate`(将通过我们的超参数搜索进行调整)来编译模型。

得到的模型返回到第 32 行**上的调用函数。**

### **创建我们的基本训练脚本(** ***否*** **超参数调优)**

在我们执行超参数搜索之前，让我们首先获得一个没有*超参数调整的基线。这样做将会给我们一个基线准确度分数，让我们去超越。*

打开项目目录结构中的`train.py`文件，让我们开始工作:

```py
# import tensorflow and fix the random seed for better reproducibility
import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
from pyimagesearch.mlp import get_mlp_model
from tensorflow.keras.datasets import mnist
```

**第 2 行和第 3 行**导入 TensorFlow 库并修复随机种子。固定随机种子有助于(但不一定保证)更好的再现性。

请记住，神经网络是**随机算法**，这意味着有一点随机性，特别是在:

*   层初始化(从随机分布初始化神经网络中的节点)
*   训练和测试集拆分
*   数据批处理过程中注入的任何随机性

使用固定种子有助于通过确保至少层初始化随机性是一致的(理想情况下)来提高可再现性。

从那里，我们加载 MNIST 数据集:

```py
# load the MNIST dataset
print("[INFO] downloading MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0
```

如果这是您第一次使用 Keras/TensorFlow，那么 MNIST 数据集将被下载并缓存到您的磁盘中。

然后，我们将训练和测试图像中的像素强度从范围*【0，255】*缩放到*【0，1】*，这是使用神经网络时常见的预处理技术。

现在让我们训练我们的基本前馈网络:

```py
# initialize our model with the default hyperparameter values
print("[INFO] initializing model...")
model = get_mlp_model()

# train the network (i.e., no hyperparameter tuning)
print("[INFO] training model...")
H = model.fit(x=trainData, y=trainLabels,
	validation_data=(testData, testLabels),
	batch_size=8,
	epochs=20)

# make predictions on the test set and evaluate it
print("[INFO] evaluating network...")
accuracy = model.evaluate(testData, testLabels)[1]
print("accuracy: {:.2f}%".format(accuracy * 100))
```

**第 19 行**调用`get_mlp_model`函数，用*默认选项*构建我们的神经网络(我们稍后将通过超参数搜索调整学习率、辍学率和隐藏层节点数)。

**第 23-26 行**训练我们的神经网络。

然后，我们通过第 30 和 31 行**在我们的测试集上评估模型的准确性。**

**在执行超参数搜索时，这一准确度将作为我们需要超越的基线。**

### **获得基线精度**

在我们为我们的网络调整超参数之前，让我们先用我们的“默认”配置获得一个基线(即，根据我们的经验，我们认为会产生良好精度的超参数)。

通过访问本教程的 ***【下载】*** 部分来检索源代码。

从那里，打开一个 shell 并执行以下命令:

```py
$ time python train.py
Epoch 1/20
7500/7500 [==============================] - 18s 2ms/step - loss: 0.8881 - accuracy: 0.7778 - val_loss: 0.4856 - val_accuracy: 0.9023
Epoch 2/20
7500/7500 [==============================] - 17s 2ms/step - loss: 0.6887 - accuracy: 0.8426 - val_loss: 0.4591 - val_accuracy: 0.8658
Epoch 3/20
7500/7500 [==============================] - 17s 2ms/step - loss: 0.6455 - accuracy: 0.8466 - val_loss: 0.4536 - val_accuracy: 0.8960
...
Epoch 18/20
7500/7500 [==============================] - 19s 2ms/step - loss: 0.8592 - accuracy: 0.7931 - val_loss: 0.6860 - val_accuracy: 0.8776
Epoch 19/20
7500/7500 [==============================] - 17s 2ms/step - loss: 0.9226 - accuracy: 0.7876 - val_loss: 0.9510 - val_accuracy: 0.8452
Epoch 20/20
7500/7500 [==============================] - 17s 2ms/step - loss: 0.9810 - accuracy: 0.7825 - val_loss: 0.8294 - val_accuracy: 0.7859
[INFO] evaluating network...
313/313 [==============================] - 1s 2ms/step - loss: 0.8294 - accuracy: 0.7859
accuracy: 78.59%

real	5m48.320s
user	19m53.908s
sys	2m25.608s
```

使用我们实现中的默认超参数，*没有超参数调整，*我们可以达到 **78.59%的准确度。**

现在我们有了基线，我们可以战胜它了——正如您将看到的，应用超参数调整会彻底击败这个结果！

### **实施我们的 Keras/TensorFlow 超参数调整脚本**

让我们学习如何使用 scikit-learn 将超参数调整到 Keras/TensorFlow 模型。

我们从进口开始:

```py
# import tensorflow and fix the random seed for better reproducibility
import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
from pyimagesearch.mlp import get_mlp_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.datasets import mnist
```

同样，**第 2-3 行**导入 TensorFlow 并修复我们的随机种子以获得更好的可重复性。

**第 6-9 行**导入我们需要的 Python 包，包括:

*   `get_mlp_model`:接受几个超参数，并基于它们构建一个神经网络
*   `KerasClassifier`:获取一个 Keras/TensorFlow 模型，并以一种与 scikit-learn 函数兼容的方式包装它(例如 scikit-learn 的超参数调整函数)
*   `RandomizedSearchCV` : scikit-learn 的随机超参数搜索的实现(如果您不熟悉随机超参数调整算法，请参见[本教程](https://pyimagesearch.com/2021/05/17/introduction-to-hyperparameter-tuning-with-scikit-learn-and-python/)
*   `mnist`:MNIST 数据集

然后，我们可以继续从磁盘加载 MNIST 数据集并对其进行预处理:

```py
# load the MNIST dataset
print("[INFO] downloading MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0
```

现在是时候构建我们的`KerasClassifier`对象了，这样我们可以用`get_mlp_model`构建一个模型，然后用`RandomizedSearchCV`调整超参数:

```py
# wrap our model into a scikit-learn compatible classifier
print("[INFO] initializing model...")
model = KerasClassifier(build_fn=get_mlp_model, verbose=0)

# define a grid of the hyperparameter search space
hiddenLayerOne = [256, 512, 784]
hiddenLayerTwo = [128, 256, 512]
learnRate = [1e-2, 1e-3, 1e-4]
dropout = [0.3, 0.4, 0.5]
batchSize = [4, 8, 16, 32]
epochs = [10, 20, 30, 40]

# create a dictionary from the hyperparameter grid
grid = dict(
	hiddenLayerOne=hiddenLayerOne,
	learnRate=learnRate,
	hiddenLayerTwo=hiddenLayerTwo,
	dropout=dropout,
	batch_size=batchSize,
	epochs=epochs
)
```

**第 21 行**实例化我们的`KerasClassifier`对象。我们传入我们的`get_mlp_model`函数，告诉 Keras/tensor flow,`get_mlp_model`函数负责构建模型架构。

接下来，**行 24-39** 定义了我们的超参数搜索空间。我们将调谐:

*   第一个全连接层中的节点数
*   第二个全连接层中的节点数
*   我们的学习速度
*   辍学率
*   批量
*   要训练的时代数

然后将超参数添加到名为`grid`的 Python 字典中。

注意，字典的关键字是`get_mlp_model`中变量的*同名*。此外，`batch_size`和`epochs`变量与您在使用 Keras/TensorFlow 调用`model.fit`时提供的变量相同。

**该命名约定是设计的*****，并且是当您构建 Keras/TensorFlow 模型并试图用 scikit-learn 调整超参数时所必需的*****。****

 **定义好`grid`超参数后，我们可以开始超参数调整过程:

```py
# initialize a random search with a 3-fold cross-validation and then
# start the hyperparameter search process
print("[INFO] performing random search...")
searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3,
	param_distributions=grid, scoring="accuracy")
searchResults = searcher.fit(trainData, trainLabels)

# summarize grid search information
bestScore = searchResults.best_score_
bestParams = searchResults.best_params_
print("[INFO] best score is {:.2f} using {}".format(bestScore,
	bestParams))
```

**第 44 和 45 行**初始化我们的`searcher`。我们传入`model`，运行`-1`值的并行作业的数量告诉 scikit——学习使用您机器上的所有内核/处理器、交叉验证折叠的数量、超参数网格以及我们想要监控的指标。

从那里，调用`searcher`的`fit`开始超参数调整过程。

一旦搜索完成，我们获得在搜索过程中找到的`bestScore`和`bestParams`，在我们的终端上显示它们(**第 49-52 行**)。

最后一步是获取`bestModel`并对其进行评估:

```py
# extract the best model, make predictions on our data, and show a
# classification report
print("[INFO] evaluating the best model...")
bestModel = searchResults.best_estimator_
accuracy = bestModel.score(testData, testLabels)
print("accuracy: {:.2f}%".format(accuracy * 100))
```

**第 57 行**从随机搜索中抓取`best_estimator_`。

然后，我们根据测试数据评估最佳模型，并在屏幕上显示精确度(**第 58 行和第 59 行**)。

### **使用 scikit-learn 结果调整 Keras/TensorFlow 超参数**

让我们看看我们的 Keras/TensorFlow 超参数调优脚本是如何执行的。

访问本教程的 ***“下载”*** 部分来检索源代码。

从那里，您可以执行以下命令:

```py
$ time python random_search_mlp.py
[INFO] downloading MNIST...
[INFO] initializing model...
[INFO] performing random search...
[INFO] best score is 0.98 using {'learnRate': 0.001, 'hiddenLayerTwo': 256, 'hiddenLayerOne': 256, 'epochs': 40, 'dropout': 0.4, 'batch_size': 32}
[INFO] evaluating the best model...
accuracy: 98.28%

real    22m52.748s
user    151m26.056s
sys     12m21.016s
```

`random_search_mlp.py`脚本占用了![\approx](img/efc44d3feeb891408f5038a6a5a7c926.png "\approx")

4x longer to run than our basic no hyperparameter tuning script (![\approx](img/efc44d3feeb891408f5038a6a5a7c926.png "\approx")23m versus ![\approx](img/efc44d3feeb891408f5038a6a5a7c926.png "\approx")6m, respectively) — **but that extra time is *well worth it* as the difference in accuracy is *tremendous.***

在没有任何超参数调整的情况下，我们仅获得了 78.59%的准确度。但是通过使用 scikit-learn 进行随机超参数搜索，**我们能够将准确率提高到 98.28%！**

这是一个巨大的准确性提升，如果没有专门的超参数搜索，这是不可能的。

## **总结**

在本教程中，您学习了如何使用 scikit-learn、Keras 和 TensorFlow 将超参数调整到深度神经网络。

通过使用 Keras/TensorFlow 的`KerasClassifier`实现，我们能够包装我们的模型架构，使其与 scikit-learn 的`RandomizedSearchCV`类兼容。

从那里，我们:

1.  从我们的超参数空间随机抽样
2.  在当前的超参数集上训练我们的神经网络(交叉验证)
3.  评估模型的性能

在试验结束时，我们从随机搜索中获得最佳超参数，训练最终模型，并评估准确性:

*   *没有*超参数调整，我们只能获得 **78.59%的准确度**
*   但是*通过*超参数调整，我们达到了 **98.28%的准确度**

正如你所看到的，调整神经网络的超参数可以在准确性上产生巨大的差异…这只是在简单的 MNIST 数据集上。想象一下它能为您更复杂的真实数据集做些什么！

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，** ***只需在下面的表格中输入您的电子邮件地址！*********