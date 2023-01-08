# 使用 scikit-learn 和 Python 调整超参数简介

> 原文：<https://pyimagesearch.com/2021/05/17/introduction-to-hyperparameter-tuning-with-scikit-learn-and-python/>

在本教程中，您将学习如何使用 scikit-learn 和 Python 调优机器学习模型超参数。

本教程是关于超参数调整的四部分系列的第一部分:

1.  *介绍使用 scikit-learn 和 Python 进行超参数调优*(今天的帖子)
2.  *用 scikit-learn ( GridSearchCV)进行网格搜索超参数调优*(下周的帖子)
3.  *使用 scikit-learn、Keras 和 TensorFlow 进行深度学习的超参数调整*(两周后的教程)
4.  *使用 Keras 调谐器和 TensorFlow 实现简单的超参数调谐*(本系列的最后一篇文章)

**调整您的超参数是** ***获得高精度模型的绝对关键*** **。许多机器学习模型都有各种旋钮、转盘和参数可供您设置。**

一个非常低精度的模型和一个高精度的模型之间的区别有时就像调整正确的刻度盘一样简单。本教程将向你展示如何调整你的机器学习模型上的转盘，以提高你的准确性。

具体来说，我们将通过以下方式介绍超参数调整的基础:

1.  获得**基线**和*无超参数调优*，我们有一个要改进的基准
2.  在一组超参数上进行彻底的**网格搜索**
3.  利用**随机搜索**从超参数空间采样

我们将使用 Python 和 scikit-learn 实现每个方法，训练我们的模型，并评估结果。

在本教程结束时，您将对如何在您自己的项目中实际使用超参数调整来提高模型精度有很深的理解。

**学习如何用 scikit-learn 和 Python 调优超参数，** ***继续阅读。***

## **介绍使用 scikit-learn 和 Python 进行超参数调整**

在本教程中，您将学习如何使用 scikit-learn 和 Python 调优模型超参数。

我们将从讨论什么是超参数调整以及为什么它如此重要开始本教程。

从那里，我们将配置您的开发环境并检查项目目录结构。

然后我们将实现三个 Python 脚本:

1.  一个是用 *no* 超参数调整来训练模型(这样我们可以获得一个基线)
2.  一种是利用一种叫做“网格搜索”的算法来彻底检查*超参数的所有组合*——这种方法可以保证对超参数值进行全面扫描，但是*非常慢*
3.  最后一种方法使用“随机搜索”，从分布中对各种超参数进行采样(不保证覆盖所有超参数值，但实际上通常与网格搜索一样准确，并且运行速度*比*快得多)

最后，我们将以对结果的讨论来结束我们的教程。

### **什么是超参数调整，为什么它很重要？**

你以前试过用老式的 AM/FM 收音机吗？你知道，模拟收音机有旋钮和转盘，你可以移动它们来选择电台，调节均衡器等等。，像是在**图一**？

这些收音机可能是过去的遗物，但如果你以前用过，你就会知道让电台选择器“恰到好处”是多么重要如果你让电台选择器位于两个频率之间，你会得到两个不同电台的音频相互渗透。噪音刺耳，几乎无法理解。

类似地，机器学习模型有*各种各样*你可以调整的旋钮和转盘:

*   神经网络具有学习速率和正则化强度
*   卷积神经网络有几层，每个卷积层的滤波器数量，全连接层中的节点数量等。
*   决策树具有节点分裂标准(基尼指数、信息增益等。)
*   随机森林具有森林中的树木总数，以及特征空间采样百分比
*   支持向量机(SVMs)具有核类型(线性、多项式、径向基函数(RBF)等)。)以及针对特定内核需要调整的任何参数

**支持向量机是** ***臭名昭著的*** **因为需要显著的超参数调整，** ***尤其是*** **如果你使用的是非线性内核。**您不仅需要为您的数据选择正确的内核类型，而且还需要调整与内核相关的任何旋钮和刻度盘— *一个错误的选择，您的准确性就会直线下降。*

### **Scikit-learn:使用网格搜索和随机搜索进行超参数调整**

scikit-learn 最常用的两种超参数方法是**网格搜索**和**随机搜索。**这两种算法背后的总体思路是:

1.  定义一组要优化的超参数
2.  将这些超参数用于网格搜索或随机搜索
3.  然后，这些算法*自动*检查超参数搜索空间，并试图找到最大化精度的最佳值

也就是说，网格搜索和随机搜索本质上是超参数调整的不同技术。**图 2** 显示了这两种超参数搜索算法:

在**图 2** 中，我们有一个 2D 网格，第一个超参数的值沿 *x* 轴绘制，第二个超参数的值沿 *y-* 轴绘制。**白色突出显示的椭圆形是*和*这两个超参数的最佳值。我们的目标是使用我们的超参数调整算法来定位这个区域。**

**图 2** ( *左)*可视化网格搜索:

1.  我们首先定义一组所有的超参数和我们想要研究的相关值
2.  网格搜索然后检查这些超参数的 ***所有组合***
3.  对于超参数的每个可能组合，我们在它们上面训练一个模型
4.  然后返回与最高精度相关的超参数

网格搜索保证检查所有可能的超参数组合。**问题是超参数越多，组合的数量就越多** ***呈指数增长！***

由于要检查的组合太多，网格搜索往往会运行得非常慢。

为了帮助加速这个过程，我们可以使用一个**随机搜索** ( **图 2** ，*右*)。

通过随机搜索，我们可以:

1.  定义我们要搜索的超参数
2.  为每个超参数设置值的下限和上限(如果是连续变量)或超参数可以采用的可能值(如果是分类变量)
3.  随机搜索然后*从这些分布中随机采样*总共 *N* 次，在每组超参数上训练一个模型
4.  然后返回与最高精度模型相关的超参数

虽然根据定义，随机搜索不是像网格搜索那样的穷举搜索，但随机搜索的好处是它的 ***比*** 快得多，并且通常获得与网格搜索一样高的准确性。

这是为什么呢？

网格搜索既然如此详尽，难道不应该获得最高的精确度吗？

不一定。

**在调整超参数时，并不是只有** ***一个*** **【黄金数值组】能给你最高的精度。取而代之的是一个** ***分布*****——每个超参数都有** ***范围*** **来获得最佳精度**。如果您落在该范围/分布内，您仍将享受相同的高精度，而不需要使用网格搜索彻底调整您的超参数。

### **配置您的开发环境**

为了遵循这个指南，您需要在您的系统上安装 scikit-learn 机器库和 Pandas 数据科学库。

幸运的是，这两个包都是 pip 可安装的:

```py
$ pip install scikit-learn
$ pip install pandas
```

如果您在配置开发环境时遇到问题，您应该参考下一节。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **我们的示例数据集**

我们今天要使用的数据集是来自 UCI 机器学习库的[鲍鱼。](https://archive.ics.uci.edu/ml/datasets/abalone)

鲍鱼是海洋生物学家经常研究的海洋蜗牛。具体来说，海洋生物学家对蜗牛的年龄感兴趣。

*问题*是确定鲍鱼的年龄是一项耗时、乏味的任务，需要生物学家:

1.  将鲍鱼壳切开
2.  弄脏它
3.  用显微镜数一数染色贝壳上的年轮数量(就像测量树上的年轮一样)

是的，就像听起来一样无聊。

然而，使用*更容易收集的值*，有可能*预测*海洋蜗牛的年龄。

鲍鱼数据集包括七个值:

1.  外壳的长度
2.  壳的直径
3.  外壳的高度
4.  重量(鲍鱼的总重量)
5.  去壳重量(鲍鱼肉的重量)
6.  内脏重量
7.  外壳重量

给定这七个值，我们的目标是训练一个机器学习模型来预测鲍鱼的年龄。**正如我们将看到的，根据我们的机器学习模型调整超参数可以提高准确性。**

### **项目结构**

让我们从回顾我们的项目目录结构开始。

请务必访问本教程的 ***“下载”*** 部分来检索源代码。

然后，您将看到以下文件:

```py
$ tree . --dirsfirst
.
├── pyimagesearch
│   └── config.py
├── abalone_train.csv
├── train_svr.py
├── train_svr_grid.py
└── train_svr_random.py

1 directory, 5 files
```

`abalone_train.csv`文件包含来自 UCI 机器学习库的[鲍鱼数据集(不需要从 UCI 下载 CSV 文件，因为我已经在与教程相关的下载中包含了 CSV 文件)。](https://archive.ics.uci.edu/ml/datasets/abalone)

鲍鱼数据集包含 3，320 行，每行 8 列(7 列用于特征，包括壳长、直径等。)和最后一列年龄(这是我们试图预测的目标值)。

为了评估超参数调整的影响，我们将实现三个 Python 脚本:

1.  `train_svr.py`:通过训练支持向量回归(SVR)在鲍鱼数据集上建立基线，其中*没有超参数调整。*
2.  `train_svr_grid.py`:利用网格搜索进行超参数调谐。
3.  `train_svr_random.py`:对超参数值的分布进行随机超参数搜索。

`pyimagesearch`模块中的`config.py`文件实现了我们在三个驱动程序脚本中使用的重要配置变量，包括输入 CSV 数据集的路径和列名。

### **创建我们的配置文件**

在我们能够实现任何训练脚本之前，让我们首先回顾一下我们的配置文件。

打开`pyimagesearch`模块中的`config.py`文件，您会发现如下代码:

```py
# specify the path of our dataset
CSV_PATH = "abalone_train.csv"

# specify the column names of our dataframe
COLS = ["Length", "Diameter", "Height", "Whole weight",
	"Shucked weight", "Viscera weight", "Shell weight", "Age"]
```

在第 2 行**，**上，我们初始化`CSV_PATH`以指向驻留在磁盘上的鲍鱼 CSV 文件。

**第 5 行和第 6 行**定义 CSV 文件中八列的名称。我们需要训练脚本中的列名来从磁盘加载并解析它。

这就是我们的配置文件！让我们继续实施我们的培训脚本。

### **实施基本培训脚本**

我们的第一个训练脚本`train_svr.py`，将通过执行*无超参数调整来建立**基线精度**。*一旦我们有了这个基线，我们将尝试通过应用超参数调整算法来改进它。

但是现在，让我们建立我们的基线。

打开项目目录中的`train_svr.py`文件，让我们开始工作:

```py
# import the necessary packages
from pyimagesearch import config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import pandas as pd
```

**第 2-6 行**导入我们需要的 Python 包，包括:

*   `config`:我们在上一节中实现的配置文件
*   `StandardScaler`:一种数据预处理技术，执行标准缩放(也称为*z*-统计世界中的分数)，通过减去特定列值的平均值除以标准偏差来缩放每个数据观察值
*   `train_test_split`:构建训练和测试分割
*   `SVR` : scikit-learn 实现了一个用于回归的支持向量机
*   `pandas`:用于从磁盘加载我们的 CSV 文件并解析数据

说到这里，现在让我们从磁盘加载我们的鲍鱼 CSV 文件:

```py
# load the dataset, separate the features and labels, and perform a
# training and testing split using 85% of the data for training and
# 15% for evaluation
print("[INFO] loading data...")
dataset = pd.read_csv(config.CSV_PATH, names=config.COLS)
dataX = dataset[dataset.columns[:-1]]
dataY = dataset[dataset.columns[-1]]
(trainX, testX, trainY, testY) = train_test_split(dataX,
	dataY, random_state=3, test_size=0.15)
```

**第 12 行**从磁盘读取我们的 CSV 文件。注意我们如何将列的`CSV_PATH`传递给磁盘上的文件。

从那里，我们需要提取我们的特征向量(即，壳的长度，直径，高度等。)和我们希望预测的目标值(年龄)— **第 13 行和第 14 行**通过简单的数组切片为我们实现了这一点。

一旦我们的数据被加载和解析，我们在第 15 行**和第 16 行**构建一个训练和测试分割，使用 85%的数据用于训练，15%用于测试。

加载数据后，我们现在需要对其进行预处理:

```py
# standardize the feature values by computing the mean, subtracting
# the mean from the data points, and then dividing by the standard
# deviation
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)
```

第 21 行实例化了我们的`StandardScaler`的一个实例。然后，我们计算训练数据中所有列值的平均值和标准偏差，然后对它们进行缩放(**第 22 行**)。

**第 23 行**使用在训练集上计算的平均值和标准偏差来缩放我们的测试数据。

现在剩下的就是训练 SVR:

```py
# train the model with *no* hyperparameter tuning
print("[INFO] training our support vector regression model")
model = SVR()
model.fit(trainX, trainY)

# evaluate our model using R^2-score (1.0 is the best value)
print("[INFO] evaluating...")
print("R2: {:.2f}".format(model.score(testX, testY)))
```

**第 27 行**实例化了我们用于回归的支持向量机。请记住，我们在这里使用回归是因为我们试图预测一个实值输出，鲍鱼的*****年龄*** **。****

 ****第 28 行**然后使用我们的`trainX`(特征向量)和`trainY`(要预测的目标年龄)训练模型。

一旦我们的模型被训练，我们就使用决定系数在第 32 行**对其进行评估。**

决定系数通常在分析回归模型的输出时使用。本质上，它衡量的是因变量和自变量之间“可解释的差异”的数量。

该系数将具有在范围 *[0，1]，*内的值，其中 *0* 意味着我们*不能*正确预测目标输出，而值 *1* 意味着我们可以准确无误地预测输出。

有关决定系数的更多信息，请务必参考本页中的[。](https://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination)

### **采集我们的基线结果(** ***否*** **超参数调优)**

在我们将超参数调整到我们的 SVM 之前，我们首先需要获得一个没有*超参数调整的基线。这样做将为我们提供一个可以改进的基线/基准。*

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例数据集。

然后，您可以执行`train_svr.py`脚本:

```py
$ time python train_svr.py
[INFO] loading data...
[INFO] training our support vector regression model
[INFO] evaluating...
R2: 0.55

real	0m1.386s
user	0m1.928s
sys	0m1.040s
```

这里我们得到一个决定值系数`0.55`，这意味着基于 *X* (我们的七个特征值)，在 *Y* (海洋蜗牛的年龄)中 55%的方差是可预测的。

**我们的目标是使用超参数调整来超越这个值。**

### **用网格搜索调整超参数**

现在我们已经建立了一个基线分数，让我们看看是否可以使用 scikit-learn 的超参数调优来击败它。

我们将从实施网格搜索开始。在本指南的后面，您还将学习如何使用随机搜索进行超参数调整。

打开项目目录结构中的`train_svr_grid.py`文件，我们将使用 scikit-learn 实现网格搜索:

```py
# import the necessary packages
from pyimagesearch import config
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
```

**第 2-8 行**导入我们需要的 Python 包。这些导入与我们之前的`train_svr.py`脚本相同，但是增加了两项:

1.  `GridSearchCV` : scikit-learn 的网格搜索超参数调整算法的实现
2.  `RepeatedKFold`:在每次迭代中使用不同的随机化，总共执行*k*-折叠交叉验证 *N* 次

执行 *k-* 折叠交叉验证允许我们*“提高机器学习模型的估计性能”* ( [来源](https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/))并且通常在执行超参数调整时使用。

在优化超参数时，您需要做的最后一件事是对一组随机数据进行长时间的实验，获得高精度，然后发现高精度是由于数据本身的随机异常造成的。利用交叉验证有助于防止这种情况发生。

我们现在可以从磁盘加载 CSV 文件并对其进行预处理:

```py
# load the dataset, separate the features and labels, and perform a
# training and testing split using 85% of the data for training and
# 15% for evaluation
print("[INFO] loading data...")
dataset = pd.read_csv(config.CSV_PATH, names=config.COLS)
dataX = dataset[dataset.columns[:-1]]
dataY = dataset[dataset.columns[-1]]
(trainX, testX, trainY, testY) = train_test_split(dataX,
	dataY, random_state=3, test_size=0.15)

# standardize the feature values by computing the mean, subtracting
# the mean from the data points, and then dividing by the standard
# deviation
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)
```

上面的*“实现一个基本的训练脚本*”一节已经介绍了这个代码，所以如果你需要关于这个代码做什么的更多细节，可以参考那里。

接下来，我们可以初始化我们的回归模型和超参数搜索空间:

```py
# initialize model and define the space of the hyperparameters to
# perform the grid-search over
model = SVR()
kernel = ["linear", "rbf", "sigmoid", "poly"]
tolerance = [1e-3, 1e-4, 1e-5, 1e-6]
C = [1, 1.5, 2, 2.5, 3]
grid = dict(kernel=kernel, tol=tolerance, C=C)
```

第 29 行初始化我们的支持向量机回归(SVR)模型。SVR 有几个要优化的超参数，包括:

1.  `kernel`:将数据投影到高维空间时使用的核的类型，在高维空间中数据理想地变成线性可分的
2.  `tolerance`:停止标准的公差
3.  `C`:SVR 的“严格性”(即，在拟合数据时允许 SVR 在多大程度上出错)

**第 30-32 行**定义了每个超参数的值，而**第 33 行**创建了一个超参数字典。

**我们的目标是搜索这些超参数，并找到`kernel`、`tolerance`和`C`的最佳值。**

说到这里，让我们现在进行网格搜索:

```py
# initialize a cross-validation fold and perform a grid-search to
# tune the hyperparameters
print("[INFO] grid searching over the hyperparameters...")
cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
	cv=cvFold, scoring="neg_mean_squared_error")
searchResults = gridSearch.fit(trainX, trainY)

# extract the best model and evaluate it
print("[INFO] evaluating...")
bestModel = searchResults.best_estimator_
print("R2: {:.2f}".format(bestModel.score(testX, testY)))
```

**第 38 行**创建我们的交叉验证折叠，表明我们将生成`10`个折叠，然后重复整个过程总共`3`次(通常，您会看到范围*【3，10】*内的折叠和重复值)。

第 39 行和第 40 行实例化我们的`GridSearchCV`对象。让我们来分析一下每个论点:

*   `estimator`:我们试图优化的模型(例如，我们的 SVR 将预测鲍鱼的年龄)
*   `param_grid`:超参数搜索空间
*   `n_jobs`:处理器上用于运行并行作业的内核数量。值`-1`意味着*所有的*核心/处理器都将被使用，从而加速网格搜索过程。
*   我们试图优化的损失函数；在这种情况下，我们试图降低我们的均方误差(MSE)，这意味着*越低*的均方误差，*越好*我们的模型在预测鲍鱼的年龄

第 41 行开始网格搜索。

在网格搜索运行之后，我们获得搜索期间找到的`bestModel`(**第 45 行**)，然后计算决定系数，这将告诉我们我们的模型做得有多好(**第 46 行**)。

### **网格搜索超参数调谐结果**

让我们来测试一下网格搜索超参数调优方法。

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例数据集。

从那里，您可以执行以下命令:

```py
$ time python train_svr_grid.py
[INFO] loading data...
[INFO] grid searching over the hyperparameters...
[INFO] evaluating...
R2: 0.56

real	4m34.825s
user	35m47.816s
sys	0m37.268s
```

以前，没有超参数调谐产生一个确定值系数`0.55`。

**使用网格搜索，我们已经将该值提高到了`0.56`，这意味着我们调整后的模型在预测蜗牛年龄方面做得更好。**

然而，获得这种精度是以速度为代价的:

*   在没有超参数调整的情况下，仅用了 1.3 秒来训练我们的原始模型
*   彻底的网格搜索花了 4m34s
*   **增长了 18，166%**

**而且更糟糕的是，你要调优的*****超参数越多，可能的值组合数量爆炸***——对于现实世界的数据集和问题来说，那就是不可行的。**

 **解决方案是利用随机搜索来调整您的超参数。

### **用随机搜索调整超参数**

我们最后的实验将使用随机超参数搜索进行探索。

打开`train_svr_random.py`文件，我们将开始:

```py
# import the necessary packages
from pyimagesearch import config
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from scipy.stats import loguniform
import pandas as pd
```

**第 2-9 行**导入我们需要的 Python 包。这些导入与我们之前的两个脚本几乎相同，最显著的区别是`RandomizedSearchCV`，scikit-learn 的随机超参数搜索的实现。

导入工作完成后，我们可以从磁盘加载鲍鱼 CSV 文件，然后使用`StandardScaler`对其进行预处理:

```py
# load the dataset, separate the features and labels, and perform a
# training and testing split using 85% of the data for training and
# 15% for evaluation
print("[INFO] loading data...")
dataset = pd.read_csv(config.CSV_PATH, names=config.COLS)
dataX = dataset[dataset.columns[:-1]]
dataY = dataset[dataset.columns[-1]]
(trainX, testX, trainY, testY) = train_test_split(dataX,
	dataY, random_state=3, test_size=0.15)

# standardize the feature values by computing the mean, subtracting
# the mean from the data points, and then dividing by the standard
# deviation
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)
```

接下来，我们初始化我们的 SVR 模型，然后定义超参数的搜索空间:

```py
# initialize model and define the space of the hyperparameters to
# perform the randomized-search over
model = SVR()
kernel = ["linear", "rbf", "sigmoid", "poly"]
tolerance = loguniform(1e-6, 1e-3)
C = [1, 1.5, 2, 2.5, 3]
grid = dict(kernel=kernel, tol=tolerance, C=C)
```

在那里，我们可以使用随机搜索来调整超参数:

```py
# initialize a cross-validation fold and perform a randomized-search
# to tune the hyperparameters
print("[INFO] grid searching over the hyperparameters...")
cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
	cv=cvFold, param_distributions=grid,
	scoring="neg_mean_squared_error")
searchResults = randomSearch.fit(trainX, trainY)

# extract the best model and evaluate it
print("[INFO] evaluating...")
bestModel = searchResults.best_estimator_
print("R2: {:.2f}".format(bestModel.score(testX, testY)))
```

**第 40-42 行**实例化我们的`RandomizedSearchCV`对象，类似于我们如何创建我们的`GridSearchCV`调谐器。

从那里，**行 43** 在我们的超参数空间上运行随机搜索。

最后，**第 47 行和第 48 行**获取在超参数空间中找到的最佳模型，并在我们的测试集上对其进行评估。

### **随机超参数搜索结果**

我们现在可以看到我们的随机超参数搜索是如何执行的。

访问本教程的 ***“下载”*** 部分来检索源代码和鲍鱼数据集。

然后，您可以执行`train_svr_random.py`脚本。

```py
$ time python train_svr_random.py
[INFO] loading data...
[INFO] grid searching over the hyperparameters...
[INFO] evaluating...
R2: 0.56

real	0m36.771s
user	4m36.760s
sys	0m6.132s
```

这里，我们获得一个确定值系数`0.56`(与我们的网格搜索相同)；但是随机搜索只用了 **36 秒**而不是网格搜索，网格搜索用了 **4m34s。**

只要有可能，我建议您使用随机搜索而不是网格搜索来进行超参数调整。您通常会在很短的时间内获得类似的准确度。

## **总结**

在本教程中，您学习了使用 scikit-learn 和 Python 进行超参数调优的基础知识。

我们通过以下方式研究了超参数调谐:

1.  使用*无超参数调优*在我们的数据集上获得基线精度—这个值成为我们要打破的分数
2.  利用彻底的网格搜索
3.  应用随机搜索

网格搜索和随机搜索都超过了原始基线，但随机搜索的时间减少了 86%。

我建议在调整超参数时使用随机搜索，因为节省的时间本身就使它在机器学习项目中更加实用和有用。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，** ***只需在下面的表格中输入您的电子邮件地址！********