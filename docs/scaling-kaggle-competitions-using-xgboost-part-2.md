# 使用 XGBoost 扩展 Kaggle 竞赛:第 2 部分

> 原文：<https://pyimagesearch.com/2022/12/12/scaling-kaggle-competitions-using-xgboost-part-2/>

* * *

## **目录**

* * *

## [**用 XGBoost 缩放 Kaggle 比赛:第二部分**](#TOC)

在我们的[之前的教程](https://pyimg.co/0c9pb)中，我们浏览了 XGBoost 背后的基本基础，并了解了将一个基本的 XGBoost 模型合并到我们的项目中是多么容易。我们讨论了理解 XGBoost 所需的核心要素，即决策树和集成学习器。

虽然我们学会了将 XGBoost 插入到我们的项目中，但是我们还没有触及到它背后的神奇之处。在本教程中，我们的目标是将 XGBoost 简化为一个黑体，并学习更多关于使它如此优秀的简单数学知识。

但是这个过程是一步一步来的。为了避免在一个教程中加入太多的数学知识，我们来看看今天这个概念背后的数学知识:AdaBoost。一旦清除了这个问题，我们将在下一个教程中最终解决 XGBoost 之前解决梯度增强问题。

在本教程中，您将学习 XGBoost 的一个先决条件背后的数学基础。

我们还将解决一个稍微更具挑战性的 Kaggle 数据集，并尝试使用 XGBoost 来获得更好的结果。

本课是关于深度学习 108 的 4 部分系列的第 2 部分:

1.  [*用 XGBoost 缩放 Kaggle 比赛:第一部分*](https://pyimg.co/0c9pb)
2.  [***缩放 Kaggle 比赛使用 XGBoost: Part 2***](https://pyimg.co/2wiy7) **(本教程)**
3.  *使用 XGBoost 扩展 Kaggle 竞赛:第 3 部分*
4.  *使用 XGBoost 扩展 Kaggle 竞赛:第 4 部分*

****要学习如何算出 AdaBoost 背后的数学，*只要坚持读下去。*****

* * *

## [**用 XGBoost 缩放 Kaggle 比赛:第二部分**](#TOC)

在本系列的前一篇博文中，我们简要介绍了决策树和梯度增强等概念，然后讨论了 XGBoost 的概念。

随后，我们看到在代码中使用它是多么容易。现在，让我们把手弄脏，更进一步了解背后的数学！

* * *

### [**阿达布斯**T3t](#TOC)

AdaBoost(自适应增强)的正式定义是“将弱学习器的输出组合成加权和，代表最终输出。”但这让很多事情变得模糊不清。让我们开始剖析理解这句话的意思。

由于我们一直在处理树，我们将假设我们的自适应提升技术正被应用于决策树。

* * *

### [**数据集**](#TOC)

为了开始我们的旅程，我们将考虑**表 1** 中所示的虚拟数据集。

这个数据集的特征主要是“课程”和“项目”，而“工作”列中的标签告诉我们这个人现在是否有工作。如果我们采用决策树方法，我们将需要找出一个合适的标准来将数据集分割成相应的标签。

现在，关于自适应增强，需要记住的一个关键点是，我们是按顺序创建树，并将每个树产生的错误传递给下一个树。在随机森林中，我们会创建全尺寸的树，但在自适应增强中，创建的树是树桩(深度为 1 的树)。

现在，一个总的思路是，一棵树基于某种标准产生的误差决定了第二棵树的性质。让一棵复杂的树过度适应数据将会破坏将许多弱学习者组合成强学习者的目的。然而，AdaBoost 在更复杂的树中也显示了结果。

* * *

### [**样本权重**](#TOC)

下一步是为数据集中的每个样本分配一个样本权重。最初，所有样本将具有相同的权重，并且它们的权重必须加起来为 1 ( **表 2** )。随着我们向最终目标迈进，样品重量的概念将更加清晰。

现在，由于我们有 7 个数据样本，我们将为每个样本分配 1/7。这意味着所有样本在决策树的最终结构中的重要性是同等重要的。

* * *

### [**选择合适的特征**](#TOC)

现在我们必须用一些信息初始化我们的第一个 stump。

粗略地看一下，我们可以看到，对于这个虚拟数据集,“courses”特性比“project”特性更好地分割了数据集。让我们在**图 1** 中亲眼看看这个。

这里的分割因子是一个学生所学课程的数量。我们注意到，在选修 7 门以上课程的学生中，所有人都有工作。相反，对于少于或等于 7 门课程的学生，大多数没有工作。

然而，这种分裂给了我们 6 个正确的预测和 1 个错误的预测。这意味着如果我们使用这个特殊的残肢，我们数据集中的一个样本会给我们错误的预测。

* * *

### [**一个树桩的意义**](#TOC)

现在，我们特别的树桩做了一个错误的预测。为了计算误差，我们必须将所有未正确分类的样本的权重相加，在本例中为 1/7，因为只有一个样本被错误分类。

现在，AdaBoost 和随机森林之间的一个关键区别是，在前者中，一个树桩在最终输出中可能比其他树桩具有更大的权重。所以这里的问题变成了如何计算

![\left(\displaystyle\frac{1}{2} \right) \ln\left(\displaystyle\frac{1 - \text{loss}}{\text{loss}}\right).](img/89070d622948c4d0036a67eaac10d7b1.png "\left(\displaystyle\frac{1}{2} \right) \ln\left(\displaystyle\frac{1 - \text{loss}}{\text{loss}}\right).")

所以，如果我们把损失值(1/7)代入这个公式，我们得到 0.89。请注意，公式应该给出介于 0 和 1 之间的值。所以 0.89 是一个非常高的值，告诉我们这个特定的树桩对树桩组合的最终输出有很大的发言权。

* * *

### [**计算新样本权重**](#TOC)

如果我们已经建立了一个分类器，可以对除了一个样本之外的所有样本进行正确的预测，那么我们的自然行动应该是确保我们更多地关注那个特定的样本，以便将其分组到正确的类别中。

目前，所有样品的重量相同(1/7)。但是我们希望我们的分类器更多地关注错误分类的样本。为此，我们必须改变权重。

请记住，当我们改变样品的重量时，所有其他样品的重量也会改变，只要所有的重量加起来为 1。

计算错误分类样本的新权重的公式为

![\text{old weight} \times e^\text{(significance of the model)}](img/11715ce0be1d3623d1deb2ebb31d83e8.png "\text{old weight} \times e^\text{(significance of the model)}")

.

插入虚拟数据集中的值就变成了

![(1/7) \times e^{(0.89)}](img/9ce2ec1cb7ce36e458de600aa161fb34.png "(1/7) \times e^{(0.89)}")

,

得出的值是 0.347。

以前，这个特定样本的样本重量是 1/7，现在变成≈0.142。新的权重为 0.347，意味着样本显著性增加。现在，我们需要相应地移动其余的样本权重。

这个公式是

![(1/7) \times e^{(-0.89)}](img/7e650414b98cd07370ce991187c45965.png "(1/7) \times e^{(-0.89)}")

,

一共是 0.0586。所以现在，代替 1/7，6 个正确预测的样本将具有权重 0.0586，而错误预测的样本将具有权重 0.347。

这里的最后一步是归一化权重，因此它们相加为 1，这使得错误预测的样本的权重为 0.494，而其他样本的权重为 0.084。

* * *

### [**向前移动:后续树桩**](#TOC)

现在有几种方法可以确定下一个树桩会是什么样子。我们目前的首要任务是确保第二个 stump 用较大的权重对样本进行分类(我们在最后一次迭代中发现权重为 0.494)。

这个样本的重点是因为这个样本导致了我们之前残肢的错误。

虽然没有选择新残肢的标准方法，因为大多数都很好，但一种流行的方法是从我们当前的数据集及其样本权重创建一个新的数据集。

让我们回忆一下目前我们走了多远(**表 3** )。

我们的新数据集将来自基于当前权重的随机采样。我们将创建范围的桶。范围如下:

*   我们选择值在 0 和 0.084 之间的第一个样本。
*   对于 0.084 和 0.168 之间的值，我们选择第二个样本。
*   我们选择第三个样本的值为 0.168 和 0.252。
*   我们选择第四个样本的值为 0.252 和 0.746。

到目前为止，我希望您已经理解了我们在这里遵循的模式。这些桶是基于权重形成的:0.084(第一样本)、0.084 + 0.084(第二样本)、0.084 + 0.084(第三样本)、0.084 + 0.084 + 0.494(第四样本)，依此类推。

自然地，由于第四个样本具有更大的范围，如果应用随机抽样，它将在我们的下一个数据集中出现更多次。因此，我们的新数据集将有 7 个条目，但它可能有 4 个属于姓名“Tom”的条目我们重置了新数据集的权重，但是直觉上，由于“Tom”出现得更频繁，这将显著影响树的结果。

因此，我们重复这个过程来寻找创建残肢的最佳特征。这一次，由于数据集不同，我们将使用不同的特征而不是球场数量来创建树桩。

请注意，我们一直在说，第一个树桩的输出将决定第二棵树如何工作。这里我们看到了从第一个树桩上获得的重量是如何决定第二棵树如何工作的。对 N 个树桩冲洗并重复该过程。

* * *

### [](#TOC)

 **我们已经了解了树桩及其产出的重要性。假设我们给我们的树桩组喂一个测试样本。第一个树桩给 1，第二个给 1，第三个给 0，第四个也给 0。

但是由于每棵树都有一个重要值，我们将得到一个最终的加权输出，确定输入到树桩的测试样本的性质。

随着 AdaBoost 的完成，我们离理解 XGBoost 算法又近了一步。现在，我们将处理一个稍微中级的 Kaggle 任务，并使用 XGBoost 解决它。

* * *

### [**配置您的开发环境**](#TOC)

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果您需要帮助配置 OpenCV 的开发环境，我们*强烈推荐*阅读我们的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让您启动并运行。

* * *

### [**在配置开发环境时遇到了问题？**](#TOC)

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

* * *

### [**设置先决条件**](#TOC)

今天，我们将处理[美国房地产数据集](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset)。让我们从为我们的项目导入必要的包开始。

```py
# import the necessary packages
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

在我们的进口中，我们有`pandas`、`xgboost`，以及来自`scikit-learn`的一些实用函数，如**行 2-5** 上的`mean_squared_error`和`train_test_split`。

```py
# load the data in the form of a csv
estData =   pd.read_csv("/content/realtor-data.csv")

# drop NaN values from the dataset
estData = estData.dropna()

# split the labels and remove non-numeric data
y = estData["price"].values
X = estData.drop(["price"], axis=1).select_dtypes(exclude=['object'])
```

在第 8 行的**上，我们使用`read_csv`来加载 csv 文件作为熊猫数据帧。对数据集进行一些探索性数据分析(EDA)会显示许多 NaN(非数字或未定义的)值。这些会妨碍我们的训练，所以我们丢弃了第 11 行**的**中所有具有 NaN 值的元组。**

我们继续准备第 14 行**和第 15 行**的标签和特征。我们在我们的特征集中排除任何不包含数值的列。包含这些数据是可能的，但是我们需要将它转换成一次性编码。

我们最终的数据集看起来类似于**表 4** 。

我们将使用这些特征来确定房子的价格(标签)。

* * *

### [**建立模型**](#TOC)

我们的下一步是初始化和训练 XGBoost 模型。

```py
# create the train test split
xTrain, xTest, yTrain, yTest = train_test_split(X, y)

# define the XGBoost regressor according to your specifications
xgbModel = xgb.XGBRegressor(
    n_estimators=1000,
    reg_lambda=1,
    gamma=0,
    max_depth=4
)

# fit the data into the model
xgbModel.fit(xTrain, yTrain,
             verbose = False)

# calculate the importance of each feature used
impFeat = pd.DataFrame(xgbModel.feature_importances_.reshape(1, -1), columns=X.columns)
```

在第 18 行的**上，我们使用`train_test_split`函数将数据集分成训练集和测试集。对于 XGBoost 模型，我们使用 1000 棵树和最大树深度 4 来初始化它(**第 21-26 行**)。**

我们在第 29 行和第 30 行将训练数据放入我们的模型中。一旦我们的培训完成，我们可以评估我们的模型更重视哪些特性(**表 5** )。

**表 5** 显示特征“bath”最重要。

* * *

### [**评估模型**](#TOC)

我们的下一步是看看我们的模型在看不见的测试数据上表现如何。

```py
# get predictions on test data
yPred = xgbModel.predict(xTest)

# store the msq error from the predictions
msqErr = mean_squared_error(yPred, yTest)

# assess your model’s results
xgbModel.score(xTest, yTest)
```

在**第 36-39 行**上，我们得到了我们的模型对测试数据集的预测，并计算了均方误差(MSE)。由于这是一个回归模型，因此高 MSE 值可以让您放心。

我们的最后一步是使用`model.score`函数获得测试集的精度值(**第 42 行**)。

准确率显示我们的模型在测试集上有将近 95%的准确率。

* * *

* * *

## [**汇总**](#TOC)

在本教程中，我们首先了解了 XGBoost 的先决条件之一——自适应增强背后的数学原理。然后，就像我们在本系列中的第一篇博文一样，我们处理了来自 Kaggle 的数据集，并使用 XGBoost 在测试数据集上获得了相当高的准确度，再次确立了 XGBoost 作为领先的经典机器学习技术之一的主导地位。

在我们最终剖析 XGBoost 的基础之前，我们的下一站将是找出梯度增强背后的数学原理。

* * *

### [**引用信息**](#TOC)

**Martinez，H.** “使用 XGBoost 扩展 Kaggle 竞赛:第 2 部分”， *PyImageSearch* ，P. Chugh，A. R. Gosthipaty，S. Huot，K. Kidriavsteva，R. Raha 和 A. Thanki 编辑。，2022 年，【https://pyimg.co/2wiy7 

```py
@incollection{Martinez_2022_XGBoost2,
  author = {Hector Martinez},
  title = {Scaling {Kaggle} Competitions Using {XGBoost}: Part 2},
  booktitle = {PyImageSearch},
  editor = {Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/2wiy7},
}
```

* * *

* * *

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******