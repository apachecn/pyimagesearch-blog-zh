# 用 Python 实现感知器神经网络

> 原文：<https://pyimagesearch.com/2021/05/06/implementing-the-perceptron-neural-network-with-python/>

由 [Rosenblatt 在 1958 年首次介绍的*感知器:大脑中信息存储和组织的概率模型*](https://doi.org/10.1037/h0042519) 可以说是最古老和最简单的人工神经网络算法。这篇文章发表后，基于感知器的技术在神经网络社区风靡一时。单单这篇论文就对今天神经网络的流行和实用性负有巨大的责任。

但后来，在 1969 年，一场“人工智能冬天”降临到机器学习社区，几乎永远冻结了神经网络。[明斯基和帕佩特出版了*感知器:计算几何*](https://mitpress.mit.edu/books/perceptrons) 导论，这本书实际上使神经网络的研究停滞了近十年——关于这本书有很多争议([奥拉扎兰，1996](http://www.jstor.org/stable/285702) )，但作者确实成功证明了*单层*感知器无法分离非线性数据点。

鉴于大多数真实世界的数据集自然是非线性可分的，这似乎是感知器，以及神经网络研究的其余部分，可能会达到一个不合时宜的结束。

在 Minsky 和 Papert 的出版物和神经网络使工业发生革命性变化的承诺之间，对神经网络的兴趣大大减少了。直到我们开始探索更深层次的网络(有时称为*多层感知器*)以及反向传播算法([沃博斯](https://www.worldcat.org/title/beyond-regression-new-tools-for-prediction-and-analysis-in-the-behavioral-sciences/oclc/77001455)和[鲁梅尔哈特等人](http://dl.acm.org/citation.cfm?id=65669.104451))，20 世纪 70 年代的“人工智能冬天”才结束，神经网络研究又开始升温。

尽管如此，感知器仍然是一个需要理解的非常重要的算法，因为它为更高级的多层网络奠定了基础。我们将从感知机架构的回顾开始这一部分，并解释用于训练感知机的训练程序(称为**德尔塔规则**)。我们还将看看网络的*终止标准*(即，感知机何时应该停止训练)。最后，我们将在纯 Python 中实现感知器算法，并使用它来研究和检查网络如何无法学习非线性可分离数据集。

### **与、或和异或数据集**

在研究感知机本身之前，我们先来讨论一下“按位运算”，包括 AND、OR 和 XOR(异或)。如果你以前学过计算机科学的入门课程，你可能已经熟悉了位函数。

按位运算符和关联的按位数据集接受两个输入位，并在应用运算后生成一个最终输出位。给定两个输入位，每个可能取值为`0`或`1`，这两个位有四种可能的组合— **表 1** 提供了 and、or 和 XOR 的可能输入和输出值:

| ***x***[0] | ***x***[1] | ***x*[0]&x[1]** | ***x***[0] | ***x***[1] | ***x***[**0**]***x***[**1**] | ***x***[0] | ***x***[1] | ***x***[**0**]**∧*****x***[**1**] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | Zero | Zero | Zero | Zero | Zero | Zero | Zero | Zero |
| Zero | one | Zero | Zero | one | one | Zero | one | one |
| one | Zero | Zero | one | Zero | one | one | Zero | one |
| one | one | one | one | one | one | one | one | Zero |

**Table 1:** *Left:* The bitwise AND dataset. Given two inputs, the output is only 1 if both inputs are 1\. *Middle:* The bitwise OR dataset. Given two inputs, the output is 1 if *either* of the two inputs is 1\. *Right:* The XOR (e(X)clusive OR) dataset. Given two inputs, the output is 1 if and only if one of the inputs is 1, *but not both*.

正如我们在左边的*上看到的，当且仅当*两个输入值都是`1`时，逻辑与为真*。如果输入值的*或*为`0`，则 AND 返回`0`。因此，当 and 的输出为真时，只有一个组合，*x*0= 1 和 *x* [1] = 1。*

在*中间的*中，我们有 OR 运算，当*输入值中至少有一个*为`1`时，OR 运算为真。因此，产生值 *y* = 1 的两位*x*0 和*x*T11 有三种可能的组合。

最后，*右*显示异或运算，当且仅当一个输入为`1` *但不是两个都为*时，该运算为真*。OR 有三种可能的情况，其中 *y* = 1，而 XOR 只有两种。*

我们经常使用这些简单的“按位数据集”来测试和调试机器学习算法。如果我们在**图 1** 中绘制并可视化 and、OR 和 XOR 值(红色圆圈表示零输出，蓝色星号表示一输出)，您会注意到一个有趣的模式:

AND 和 OR 都是线性可分的——我们可以清楚地画出一条线来区分`0`和`1`类 XOR 则不是这样。现在花点时间说服自己*不可能*在异或问题中划出一条清晰的分界线来区分这两个类。因此，XOR 是一个*非线性可分离*数据集的例子。

理想情况下，我们希望我们的机器学习算法能够分离非线性类，因为现实世界中遇到的大多数数据集都是非线性的。因此，当构造、调试和评估给定的机器学习算法时，我们可以使用按位值 *x* 、T2【0】T3 和 *x* 、 1 作为我们的*设计矩阵*，然后尝试预测相应的 *y* 值。

与我们将数据分割成*训练*和*测试*分割的标准程序不同，当使用逐位数据集时，我们只是在同一组数据上训练和评估我们的网络。我们在这里的目标仅仅是确定我们的学习算法是否有*可能*学习数据中的模式。我们将会发现，感知器算法可以正确分类 AND 和 OR 函数，但无法分类 XOR 数据。

### **感知器架构**

[Rosenblatt (1958)](https://doi.org/10.1037/h0042519) 将感知器定义为使用特征向量(或原始像素强度)的标记示例(即监督学习)进行学习的系统，将这些输入映射到其相应的输出类别标签。

在其最简单的形式中，感知器包含 *N* 个输入节点，一个用于设计矩阵的*输入行*中的每个条目，随后是网络中的*仅一层*，在该层中仅有一个*单节点*(**图 2** )。

从输入端 *x [i]* 到网络中的单个输出节点存在连接和它们相应的权重 *w* [1] *，w* [2] *，…，w [i]* 。该节点获取输入的加权和，并应用*阶跃函数*来确定输出类别标签。感知器为类#1 输出`0`或`1` — `0`，为类#2 输出`1`；因此，在其原始形式中，感知器只是一个二元、两类分类器。

| 1.用小的随机值
2 初始化我们的权重向量 ***w*** 。直到感知器收敛:
(a)循环遍历我们训练集 *D* 中的每个特征向量 *x [j]* 和真实类标签*D[I]*(b)取 ***x*** 通过网络， 计算输出值:*y[j]j*=*f*(***w*(*t*)*****x[j]***)
(c)更新权重***w***:*w[I]*( +*α*(*d[j]y[j]*)*x[j，i]* 对于所有特性 0*<*=*I<*=*n* |

**Figure 3:** The Perceptron algorithm training procedure.

### **感知器训练程序和德尔塔法则**

训练一个感知机是一个相当简单的操作。我们的目标是获得一组权重 ***w*** ，这些权重能够准确地对我们训练集中的每个实例进行分类。为了训练我们的感知机，我们多次用我们的训练数据*迭代地输入网络*。每当网络已经看到训练数据的*全套*，我们就说一个*时期*已经过去。通常需要许多代才能学习到权重向量 ***w*** 来线性分离我们的两类数据。

感知器训练算法的伪代码可以在下面找到:

实际的“学习”发生在步骤 2b 和 2c 中。首先，我们通过网络传递特征向量*x[j]，取与权重*w的点积，得到输出*y[j]。然后，该值通过 step 函数传递，如果*x>0，该函数将返回 1，否则返回 0。****

现在我们需要更新我们的权重向量 ***w*** 以朝着“更接近”正确分类的方向前进。权重向量的更新由步骤 2c 中的*德尔塔规则*处理。

表达式(*d**[j]**——y**[j]*)决定输出分类是否正确。如果分类是*正确的*，那么这个差值将为零。否则，差异将是正的或负的，给我们权重更新的方向(最终使我们更接近正确的分类)。然后我们将(*d**[j]**——y**[j]*)乘以 *x* *[j]* ，让我们更接近正确的分类。

值 *α* 是我们的*学习速率*，它控制着我们迈出的一步的大小。该值设置正确是*的关键*。更大的值 *α* 会使我们朝着正确的方向迈出一步；然而，这一步可能*太大*，我们很容易超越局部/全局最优。

相反，一个小的值 *α* 允许我们在正确的方向上迈出小步，确保我们不会超越局部/全局最小值；然而，这些小小的步骤可能需要很长时间才能让我们的学习趋于一致。

最后，我们在时间 *t* 、*wjT7(*t*)添加先前的权重向量，这完成了朝向正确分类的“步进”过程。如果你觉得这个训练过程有点混乱，不要担心。*

### **感知器训练终止**

感知器训练过程被允许继续进行，直到所有训练样本被正确分类*或*达到预设数量的时期。如果 *α* 足够小*并且*训练数据是线性可分的，则确保终止。

那么，如果我们的数据不是线性可分的，或者我们在 *α* 中做了一个糟糕的选择，会发生什么？训练会无限延续下去吗？在这种情况下，否-我们通常在达到设定数量的历元后停止，或者如果在大量历元中错误分类的数量没有变化(表明数据不是线性可分的)。关于感知机算法的更多细节，请参考[吴恩达的斯坦福讲座](https://www.coursera.org/learn/machine-learning)或[梅罗塔等人(1997)](https://mitpress.mit.edu/books/elements-artificial-neural-networks) 的介绍章节。

### **在 Python 中实现感知器**

现在我们已经学习了感知器算法，让我们用 Python 实现实际的算法。在您的`pyimagesearch.nn`包中创建一个名为`perceptron.py`的文件——这个文件将存储我们实际的`Perceptron`实现:

```py
|--- pyimagesearch
|    |--- __init__.py
|    |--- nn
|    |    |--- __init__.py
|    |    |--- perceptron.py
```

创建文件后，打开它，并插入以下代码:

```py
# import the necessary packages
import numpy as np

class Perceptron:
	def __init__(self, N, alpha=0.1):
		# initialize the weight matrix and store the learning rate
		self.W = np.random.randn(N + 1) / np.sqrt(N)
		self.alpha = alpha
```

**第 5 行**定义了我们的`Perceptron`类的构造函数，它接受一个必需的参数，后跟第二个可选的参数:

1.  `N`:我们输入特征向量中的列数。在我们的按位数据集的上下文中，我们将设置`N`等于 2，因为有两个输入。
2.  感知器算法的学习率。默认情况下，我们将这个值设置为`0.1`。学习率的常见选择通常在 *α* = 0 *的范围内。* 1 *，* 0 *。* 01 *，* 0 *。* 001。

**第 7 行**用从均值和单位方差为零的“正态”(高斯)分布中采样的随机值填充我们的权重矩阵`W`。权重矩阵将具有 *N* +1 个条目，一个用于特征向量中的每个`N`输入，另一个用于偏差。我们将`W`除以输入数量的平方根，这是一种用于调整权重矩阵的常用技术，可以加快收敛速度。我们将在本章后面讨论权重初始化技术。

接下来，让我们定义`step`函数:

```py
	def step(self, x):
		# apply the step function
		return 1 if x > 0 else 0
```

这个函数模拟了步进方程的行为——如果`x`为正，我们返回`1`，否则，我们返回`0`。

为了实际训练感知器，我们将定义一个名为`fit`的函数。如果你以前有过机器学习、Python 和 scikit-learn 库的经验，那么你会知道将你的训练过程函数命名为`fit`是很常见的，比如*“根据数据拟合模型”*:

```py
	def fit(self, X, y, epochs=10):
		# insert a column of 1's as the last entry in the feature
		# matrix -- this little trick allows us to treat the bias
		# as a trainable parameter within the weight matrix
		X = np.c_[X, np.ones((X.shape[0]))]
```

`fit`方法需要两个参数，后跟一个可选参数:

`X`值是我们实际的训练数据。 *`y`* 变量是我们的目标输出类标签(即，我们的网络*应该*预测什么)。最后，我们提供`epochs`，我们的感知机将训练的纪元数量。

**第 18 行**通过在训练数据中插入一列 1 来应用偏差技巧，这允许我们将偏差作为权重矩阵中的可训练参数*直接*来处理。

接下来，让我们回顾一下实际的培训程序:

```py
		# loop over the desired number of epochs
		for epoch in np.arange(0, epochs):
			# loop over each individual data point
			for (x, target) in zip(X, y):
				# take the dot product between the input features
				# and the weight matrix, then pass this value
				# through the step function to obtain the prediction
				p = self.step(np.dot(x, self.W))

				# only perform a weight update if our prediction
				# does not match the target
				if p != target:
					# determine the error
					error = p - target

					# update the weight matrix
					self.W += -self.alpha * error * x
```

在第 21 行的**上，我们开始循环所需数量的`epochs`。对于每个时期，我们还循环每个单独的数据点`x`并输出`target`类标签(**行 23** )。**

**第 27 行**获取输入特征`x`和权重矩阵`W`之间的点积，然后通过`step`函数传递输出，以获得感知器的预测。

应用**图 3** 中详述的相同训练程序，我们仅在我们的预测*与目标*不匹配的情况下执行权重更新(**第 31 行**)。如果是这种情况，我们通过差分运算计算符号(正或负)来确定`error` ( **行 33** )。

在**第 36 行**处理权重矩阵的更新，在这里我们向正确的分类迈出一步，通过我们的学习速率`alpha`缩放这一步。经过一系列时期，我们的感知机能够学习底层数据中的模式，并移动权重矩阵的值，以便我们正确地对输入样本进行分类`x`。

我们需要定义的最后一个函数是`predict`，顾名思义，它用于*预测*给定输入数据集的类别标签:

```py
	def predict(self, X, addBias=True):
		# ensure our input is a matrix
		X = np.atleast_2d(X)

		# check to see if the bias column should be added
		if addBias:
			# insert a column of 1's as the last entry in the feature
			# matrix (bias)
			X = np.c_[X, np.ones((X.shape[0]))]

		# take the dot product between the input features and the
		# weight matrix, then pass the value through the step
		# function
		return self.step(np.dot(X, self.W))
```

我们的`predict`方法需要一组需要分类的输入数据`X`。对**线 43** 进行检查，查看是否需要添加偏置柱。

获取`X`的输出预测与训练过程相同——只需获取输入特征`X`和我们的权重矩阵`W`之间的点积，然后通过我们的阶跃函数传递该值。阶跃函数的输出返回给调用函数。

现在我们已经实现了我们的`Perceptron`类，让我们试着将它应用到我们的按位数据集，看看神经网络如何执行。

### **评估感知器逐位数据集**

首先，让我们创建一个名为`perceptron_or.py`的文件，该文件试图将感知器模型与按位 OR 数据集相匹配:

```py
# import the necessary packages
from pyimagesearch.nn import Perceptron
import numpy as np

# construct the OR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

# define our perceptron and train it
print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)
```

**第 2 行和第 3 行**导入我们需要的 Python 包。我们将使用我们的`Perceptron`实现。**第 6 行和第 7 行**根据**表 1** 定义 OR 数据集。

**第 11 行和第 12 行**以 *α* = 0 *的学习率训练我们的感知机。* 1 共 20 个历元。

然后，我们可以根据数据评估我们的感知器，以验证它确实学习了 OR 函数:

```py
# now that our perceptron is trained we can evaluate it
print("[INFO] testing perceptron...")

# now that our network is trained, loop over the data points
for (x, target) in zip(X, y):
	# make a prediction on the data point and display the result
	# to our console
	pred = p.predict(x)
	print("[INFO] data={}, ground-truth={}, pred={}".format(
		x, target[0], pred))
```

在**第 18** 行，我们循环 OR 数据集中的每个数据点。对于这些数据点中的每一个，我们通过网络传递它并获得预测(**第 21 行**)。

最后，**行 22 和 23** 向我们的控制台显示输入数据点、地面实况标签以及我们的预测标签。

要查看我们的感知器算法是否能够学习 OR 函数，只需执行以下命令:

```py
$ python perceptron_or.py 
[INFO] training perceptron...
[INFO] testing perceptron...
[INFO] data=[0 0], ground-truth=0, pred=0
[INFO] data=[0 1], ground-truth=1, pred=1
[INFO] data=[1 0], ground-truth=1, pred=1
[INFO] data=[1 1], ground-truth=1, pred=1
```

果然，我们的神经网络能够正确地预测出 *x* [0] = 0 和 *x* [1] = 0 的 OR 运算是零——所有其他组合都是一。

现在，让我们继续讨论 AND 函数——创建一个名为`perceptron_and.py`的新文件，并插入以下代码:

```py
# import the necessary packages
from pyimagesearch.nn import Perceptron
import numpy as np

# construct the AND dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# define our perceptron and train it
print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

# now that our perceptron is trained we can evaluate it
print("[INFO] testing perceptron...")

# now that our network is trained, loop over the data points
for (x, target) in zip(X, y):
	# make a prediction on the data point and display the result
	# to our console
	pred = p.predict(x)
	print("[INFO] data={}, ground-truth={}, pred={}".format(
		x, target[0], pred))
```

注意，这里只有*行代码发生了变化，这是**的第 6 行和第 7 行**，在这里我们定义了 and 数据集，而不是 OR 数据集。*

执行以下命令，我们可以评估 and 函数上的感知器:

```py
$ python perceptron_and.py 
[INFO] training perceptron...
[INFO] testing perceptron...
[INFO] data=[0 0], ground-truth=0, pred=0
[INFO] data=[0 1], ground-truth=0, pred=0
[INFO] data=[1 0], ground-truth=0, pred=0
[INFO] data=[1 1], ground-truth=1, pred=1
```

同样，我们的感知器能够正确地模拟这个函数。只有当*x*[0]= 1 且 *x* [1] = 1 时，and 函数才成立——对于所有其他组合，按位 AND 为零。

最后，我们来看看`perceptron_xor.py`内部的非线性可分 XOR 函数:

```py
# import the necessary packages
from pyimagesearch.nn import Perceptron
import numpy as np

# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define our perceptron and train it
print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

# now that our perceptron is trained we can evaluate it
print("[INFO] testing perceptron...")

# now that our network is trained, loop over the data points
for (x, target) in zip(X, y):
	# make a prediction on the data point and display the result
	# to our console
	pred = p.predict(x)
	print("[INFO] data={}, ground-truth={}, pred={}".format(
		x, target[0], pred))
```

同样，唯一被修改的代码行是第 6 行**和第 7 行**，在那里我们定义了 XOR 数据。异或运算符为真*当且仅当*一(但不是两个) *x* 为一。

执行以下命令，我们可以看到感知器*无法*学习这种非线性关系:

```py
$ python perceptron_xor.py 
[INFO] training perceptron...
[INFO] testing perceptron...
[INFO] data=[0 0], ground-truth=0, pred=1
[INFO] data=[0 1], ground-truth=1, pred=1
[INFO] data=[1 0], ground-truth=1, pred=0
[INFO] data=[1 1], ground-truth=0, pred=0
```

无论你用不同的学习速率或不同的权重初始化方案运行这个实验多少次，你都*永远*不能用单层感知器正确地模拟异或函数。相反，我们需要的是*更多层*和*非线性激活函数*，随之而来的是深度学习的开始。
****