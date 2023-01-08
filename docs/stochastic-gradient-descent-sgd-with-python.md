# 用 Python 实现随机梯度下降(SGD)

> 原文：<https://pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/>

在上一节中，我们讨论了梯度下降，这是一种一阶优化算法，可用于学习一组用于参数化学习的分类器权重。然而，梯度下降的这种“普通”实现在大型数据集上运行会非常慢——事实上，它甚至可以被认为是计算浪费。

相反，我们应该应用**随机梯度下降(SGD)** ，这是对标准梯度下降算法的一个简单修改，该算法由*计算梯度*，由*更新权重矩阵* ***W*** 对**小批训练数据**，而不是整个训练集。虽然这种修改导致“更多噪声”的更新，但它也允许我们沿着梯度采取*更多的步骤(每批一步对每个时期一步)，最终导致更快的收敛，并且对损失和分类精度没有负面影响。*

谈到训练深度神经网络，SGD 可以说是**最重要的算法。即使 SGD 的最初版本是在 57 年前推出的([斯坦福电子实验室等，1960](https://www.google.com/books/edition/Adaptive_adaline_Neuron_Using_Chemical_m/Yc4EAAAAIAAJ?hl=en) )，它仍然是*使我们能够训练大型网络从数据点学习模式的引擎。在本书涵盖的所有其他算法中，请花时间理解 SGD。***

 **### **小批量新币**

回顾一下传统的梯度下降算法，很明显该方法在大型数据集上运行*非常慢*。这种缓慢的原因是因为梯度下降的每次迭代需要我们在允许我们更新我们的权重矩阵之前为我们的训练数据*中的每个训练点计算预测。对于像 ImageNet 这样的图像数据集，我们有超过*120 万张*训练图像，这种计算可能需要很长时间。*

事实证明，在沿着我们的权重矩阵前进之前，计算每个训练点的*预测是计算上的浪费，并且对我们的模型覆盖没有什么帮助。*

相反，我们应该做的是*批量*我们的更新。我们可以通过添加额外的函数调用来更新伪代码，将 vanilla gradient descent 转换为 SGD:

```py
while True:
	batch = next_training_batch(data, 256)
	Wgradient = evaluate_gradient(loss, batch, W)
	W += -alpha * Wgradient
```

普通梯度下降和 SGD 的唯一区别是增加了`next_training_batch`函数。我们不是在整个*`data`集合上计算我们的梯度，而是对我们的数据进行采样，产生一个`batch`。我们评估`batch`上的梯度，并更新我们的权重矩阵`W`。从实现的角度来看，我们还试图在应用 SGD 之前随机化我们的训练样本*，因为该算法对批次敏感。**

 *在查看了 SGD 的伪代码之后，您会立即注意到一个新参数的引入:*。在 SGD 的“纯粹”实现中，您的小批量大小将是 1，这意味着我们将从训练集中随机抽取*一个*数据点，计算梯度，并更新我们的参数。然而，我们经常使用的小批量是 *>* 1。典型的批量大小包括 32、64、128 和 256。*

 *那么，为什么要使用批量大小 *>* 1 呢？首先，批量大小 *>* 1 有助于减少参数更新(【http://pyimg.co/pd5w0】)中的方差，从而导致更稳定的收敛。其次，对于批量大小，2 的幂通常是可取的，因为它们允许内部线性代数优化库更有效。

一般来说，小批量不是一个你应该过分担心的超参数([http://cs231n.stanford.edu](http://cs231n.stanford.edu/))。如果您使用 GPU 来训练您的神经网络，您可以确定有多少训练示例适合您的 GPU，然后使用最接近的 2 的幂作为批处理大小，以便该批处理适合 GPU。对于 CPU 培训，您通常使用上面列出的批量大小之一，以确保您获得线性代数优化库的好处。

### **实施小批量 SGD**

让我们继续实现 SGD，看看它与标准的普通梯度下降有什么不同。打开一个新文件，将其命名为`sgd.py`，并插入以下代码:

```py
# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
	# compute the sigmoid activation value for a given input
	return 1.0 / (1 + np.exp(-x))

def sigmoid_deriv(x):
	# compute the derivative of the sigmoid function ASSUMING
	# that the input "x" has already been passed through the sigmoid
	# activation function
	return x * (1 - x)
```

**第 2-7 行**导入我们需要的 Python 包，和本章前面的`gradient_descent.py`例子完全一样。**第 9-17 行**定义了我们的`sigmoid_activation`和`sigmoid_deriv`函数，这两个函数与之前版本的梯度下降相同。

事实上，`predict`方法也没有改变:

```py
def predict(X, W):
	# take the dot product between our features and weight matrix
	preds = sigmoid_activation(X.dot(W))

	# apply a step function to threshold the outputs to binary
	# class labels
	preds[preds <= 0.5] = 0
	preds[preds > 0] = 1

	# return the predictions
	return preds
```

然而，改变的*是增加了`next_batch`功能:*

```py
def next_batch(X, y, batchSize):
	# loop over our dataset "X" in mini-batches, yielding a tuple of
	# the current batched data and labels
	for i in np.arange(0, X.shape[0], batchSize):
		yield (X[i:i + batchSize], y[i:i + batchSize])
```

`next_batch`方法需要三个参数:

1.  特征向量/原始图像像素强度的训练数据集。
2.  `y`:与每个训练数据点相关联的类别标签。
3.  `batchSize`:将被退回的每个小批量的大小。

**第 34 行和第 35 行**然后遍历训练示例，产生`X`和`y`的子集作为小批量。

接下来，我们可以解析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
	help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32,
	help="size of SGD mini-batches")
args = vars(ap.parse_args())
```

我们已经回顾了普通梯度下降示例中的`--epochs`(历元数)和`--alpha`(学习速率)开关，但也注意到我们引入了第三个开关:`--batch-size`，顾名思义，它是我们每个小批量的大小。我们将默认该值为每个小批量的`32`个数据点。

我们的下一个代码块处理生成具有 1，000 个数据点的 2 类分类问题，添加偏差列，然后执行训练和测试分割:

```py
# generate a 2-class classification problem with 1,000 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
	cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# insert a column of 1's as the last entry in the feature
# matrix -- this little trick allows us to treat the bias
# as a trainable parameter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% of
# the data for training and the remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y,
	test_size=0.5, random_state=42)
```

然后，我们将初始化权重矩阵和损失，就像前面的例子一样:

```py
# initialize our weight matrix and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []
```

接下来是*真正的*变化，我们循环期望的历元数，沿途采样小批量:

```py
# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
	# initialize the total loss for the epoch
	epochLoss = []

	# loop over our data in batches
	for (batchX, batchY) in next_batch(trainX, trainY, args["batch_size"]):
		# take the dot product between our current batch of features
		# and the weight matrix, then pass this value through our
		# activation function
		preds = sigmoid_activation(batchX.dot(W))

		# now that we have our predictions, we need to determine the
		# "error", which is the difference between our predictions
		# and the true values
		error = preds - batchY
		epochLoss.append(np.sum(error ** 2))
```

在第 69 行**，**上，我们开始循环所提供的`--epochs`号。然后我们在**行 74** 上批量循环我们的训练数据。对于每个批次，我们计算批次和`W`之间的点积，然后将结果通过 sigmoid 激活函数来获得我们的预测。我们计算第**行 83** 上批次的误差，并使用该值更新第**行 84** 上的最小二乘`epochLoss`。

现在我们有了`error`，我们可以计算梯度下降更新，与从普通梯度下降计算梯度相同，只是这次我们对*批*而不是*整个*训练集执行更新:

```py
		# the gradient descent update is the dot product between our
		# (1) current batch and (2) the error of the sigmoid
		# derivative of our predictions
		d = error * sigmoid_deriv(preds)
		gradient = batchX.T.dot(d)

		# in the update stage, all we need to do is "nudge" the
		# weight matrix in the negative direction of the gradient
		# (hence the term "gradient descent" by taking a small step
		# towards a set of "more optimal" parameters
		W += -args["alpha"] * gradient
```

**第 96 行**处理基于梯度更新我们的权重矩阵，由我们的学习速率`--alpha`缩放。注意重量更新阶段是如何在批处理循环中发生的*——这意味着每个时期有*多次重量更新*。*

然后，我们可以通过取时段中所有批次的平均值来更新我们的损失历史，然后在必要时向我们的终端显示更新:

```py
	# update our loss history by taking the average loss across all
	# batches
	loss = np.average(epochLoss)
	losses.append(loss)

	# check to see if an update should be displayed
	if epoch == 0 or (epoch + 1) % 5 == 0:
		print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1),
			loss))
```

评估我们的分类器的方式与普通梯度下降法相同——只需使用我们学习的`W`权重矩阵对`testX`数据调用`predict`:

```py
# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))
```

我们将通过绘制测试分类数据以及每个时期的损失来结束我们的脚本:

```py
# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
```

### **SGD 结果**

要可视化我们实现的结果，只需执行以下命令:

```py
$ python sgd.py
[INFO] training...
[INFO] epoch=1, loss=0.1317637
[INFO] epoch=5, loss=0.0162487
[INFO] epoch=10, loss=0.0112798
[INFO] epoch=15, loss=0.0100234
[INFO] epoch=20, loss=0.0094581
[INFO] epoch=25, loss=0.0091053
[INFO] epoch=30, loss=0.0088366
[INFO] epoch=35, loss=0.0086082
[INFO] epoch=40, loss=0.0084031
[INFO] epoch=45, loss=0.0082138
[INFO] epoch=50, loss=0.0080364
[INFO] epoch=55, loss=0.0078690
[INFO] epoch=60, loss=0.0077102
[INFO] epoch=65, loss=0.0075593
[INFO] epoch=70, loss=0.0074153
[INFO] epoch=75, loss=0.0072779
[INFO] epoch=80, loss=0.0071465
[INFO] epoch=85, loss=0.0070207
[INFO] epoch=90, loss=0.0069001
[INFO] epoch=95, loss=0.0067843
[INFO] epoch=100, loss=0.0066731
[INFO] evaluating...
             precision    recall  f1-score   support

          0       1.00      1.00      1.00       250
          1       1.00      1.00      1.00       250

avg / total       1.00      1.00      1.00       500
```

SGD 示例使用学习率(0.1)和与普通梯度下降相同的历元数(100)。其结果可以在**图 1** 中看到。

调查第 100 个历元结束时的实际损耗值，您会注意到 SGD 获得的损耗比普通梯度下降(0 *)低近两个数量级* 。 006 vs 0 *。*分别为 447)。这种差异是由于每个时期的多个权重更新，给了我们的模型更多的机会从对权重矩阵的更新中学习。这种影响在大型数据集上更加明显，例如 ImageNet，我们有数百万个训练样本，参数的小规模增量更新可以产生低损失(但不一定是最佳)的解决方案。****