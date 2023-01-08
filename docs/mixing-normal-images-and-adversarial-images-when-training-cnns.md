# 训练细胞神经网络时混合正常图像和敌对图像

> 原文：<https://pyimagesearch.com/2021/03/15/mixing-normal-images-and-adversarial-images-when-training-cnns/>

在本教程中，您将学习如何在训练过程中生成(1)正常图像和(2)敌对图像的图像批次。这样做可以提高您的模型的概括能力和防御敌对攻击的能力。

上周，我们学习了一个简单的方法来抵御敌对攻击。这种方法是一个简单的三步过程:

1.  在你的原始训练集上训练 CNN
2.  从测试集(或等价的抵制集)中生成对立的例子
3.  微调 CNN 上的对抗性例子

这种方法效果很好，但可以通过改变训练过程来大大改善。

我们可以改变批量生成过程本身，而不是在一组对立的例子上微调网络。

当我们训练神经网络时，我们是在批量数据中进行的。每批都是训练数据的子集，通常大小为 2 的幂(8、16、32、64、128 等。).对于每一批，我们执行网络的正向传递，计算损耗，执行反向传播，然后更新网络的权重。这是基本上任何神经网络的标准训练协议。

我们可以通过以下方式修改此标准培训程序，以纳入对抗性示例:

1.  初始化我们的神经网络
2.  选择总共 *N* 个训练实例
3.  使用该模型和类似于 [FGSM](https://pyimagesearch.com/2021/03/01/adversarial-attacks-with-fgsm-fast-gradient-sign-method/) 的方法来生成总共 *N* 个对抗示例
4.  将两组组合，形成一批尺寸 *Nx2*
5.  在对立示例和原始训练样本上训练模型

这种方法的好处是模型可以自我学习。

每次批量更新后，模型都有两个因素的改进。首先，模型已经在训练数据中理想地学习了更多的辨别模式。第二，模型已经学会了抵御模型本身产生的对抗性例子。

在整个训练过程中(几十到几百个时期，有几万到几十万次批量更新)，模型自然地学会了防御敌对攻击。

这种方法比基本的微调方法更复杂，但好处远远大于坏处。

**学习如何在训练中混合正常图像和对立图像，以提高模型的鲁棒性，*继续阅读。***

## **训练 CNN 时混合正常图像和敌对图像**

在本教程的第一部分，我们将学习如何在训练过程中混合正常图像和敌对图像。

从那里，我们将配置我们的开发环境，然后检查我们的项目目录结构。

我们今天要实现几个 Python 脚本，包括:

1.  我们的 CNN 架构
2.  对抗图像生成器
3.  一个数据生成器，它(1)对训练数据点进行采样，以及(2)动态生成对立的示例
4.  一份将所有内容整合在一起的培训脚本

我们将通过在混合敌对图像生成过程中训练我们的模型来结束本教程，然后讨论结果。

我们开始吧！

### 如何在训练中混合正常图像和敌对图像？

将训练图像与敌对图像混合在一起可以得到最好的视觉解释。我们从神经网络架构和训练集开始:

*普通*训练过程的工作方式是从训练集中抽取一批数据，然后训练模型:

**然而，我们想要合并对抗性训练，所以我们需要一个*单独的过程*，它使用模型来生成对抗性的例子:**

现在，在我们的训练过程中，我们对训练集进行采样并生成对立的例子，然后训练网络:

训练过程稍微复杂一点，因为我们从训练集*和*中取样，动态生成对立的例子。不过，好处是该模型可以:

1.  从*原始*训练集中学习模式
2.  从*对立的例子*中学习句型

由于该模型现在已经在对立的例子上被训练，所以当呈现对立的图像时，它将更健壮并且概括得更好。

### **配置您的开发环境**

这篇关于防御恶意图像攻击的教程使用了 Keras 和 TensorFlow。如果你打算遵循这个教程，我建议你花时间配置你的深度学习开发环境。

您可以利用这两个指南中的任何一个在您的系统上安装 TensorFlow 和 Keras:

*   [*如何在 Ubuntu 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   [*如何在 macOS 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)

这两个教程都将帮助您在一个方便的 Python 虚拟环境中，用这篇博文所需的所有软件来配置您的系统。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   准备好在您的 Windows、macOS 或 Linux 系统上运行代码*了吗*？

那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-plus/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

让我们从回顾我们的项目目录结构开始本教程。

使用本指南的 ***“下载”*** 部分检索源代码。然后，您将看到以下目录:

```py
$ tree . --dirsfirst
.
├── pyimagesearch
│   ├── __init__.py
│   ├── datagen.py
│   ├── fgsm.py
│   └── simplecnn.py
└── train_mixed_adversarial_defense.py

1 directory, 5 files
```

我们的目录结构基本上与上周关于用 Keras 和 TensorFlow 防御对抗性图像攻击的教程 [*相同。主要区别在于:*](https://pyimagesearch.com/2021/03/08/defending-against-adversarial-image-attacks-with-keras-and-tensorflow/)

1.  我们在我们的`datagen.py`文件中添加了一个新函数，来处理同时混合训练图像和动态生成的敌对图像*。*
2.  我们的驱动程序训练脚本`train_mixed_adversarial_defense.py`，有一些额外的附加功能来处理混合训练。

**如果你还没有，我强烈建议你阅读本系列的前两个教程:**

1.  [](https://pyimagesearch.com/2021/03/01/adversarial-attacks-with-fgsm-fast-gradient-sign-method/)
2.  *[*用 Keras 和 TensorFlow 防御对抗性图像攻击*](https://pyimagesearch.com/2021/03/08/defending-against-adversarial-image-attacks-with-keras-and-tensorflow/)*

 ***他们被认为是** ***必读*** **才继续！**

### **我们的基本 CNN**

我们的 CNN 架构可以在项目结构的`simplecnn.py`文件中找到。在我们的[快速梯度符号方法教程](https://pyimagesearch.com/2021/03/01/adversarial-attacks-with-fgsm-fast-gradient-sign-method/)中，我已经详细回顾了这个模型定义，所以我将把对代码的完整解释留给那个指南。

也就是说，我在下面列出了`SimpleCNN`的完整实现供您查看:

```py
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
```

**第 2-8 行**导入我们需要的 Python 包。

然后我们可以创建`SimpleCNN`架构:

```py
class SimpleCNN:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# first CONV => RELU => BN layer set
		model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		# second CONV => RELU => BN layer set
		model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
```

这种架构的突出之处包括:

1.  **第一套`CONV => RELU => BN`图层。**`CONV`层学习总共 32 个 *3×3* 滤波器，用 *2×2* 步进卷积减少体积大小。
2.  **第二套`CONV => RELU => BN`图层。**同上，但这次`CONV`层学习 64 个滤镜。
3.  一组密集/完全连接的层。其输出是我们的 softmax 分类器，用于返回每个类别标签的概率。

### **使用 FGSM 生成对抗图像**

我们使用快速梯度符号方法(FGSM)来生成图像对手。我们已经在本系列的前面[中详细介绍了这个实现，所以你可以在那里查阅完整的代码回顾。](https://pyimagesearch.com/2021/03/01/adversarial-attacks-with-fgsm-fast-gradient-sign-method/)

也就是说，如果你打开项目目录结构中的`fgsm.py`文件，你会发现下面的代码:

```py
# import the necessary packages
from tensorflow.keras.losses import MSE
import tensorflow as tf

def generate_image_adversary(model, image, label, eps=2 / 255.0):
	# cast the image
	image = tf.cast(image, tf.float32)

	# record our gradients
	with tf.GradientTape() as tape:
		# explicitly indicate that our image should be tacked for
		# gradient updates
		tape.watch(image)

		# use our model to make predictions on the input image and
		# then compute the loss
		pred = model(image)
		loss = MSE(label, pred)

	# calculate the gradients of loss with respect to the image, then
	# compute the sign of the gradient
	gradient = tape.gradient(loss, image)
	signedGrad = tf.sign(gradient)

	# construct the image adversary
	adversary = (image + (signedGrad * eps)).numpy()

	# return the image adversary to the calling function
	return adversary
```

概括地说，这段代码是:

1.  接受一个我们想要“愚弄”做出错误预测的`model`
2.  获取`model`并使用它对输入`image`进行预测
3.  基于地面实况类`label`计算模型的`loss`
4.  计算损失相对于图像的梯度
5.  获取梯度的符号(或者`-1`、`0`、`1`)，然后使用带符号的梯度来创建图像对手

**最终结果将是一个输出图像，看起来*与原始图像*在视觉上*相同，但是 CNN 将会错误地分类。***

同样，你可以参考我们的 [FGSM 指南来详细查看代码](https://pyimagesearch.com/2021/03/01/adversarial-attacks-with-fgsm-fast-gradient-sign-method/)。

### **更新我们的数据生成器，在运行中混合正常图像和敌对图像**

在本节中，我们将实现两个功能:

1.  `generate_adversarial_batch`:使用我们的 FGSM 实现生成总共 *N 张*敌对图像。
2.  `generate_mixed_adverserial_batch`:生成一批 *N 张*图像，一半是正常图像，另一半是敌对图像。

我们在上周的教程 [*中实现了第一种方法，用 Keras 和 TensorFlow*](https://pyimagesearch.com/2021/03/08/defending-against-adversarial-image-attacks-with-keras-and-tensorflow/) 防御对抗性图像攻击。第二个函数是全新的，是本教程独有的。

让我们从数据批处理生成器开始吧。在我们的项目结构中打开`datagen.py`文件，并插入以下代码:

```py
# import the necessary packages
from .fgsm import generate_image_adversary
from sklearn.utils import shuffle
import numpy as np
```

2-4 号线处理我们所需的进口。

我们正在从我们的`fgsm`模块导入`generate_image_adversary`，这样我们就可以生成镜像对手。

导入`shuffle`函数以将图像和标签混洗在一起。

下面是我们的`generate_adversarial_batch`函数的定义，我们在上周实现了[:](https://pyimagesearch.com/2021/03/08/defending-against-adversarial-image-attacks-with-keras-and-tensorflow/)

```py
def generate_adversarial_batch(model, total, images, labels, dims,
	eps=0.01):
	# unpack the image dimensions into convenience variables
	(h, w, c) = dims

	# we're constructing a data generator here so we need to loop
	# indefinitely
	while True:
		# initialize our perturbed images and labels
		perturbImages = []
		perturbLabels = []

		# randomly sample indexes (without replacement) from the
		# input data
		idxs = np.random.choice(range(0, len(images)), size=total,
			replace=False)

		# loop over the indexes
		for i in idxs:
			# grab the current image and label
			image = images[i]
			label = labels[i]

			# generate an adversarial image
			adversary = generate_image_adversary(model,
				image.reshape(1, h, w, c), label, eps=eps)

			# update our perturbed images and labels lists
			perturbImages.append(adversary.reshape(h, w, c))
			perturbLabels.append(label)

		# yield the perturbed images and labels
		yield (np.array(perturbImages), np.array(perturbLabels))
```

由于我们在之前的帖子中详细讨论了该函数，所以我将推迟对该函数的完整讨论，但在高级别上，您可以看到该函数:

1.  从我们的输入`images`集(通常是我们的训练集或测试集)中随机抽取 *N 个*图像(`total`)
2.  然后，我们使用 FGSM 从我们随机采样的图像中生成对立的例子
3.  该函数通过将对立的图像和标签返回给调用函数来完成

**这里最重要的一点是`generate_adversarial_batch`方法只返回*和*对立的图像。**

然而，这个帖子的目标是**包含*正常图像*和敌对图像**的混合训练。因此，我们需要实现第二个助手函数:

```py
def generate_mixed_adverserial_batch(model, total, images, labels,
	dims, eps=0.01, split=0.5):
	# unpack the image dimensions into convenience variables
	(h, w, c) = dims

	# compute the total number of training images to keep along with
	# the number of adversarial images to generate
	totalNormal = int(total * split)
	totalAdv = int(total * (1 - split))
```

顾名思义，`generate_mixed_adverserial_batch`创造了一个混合了*和*正常形象和敌对形象的作品。

此方法有几个参数，包括:

1.  我们正在训练和使用 CNN 来生成敌对的图像
2.  `total`:我们每批需要的图像总数
3.  `images`:输入图像集(通常是我们的训练或测试分割)
4.  `labels`:对应的属于`images`的分类标签
5.  `dims`:输入图像的空间尺寸
6.  `eps`:用于生成对抗图像的小ε值
7.  `split`:正常图像与对抗图像的百分比；在这里，我们做的是五五分成

从那里，我们将`dims`元组分解成我们的高度、宽度和通道数(**第 43 行**)。

我们还基于我们的`split` ( **第 47 行和第 48 行**)导出训练图像的总数和对抗图像的数量。

现在让我们深入了解数据生成器本身:

```py
	# we're constructing a data generator so we need to loop
	# indefinitely
	while True:
		# randomly sample indexes (without replacement) from the
		# input data and then use those indexes to sample our normal
		# images and labels
		idxs = np.random.choice(range(0, len(images)),
			size=totalNormal, replace=False)
		mixedImages = images[idxs]
		mixedLabels = labels[idxs]

		# again, randomly sample indexes from the input data, this
		# time to construct our adversarial images
		idxs = np.random.choice(range(0, len(images)), size=totalAdv,
			replace=False)
```

**第 52 行**开始一个无限循环，一直持续到训练过程完成。

然后，我们从我们的输入集中随机抽取总共`totalNormal`幅图像(**第 56-59 行**)。

接下来，**行 63 和 64** 执行第二轮随机采样，这次是为了对抗图像生成。

我们现在可以循环这些`idxs`:

```py
		# loop over the indexes
		for i in idxs:
			# grab the current image and label, then use that data to
			# generate the adversarial example
			image = images[i]
			label = labels[i]
			adversary = generate_image_adversary(model,
				image.reshape(1, h, w, c), label, eps=eps)

			# update the mixed images and labels lists
			mixedImages = np.vstack([mixedImages, adversary])
			mixedLabels = np.vstack([mixedLabels, label])

		# shuffle the images and labels together
		(mixedImages, mixedLabels) = shuffle(mixedImages, mixedLabels)

		# yield the mixed images and labels to the calling function
		yield (mixedImages, mixedLabels)
```

对于每个图像索引`i`，我们:

1.  抓取当前`image`和`label` ( **线 70 和 71** )
2.  通过 FGSM 生成对抗图像(**行 72 和 73** )
3.  用我们的敌对形象和标签更新我们的`mixedImages`和`mixedLabels`列表(**第 76 行和第 77 行**

**80 线**联合洗牌我们的`mixedImages`和`mixedLabels`。我们执行这种混洗操作是因为正常图像和敌对图像被顺序地加在一起，这意味着正常图像出现在列表的*前面*，而敌对图像出现在列表的*后面*。洗牌确保我们的数据样本在整个批次中随机分布。

然后，将混洗后的一批数据交给调用函数。

### **创建我们的混合图像和对抗图像训练脚本**

实现了所有的助手函数后，我们就可以创建我们的训练脚本了。

打开项目结构中的`train_mixed_adverserial_defense.py`文件，让我们开始工作:

```py
# import the necessary packages
from pyimagesearch.simplecnn import SimpleCNN
from pyimagesearch.datagen import generate_mixed_adverserial_batch
from pyimagesearch.datagen import generate_adversarial_batch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
```

**第 2-8 行**导入我们需要的 Python 包。请注意我们的定制实施，包括:

1.  我们将要训练的 CNN 架构。
2.  `generate_mixed_adverserial_batch`:批量生成*正常图像和对抗图像*
3.  *`generate_adversarial_batch`:生成一批*专属*的敌对图像*

 *我们将在 MNIST 数据集上训练`SimpleCNN`，现在让我们加载并预处理它:

```py
# load MNIST dataset and scale the pixel values to the range [0, 1]
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0

# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

# one-hot encode our labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)
```

**第 12 行**从磁盘加载 MNIST 数字数据集。然后，我们通过以下方式对其进行预处理:

1.  将像素强度从范围*【0，255】*缩放到*【0，1】*
2.  向数据添加批次维度
3.  一键编码标签

我们现在可以编译我们的模型了:

```py
# initialize our optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the simple CNN on MNIST
print("[INFO] training network...")
model.fit(trainX, trainY,
	validation_data=(testX, testY),
	batch_size=64,
	epochs=20,
	verbose=1)
```

**第 26-29 行**编译我们的模型。然后，我们根据我们的`trainX`和`trainY`数据在**的第 33-37 行**上训练它。

培训后，下一步是评估模型:

```py
# make predictions on the testing set for the model trained on
# non-adversarial images
(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] normal testing images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))

# generate a set of adversarial from our test set (so we can evaluate
# our model performance *before* and *after* mixed adversarial
# training)
print("[INFO] generating adversarial examples with FGSM...\n")
(advX, advY) = next(generate_adversarial_batch(model, len(testX),
	testX, testY, (28, 28, 1), eps=0.1))

# re-evaluate the model on the adversarial images
(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] adversarial testing images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
```

**第 41-43 行**根据我们的测试数据评估模型。

然后我们在第 49 行和第 50 行的**上生成一组*专有的*敌对图像。**

然后，我们的模型被重新评估，这一次是在对抗性图像上(**第 53-55 行**)。

正如我们将在下一节中看到的，我们的模型将在原始测试数据上表现良好，但在敌对图像上准确性将*骤降*。

**为了帮助防御敌对攻击，我们可以在由*正常图像和敌对示例组成的数据批次上微调模型。***

下面的代码块完成了这项任务:

```py
# lower the learning rate and re-compile the model (such that we can
# fine-tune it on the mixed batches of normal images and dynamically
# generated adversarial images)
print("[INFO] re-compiling model...")
opt = Adam(lr=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# initialize our data generator to create data batches containing
# a mix of both *normal* images and *adversarial* images
print("[INFO] creating mixed data generator...")
dataGen = generate_mixed_adverserial_batch(model, 64,
	trainX, trainY, (28, 28, 1), eps=0.1, split=0.5)

# fine-tune our CNN on the adversarial images
print("[INFO] fine-tuning network on dynamic mixed data...")
model.fit(
	dataGen,
	steps_per_epoch=len(trainX) // 64,
	epochs=10,
	verbose=1)
```

第 61-63 行降低我们的学习速度，然后重新编译我们的模型。

从那里，我们创建我们的数据生成器(**第 68 行和第 69 行**)。在这里，我们告诉我们的数据生成器使用我们的`model`来生成数据批次(每批中有`64`个数据点)，从我们的训练数据中采样，对于正常图像和敌对图像，以 50/50 的比例进行分割。

**通过我们的`dataGen`到`model.fit`允许我们的 CNN 在这些混合批次上接受训练。**

让我们进行最后一轮评估:

```py
# now that our model is fine-tuned we should evaluate it on the test
# set (i.e., non-adversarial) again to see if performance has degraded
(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print("")
print("[INFO] normal testing images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))

# do a final evaluation of the model on the adversarial images
(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] adversarial images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))
```

**第 81-84 行**在对混合批次进行微调后，在我们的原始测试集*上评估我们的 CNN。*

然后，我们再一次评估 CNN 对我们的原始敌对图像(**第 87-89 行**)。

理想情况下，我们将看到的是我们的正常图像和敌对图像之间的平衡准确性，从而使我们的模型更加健壮，能够抵御敌对攻击。

### **在正常图像和敌对图像上训练我们的 CNN**

我们现在准备在正常训练图像和动态生成的敌对图像上训练我们的 CNN。

通过访问本教程的 ***【下载】*** 部分来检索源代码。

从那里，打开一个终端并执行以下命令:

```py
$ time python train_mixed_adversarial_defense.py
[INFO] loading MNIST dataset...
[INFO] compiling model...
[INFO] training network...
Epoch 1/20
938/938 [==============================] - 6s 6ms/step - loss: 0.2043 - accuracy: 0.9377 - val_loss: 0.0615 - val_accuracy: 0.9805
Epoch 2/20
938/938 [==============================] - 6s 6ms/step - loss: 0.0782 - accuracy: 0.9764 - val_loss: 0.0470 - val_accuracy: 0.9846
Epoch 3/20
938/938 [==============================] - 6s 6ms/step - loss: 0.0597 - accuracy: 0.9810 - val_loss: 0.0493 - val_accuracy: 0.9828
...
Epoch 18/20
938/938 [==============================] - 6s 6ms/step - loss: 0.0102 - accuracy: 0.9965 - val_loss: 0.0478 - val_accuracy: 0.9889
Epoch 19/20
938/938 [==============================] - 6s 6ms/step - loss: 0.0116 - accuracy: 0.9961 - val_loss: 0.0359 - val_accuracy: 0.9915
Epoch 20/20
938/938 [==============================] - 6s 6ms/step - loss: 0.0105 - accuracy: 0.9967 - val_loss: 0.0477 - val_accuracy: 0.9891
[INFO] normal testing images:
[INFO] loss: 0.0477, acc: 0.9891
```

上面，你可以看到在*正常* MNIST 训练集上训练我们 CNN 的输出。这里，我们在训练集上获得了 **99.67%** 的准确率，在测试集上获得了 **98.91%** 的准确率。

现在，让我们看看当我们用[快速梯度符号方法](https://pyimagesearch.com/2021/03/01/adversarial-attacks-with-fgsm-fast-gradient-sign-method/)生成一组对立图像时会发生什么:

```py
[INFO] generating adversarial examples with FGSM...

[INFO] adversarial testing images:
[INFO] loss: 14.0658, acc: 0.0188
```

我们的准确度从 98.91%的准确度下降到 1.88%的准确度。显然，我们的模型不能很好地处理对立的例子。

我们现在要做的是降低学习率，重新编译模型，然后使用数据生成器进行微调，数据生成器包括*和*原始训练图像*和*动态生成的对抗图像:

```py
[INFO] re-compiling model...
[INFO] creating mixed data generator...
[INFO] fine-tuning network on dynamic mixed data...
Epoch 1/10
937/937 [==============================] - 162s 173ms/step - loss: 1.5721 - accuracy: 0.7653
Epoch 2/10
937/937 [==============================] - 146s 156ms/step - loss: 0.4189 - accuracy: 0.8875
Epoch 3/10
937/937 [==============================] - 146s 156ms/step - loss: 0.2861 - accuracy: 0.9154
...
Epoch 8/10
937/937 [==============================] - 146s 155ms/step - loss: 0.1423 - accuracy: 0.9541
Epoch 9/10
937/937 [==============================] - 145s 155ms/step - loss: 0.1307 - accuracy: 0.9580
Epoch 10/10
937/937 [==============================] - 146s 155ms/step - loss: 0.1234 - accuracy: 0.9604
```

使用这种方法，我们获得了 96.04%的准确率。

当我们将其应用于最终测试图像时，我们会得出以下结论:

```py
[INFO] normal testing images *after* fine-tuning:
[INFO] loss: 0.0315, acc: 0.9906

[INFO] adversarial images *after* fine-tuning:
[INFO] loss: 0.1190, acc: 0.9641

real    27m17.243s
user    43m1.057s
sys     14m43.389s
```

在使用动态数据生成过程对我们的模型进行微调之后，我们在原始测试图像上获得了 **99.06%** 的准确性(高于上周[方法](https://pyimagesearch.com/2021/03/08/defending-against-adversarial-image-attacks-with-keras-and-tensorflow/)的 98.44%)。

我们的对抗性图像准确率为 **96.41%，**比上周的 99%有所下降，但这在这种情况下是有意义的——请记住，我们不是在*上微调模型，而是像上周一样在*上微调对抗性示例。相反，我们允许模型“反复愚弄自己”,并从它产生的对立例子中学习。

通过仅在*对立示例(没有任何原始训练样本)的*上再次微调，可以潜在地获得进一步的准确性。尽管如此，我还是把它作为一个练习留给读者去探索。

### **演职员表和参考资料**

FGSM 和数据生成器的实现受到了塞巴斯蒂安·蒂勒关于对抗性攻击和防御的优秀文章的启发。非常感谢 Sebastian 分享他的知识。

## **总结**

在本教程中，您学习了如何修改 CNN 的训练过程来生成图像批，包括:

1.  正常训练图像
2.  CNN 产生的对立例子

这种方法不同于我们上周学过的方法，在那种方法中，我们简单地对一个对立图像的样本微调 CNN。

今天的方法的好处是，CNN 可以通过以下方式更好地抵御敌对的例子:

1.  从原始训练示例中学习模式
2.  从即时生成的敌对图像中学习模式

由于模型可以在每批训练的*期间生成自己的对抗实例，它可以不断地自我学习。*

总的来说，我认为你会发现这种方法在训练你自己的模型抵御对抗性攻击时更有益。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****