# 使用 Keras 和 TensorFlow 的敌对图像和攻击

> 原文：<https://pyimagesearch.com/2020/10/19/adversarial-images-and-attacks-with-keras-and-tensorflow/>

在本教程中，您将学习如何使用基于图像的对抗性攻击来打破深度学习模型。我们将使用 Keras 和 TensorFlow 深度学习库来实施我们的对抗性攻击。

想象一下二十年后。路上几乎所有的汽车和卡车都被自动驾驶汽车取代，由人工智能、深度学习和计算机视觉提供动力——每一次转弯、车道切换、加速和刹车都由深度神经网络提供动力。

现在，想象你在高速公路上。你正坐在“驾驶座”上(如果汽车在驾驶，这真的是“驾驶座”吗？)而你的配偶坐在副驾驶座上，你的孩子坐在后排。

向前看，你看到你的车正在行驶的车道上贴着一张大贴纸。看起来够无辜的。这只是涂鸦艺术家班克斯广受欢迎的 *[拿着气球的女孩](https://en.wikipedia.org/wiki/Girl_with_Balloon)* 作品的大幅印刷。一些高中生可能只是把它放在那里作为一个奇怪的挑战/恶作剧的一部分。

**一瞬间之后，你的汽车做出反应，猛地刹车，然后变换车道，就好像贴在路上的大型艺术印刷品是一个人、一只动物或另一辆车**。你被猛地拉了一下，感觉到了鞭打。你的配偶尖叫着，而后座上你孩子的麦片向前飞去，撞在挡风玻璃上，在中控台上弹来弹去。

你和你的家人都很安全… *但是情况可能会更糟。*

发生了什么事？为什么你的自动驾驶汽车会有这种反应？是你的汽车运行的代码/软件中的某种奇怪的“错误”吗？

**答案是，驱动车辆“视觉”组件的深度神经网络刚刚看到了一幅*敌对图像。***

对立的图像有:

1.  有像素的图像*被故意扰乱*以迷惑和欺骗模型…
2.  …但同时，看起来对人类无害和无辜。

这些图像导致深度神经网络*故意*做出不正确的预测。对立的图像受到干扰，以至于模型无法正确地对它们进行分类。

事实上，对于人类来说，从一个因敌对攻击而受到视觉干扰的图像中识别出一个正常的图像可能是不可能的，从本质上来说，这两个图像在人眼看来是完全相同的。

虽然不是精确(或正确)的比较，但我喜欢在图像隐写术的背景下解释对抗性攻击。使用隐写术算法，我们可以在图像*中嵌入数据(例如明文消息),而不会扭曲图像本身的外观。该图像可以被无害地传输给接收者，然后接收者可以从图像中提取隐藏的信息。*

类似地，对抗性攻击在输入图像中嵌入一条消息——但是，**对抗性攻击在输入图像中嵌入一个*噪声向量*,而不是用于人类消费的明文消息。**这个噪声向量是*特意构建的*用来愚弄和混淆深度学习模型。

但是对抗性攻击是如何工作的呢？我们如何抵御它们？

这篇教程，以及本系列的其他文章，将会涉及到同样的问题。

**要学习如何使用 Keras/TensorFlow 用对抗性攻击和图像打破深度学习模型，*继续阅读。***

## **使用 Keras 和 TensorFlow 的敌对图像和攻击**

在本教程的第一部分，我们将讨论什么是对抗性攻击，以及它们如何影响深度学习模型。

从这里开始，我们将实现三个独立的 Python 脚本:

1.  第一个将是一个助手工具，用于从 ImageNet 数据集加载和解析类标签。
2.  我们的下一个 Python 脚本将使用 ResNet 执行基本的影像分类，在 ImageNet 数据集上进行预训练(从而演示“标准”影像分类)。
3.  最终的 Python 脚本将执行对抗性攻击，并构建一个对抗性图像，故意混淆我们的 ResNet 模型，即使这两个图像在人眼看来是相同的。

我们开始吧！

### **什么是对抗性图像和对抗性攻击？它们如何影响深度学习模型？**

2014 年，Goodfellow 等人发表了一篇题为 *[的论文，解释并利用了对立的例子](https://arxiv.org/abs/1412.6572)* ，该论文显示了深度神经网络的一个有趣的属性——**有可能*故意*扰动一个输入图像，以至于神经网络将其错误分类。**这种类型的干扰被称为**对抗性攻击。**

对抗性攻击的经典例子可以在上面的**图 2** 中看到。在左边的*，*是我们的输入图像，我们的神经网络以 57.7%的置信度将其正确分类为“熊猫”。

在中间的*，*我们有一个噪声向量，对于人眼来说，它看起来是随机的。然而，它离随机的*很远*。

相反，噪声向量中的像素*等于相对于输入图像的成本函数的梯度元素的符号* (Goodfellow 等人)。

然后，我们将这个噪声向量添加到输入图像中，这产生了图 2 中**的输出*(右)*。**对我们来说，这个图像看起来与输入相同；然而，**我们的神经网络现在以 99.7%的置信度将图像分类为“长臂猿”**(一种小猿，类似于猴子)。

令人毛骨悚然，对吧？

### **对抗性攻击和图像的简史**

对抗性机器学习不是一个新领域，这些攻击也不是针对深度神经网络的。2006 年，Barreno 等人发表了一篇名为 *[的论文，机器学习能安全吗？](https://people.eecs.berkeley.edu/~adj/publications/paper-files/asiaccs06.pdf)* 本文讨论了对抗性攻击，包括针对它们提出的防御措施。

早在 2006 年，最先进的机器学习模型包括支持向量机(SVMs)和随机森林(RFs)——已经证明这两种类型的模型都容易受到敌对攻击。

随着深度神经网络从 2012 年开始流行起来，人们*希望*这些高度非线性的模型不那么容易受到攻击；然而，古德费勒等人(以及其他人)粉碎了这些希望。

事实证明，深度神经网络容易受到对抗性攻击，就像它们的前辈一样。

关于对抗性攻击历史的更多信息，我推荐阅读比格吉奥和花小蕾 2017 年的优秀论文， *[狂野模式:对抗性机器学习兴起后的十年。](https://arxiv.org/abs/1712.03141)*

### 为什么对抗性攻击和图像是一个问题？

本教程顶部的例子概述了为什么敌对攻击会对健康、生命和财产造成巨大损失。

后果*不太严重的例子*可能是一群黑客发现谷歌在 Gmail 中使用特定模型过滤垃圾邮件，或者脸书在 NSFW 过滤器中使用给定模型自动检测色情内容。

如果这些黑客想向 Gmail 用户发送大量绕过 Gmail 垃圾邮件过滤器的电子邮件，或者绕过 NSFW 过滤器向脸书上传大量色情内容，理论上他们可以这么做。

这些都是后果较小的对抗性攻击的例子。

在后果*更高*的场景中，对抗性攻击可能包括黑客-恐怖分子识别出一个特定的深度神经网络正被用于世界上几乎所有的自动驾驶汽车(想象一下，如果特斯拉垄断了市场，并且是*唯一的*自动驾驶汽车制造商)。

敌对的图像可能会被战略性地放置在道路和高速公路上，造成大规模的连环相撞、财产损失，甚至车内乘客的伤亡。

对抗性攻击的限制只受您的想象力、您对给定模型的了解以及您对模型本身的访问权限的限制。

### 我们能抵御敌对攻击吗？

好消息是我们可以帮助减少对抗性攻击的影响(但不一定完全消除它们)。

这个主题不会在今天的教程中讨论，但是会在以后的 PyImageSearch 教程中讨论。

### 配置您的开发环境

要针对本教程配置您的系统，我建议您遵循以下任一教程:

*   [*如何在 Ubuntu 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   [*如何在 macOS 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)

这两个教程都将帮助您在一个方便的 Python 虚拟环境中，用这篇博文所需的所有软件来配置您的系统。

**这样说来，你是:**

*   时间紧迫？
*   在你雇主被行政锁定的笔记本电脑上学习？
*   想要跳过与包管理器、bash/ZSH 概要文件和虚拟环境的争论吗？
*   准备好运行代码*了吗*(并尽情体验它)？

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！在您的浏览器中访问运行在**谷歌的 Colab 生态系统*上的 PyImageSearch 教程 **Jupyter 笔记本**——无需安装！***

### **项目结构**

首先使用本教程的 ***【下载】*** 部分下载源代码和示例图像。从那里，让我们检查我们的项目目录结构。

```py
$ tree --dirsfirst
.
├── pyimagesearch
│   ├── __init__.py
│   ├── imagenet_class_index.json
│   └── utils.py
├── adversarial.png
├── generate_basic_adversary.py
├── pig.jpg
└── predict_normal.py

1 directory, 7 files
```

准备好用 Keras 和 TensorFlow 实施你的第一次对抗性攻击了吗？

让我们开始吧。

### **我们的 ImageNet 类标签/索引辅助工具**

在我们可以执行正常的图像分类或对受到恶意攻击的图像进行分类之前，我们首先需要创建一个 Python 辅助函数，用于加载和解析 ImageNet 数据集的类标签。

我们已经在项目目录结构的`pyimagesearch`模块的`imagenet_class_index.json`文件中提供了一个 JSON 文件，其中包含 ImageNet 类标签索引、标识符和人类可读的字符串。

我在下面包含了这个 JSON 文件的前几行:

```py
{
  "0": [
    "n01440764",
    "tench"
  ],
  "1": [
    "n01443537",
    "goldfish"
  ],
  "2": [
    "n01484850",
    "great_white_shark"
  ],
  "3": [
    "n01491361",
    "tiger_shark"
  ],
...
"106": [
    "n01883070",
    "wombat"
  ],
...
```

这里你可以看到这个文件是一个字典。字典的关键字是整数类标签索引，而值是由以下内容组成的二元组:

1.  标签的 ImageNet 唯一标识符
2.  人类可读的类标签

我们的目标是实现一个 Python 函数，它将通过以下方式解析 JSON 文件:

1.  接受输入类标签
2.  返回相应标签的整数类标签索引

```py
# import necessary packages
import json
import os

def get_class_idx(label):
	# build the path to the ImageNet class label mappings file
	labelPath = os.path.join(os.path.dirname(__file__),
		"imagenet_class_index.json")
```

现在让我们加载 JSON 文件的内容:

```py
	# open the ImageNet class mappings file and load the mappings as
	# a dictionary with the human-readable class label as the key and
	# the integer index as the value
	with open(labelPath) as f:
		imageNetClasses = {labels[1]: int(idx) for (idx, labels) in
			json.load(f).items()}

	# check to see if the input class label has a corresponding
	# integer index value, and if so return it; otherwise return
	# a None-type value
	return imageNetClasses.get(label, None)
```

*   标签的整数索引(如果它存在于字典中)
*   如果`label` *在`imageNetClasses`中不存在*，则返回`None`

### **正常图像分类*无*使用 Keras 和 TensorFlow 的对抗性攻击**

实现了我们的 ImageNet 类标签/索引助手函数之后，让我们首先创建一个图像分类脚本，该脚本执行*基本分类*，并且没有*恶意攻击。*

该脚本将证明我们的 ResNet 模型正在按照我们的预期运行(即做出正确的预测)。在本教程的后面，你会发现如何构建一个敌对的形象，使其混淆 ResNet。

让我们从基本的图像分类脚本开始——打开项目目录结构中的`predict_normal.py`文件，并插入以下代码:

```py
# import necessary packages
from pyimagesearch.utils import get_class_idx
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import argparse
import imutils
import cv2
```

我们在第 2-9 行的**中导入我们需要的 Python 包。如果你以前使用过 Keras、TensorFlow 和 OpenCV，这些对你来说都是相当标准的。**

也就是说，如果你是 Keras 和 TensorFlow 的新手，我强烈建议你阅读我的 *[Keras 教程:如何入门 Keras、深度学习和 Python](https://pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/)* 指南。此外，你可能想阅读我的书 *[用 Python 进行计算机视觉的深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)* ，以更深入地了解如何训练你自己的定制神经网络。

```py
def preprocess_image(image):
	# swap color channels, preprocess the image, and add in a batch
	# dimension
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = preprocess_input(image)
	image = cv2.resize(image, (224, 224))
	image = np.expand_dims(image, axis=0)

	# return the preprocessed image
	return image
```

接下来，让我们解析我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
```

```py
# load image from disk and make a clone for annotation
print("[INFO] loading image...")
image = cv2.imread(args["image"])
output = image.copy()

# preprocess the input image
output = imutils.resize(output, width=400)
preprocessedImage = preprocess_image(image)
```

```py
# load the pre-trained ResNet50 model
print("[INFO] loading pre-trained ResNet50 model...")
model = ResNet50(weights="imagenet")

# make predictions on the input image and parse the top-3 predictions
print("[INFO] making predictions...")
predictions = model.predict(preprocessedImage)
predictions = decode_predictions(predictions, top=3)[0]
```

```py
# loop over the top three predictions
for (i, (imagenetID, label, prob)) in enumerate(predictions):
	# print the ImageNet class label ID of the top prediction to our
	# terminal (we'll need this label for our next script which will
	# perform the actual adversarial attack)
	if i == 0:
		print("[INFO] {} => {}".format(label, get_class_idx(label)))

	# display the prediction to our screen
	print("[INFO] {}. {}: {:.2f}%".format(i + 1, label, prob * 100))
```

```py
# draw the top-most predicted label on the image along with the
# confidence score
text = "{}: {:.2f}%".format(predictions[0][1],
	predictions[0][2] * 100)
cv2.putText(output, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
	(0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
```

在点击 OpenCV 打开的窗口并按下一个键之前,`output`图像一直显示在我们的终端上。

### **非对抗性图像分类结果**

我们现在准备好使用 ResNet 执行基本的图像分类(即，没有恶意攻击)。

首先使用本教程的 ***【下载】*** 部分下载源代码和示例图像。

从那里，打开一个终端并执行以下命令:

```py
$ python predict_normal.py --image pig.jpg
[INFO] loading image...
[INFO] loading pre-trained ResNet50 model...
[INFO] making predictions...
[INFO] hog => 341
[INFO] 1\. hog: 99.97%
[INFO] 2\. wild_boar: 0.03%
[INFO] 3\. piggy_bank: 0.00%
```

### **用 Keras 和 TensorFlow 实现对抗性图像和攻击**

我们现在将学习如何用 Keras 和 TensorFlow 实现对抗性攻击。

在我们的项目目录结构中打开`generate_basic_adversary.py`文件，并插入以下代码:

```py
# import necessary packages
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import numpy as np
import argparse
import cv2
```

我们从在第 2-10 行导入我们需要的 Python 包开始。你会注意到我们再次使用`ResNet50`架构及其相应的`preprocess_input`函数(用于预处理/缩放输入图像)和`decode_predictions`实用程序来解码输出预测并显示人类可读的 ImageNet 标签。

`SparseCategoricalCrossentropy`计算标签和预测之间的分类交叉熵损失。通过使用分类交叉熵的*稀疏*版本实现，我们*不需要*像使用 scikit-learn 的`LabelBinarizer`或 Keras/TensorFlow 的`to_categorical`实用程序那样显式地一次性编码我们的类标签。

就像我们的`predict_normal.py`脚本中有一个`preprocess_image`实用程序一样，我们也需要一个用于这个脚本的实用程序:

```py
def preprocess_image(image):
	# swap color channels, resize the input image, and add a batch
	# dimension
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	image = np.expand_dims(image, axis=0)

	# return the preprocessed image
	return image
```

```py
def clip_eps(tensor, eps):
	# clip the values of the tensor to a given range and return it
	return tf.clip_by_value(tensor, clip_value_min=-eps,
		clip_value_max=eps)
```

```py
def generate_adversaries(model, baseImage, delta, classIdx, steps=50):
	# iterate over the number of steps
	for step in range(0, steps):
		# record our gradients
		with tf.GradientTape() as tape:
			# explicitly indicate that our perturbation vector should
			# be tracked for gradient updates
			tape.watch(delta)
```

```py
			# add our perturbation vector to the base image and
			# preprocess the resulting image
			adversary = preprocess_input(baseImage + delta)

			# run this newly constructed image tensor through our
			# model and calculate the loss with respect to the
			# *original* class index
			predictions = model(adversary, training=False)
			loss = -sccLoss(tf.convert_to_tensor([classIdx]),
				predictions)

			# check to see if we are logging the loss value, and if
			# so, display it to our terminal
			if step % 5 == 0:
				print("step: {}, loss: {}...".format(step,
					loss.numpy()))

		# calculate the gradients of loss with respect to the
		# perturbation vector
		gradients = tape.gradient(loss, delta)

		# update the weights, clip the perturbation vector, and
		# update its value
		optimizer.apply_gradients([(gradients, delta)])
		delta.assign_add(clip_eps(delta, eps=EPS))

	# return the perturbation vector
	return delta
```

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to original input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output adversarial image")
ap.add_argument("-c", "--class-idx", type=int, required=True,
	help="ImageNet class ID of the predicted label")
args = vars(ap.parse_args())
```

```py
# define the epsilon and learning rate constants
EPS = 2 / 255.0
LR = 0.1

# load the input image from disk and preprocess it
print("[INFO] loading image...")
image = cv2.imread(args["input"])
image = preprocess_image(image)
```

**第 76 行**定义了我们的ε(`EPS`)值，用于在构建对抗图像时裁剪张量。`2 / 255.0`的`EPS`值是敌对出版物和教程中使用的标准值(如果您有兴趣了解关于这个“默认”值的更多信息，下面指南中的[也很有帮助)。](https://adversarial-ml-tutorial.org/)

然后我们在第 77 行**定义我们的学习率。**通过经验调整获得了一个值`LR = 0.1`—*在构建您自己的对立图像时，您可能需要更新这个值。*

```py
# load the pre-trained ResNet50 model for running inference
print("[INFO] loading pre-trained ResNet50 model...")
model = ResNet50(weights="imagenet")

# initialize optimizer and loss function
optimizer = Adam(learning_rate=LR)
sccLoss = SparseCategoricalCrossentropy()
```

```py
# create a tensor based off the input image and initialize the
# perturbation vector (we will update this vector via training)
baseImage = tf.constant(image, dtype=tf.float32)
delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)

# generate the perturbation vector to create an adversarial example
print("[INFO] generating perturbation...")
deltaUpdated = generate_adversaries(model, baseImage, delta,
	args["class_idx"])

# create the adversarial example, swap color channels, and save the
# output image to disk
print("[INFO] creating adversarial example...")
adverImage = (baseImage + deltaUpdated).numpy().squeeze()
adverImage = np.clip(adverImage, 0, 255).astype("uint8")
adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
cv2.imwrite(args["output"], adverImage)
```

之后，我们通过以下方式对最终的对抗图像进行后处理:

1.  剪裁超出范围*【0，255】*的任何值
2.  将图像转换为无符号的 8 位整数(以便 OpenCV 现在可以对图像进行操作)
3.  交换从 RGB 到 BGR 的颜色通道排序

经过上述预处理步骤后，我们将输出的对抗图像写入磁盘。

**真正的问题是，*我们新构建的对抗性图像能骗过我们的 ResNet 模型吗？***

下一个代码块将解决这个问题:

```py
# run inference with this adversarial example, parse the results,
# and display the top-1 predicted result
print("[INFO] running inference on the adversarial example...")
preprocessedImage = preprocess_input(baseImage + deltaUpdated)
predictions = model.predict(preprocessedImage)
predictions = decode_predictions(predictions, top=3)[0]
label = predictions[0][1]
confidence = predictions[0][2] * 100
print("[INFO] label: {} confidence: {:.2f}%".format(label,
	confidence))

# draw the top-most predicted label on the adversarial image along
# with the confidence score
text = "{}: {:.2f}%".format(label, confidence)
cv2.putText(adverImage, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
	(0, 255, 0), 2)

# show the output image
cv2.imshow("Output", adverImage)
cv2.waitKey(0)
```

我们再次在第 113 行**上构建我们的对抗图像，方法是将 delta 噪声向量添加到我们的原始输入图像中，但这次我们调用 ResNet 的`preprocess_input`实用程序。**

产生的预处理图像通过 ResNet，之后我们获取前 3 个预测并解码它们(**行 114 和 115** )。

然后，我们用 top-1 预测获取标签和相应的概率/置信度，并将这些值显示到我们的终端(**行 116-119** )。

最后一步是在我们输出的敌对图像上绘制顶部预测，并将其显示到我们的屏幕上。

### **对抗性图像和攻击的结果**

准备好观看对抗性攻击了吗？

确保您使用了本教程的 ***“下载”*** 部分来下载源代码和示例图像。

从那里，您可以打开一个终端并执行以下命令:

```py
$ python generate_basic_adversary.py --input pig.jpg --output adversarial.png --class-idx 341
[INFO] loading image...
[INFO] loading pre-trained ResNet50 model...
[INFO] generating perturbation...
step: 0, loss: -0.0004124982515349984...
step: 5, loss: -0.0010656398953869939...
step: 10, loss: -0.005332294851541519...
step: 15, loss: -0.06327803432941437...
step: 20, loss: -0.7707189321517944...
step: 25, loss: -3.4659299850463867...
step: 30, loss: -7.515471935272217...
step: 35, loss: -13.503922462463379...
step: 40, loss: -16.118188858032227...
step: 45, loss: -16.118192672729492...
[INFO] creating adversarial example...
[INFO] running inference on the adversarial example...
[INFO] label: wombat confidence: 100.00%
```

在*左边*是原始的猪图像，而在*右边*我们有输出的对抗图像，它被错误地分类为*“袋熊”。*

正如你所看到的，这两幅图像之间没有明显的差异，我们的人眼可以看到这两幅图像之间的差异，但对雷斯内特来说，它们是完全不同的 T2。

这很好，但是我们显然无法控制对抗性图像中的最终类标签。这就提出了一个问题:

**可以控制输入图像的最终输出类标签是什么吗？**答案是*是的*——我将在下周的教程中讨论这个问题。

我最后要说的是，如果你让你的想象力占了上风，你很容易被对抗性的图像和对抗性的攻击吓到。但是正如我们将在后面的 PyImageSearch 教程中看到的，我们实际上可以*防御*这些类型的攻击。稍后会详细介绍。

### **学分**

如果没有 Goodfellow、Szegedy 和许多其他深度学习研究人员的研究，本教程是不可能的。

此外，我想指出，今天教程中使用的实现是受 [TensorFlow 的快速梯度符号方法的官方实现的启发。](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)我强烈建议你看一下他们的例子，这个例子很好地解释了本教程更多的理论和数学方面。

## **总结**

在本教程中，您学习了对抗性攻击，它们如何工作，以及它们对越来越依赖人工智能和深度神经网络的世界构成的威胁。

然后，我们使用 Keras 和 TensorFlow 深度学习库实现了一个基本的对抗性攻击算法。

使用对抗性攻击，我们可以*故意*干扰输入图像，使得:

1.  输入图像被*错误分类*
2.  然而，对于人眼来说，被打乱的图像看起来与原始图像完全相同

然而，使用今天在这里应用的方法，我们绝对无法控制图像的最终类别标签是什么——我们所做的只是创建和嵌入一个噪声向量，这导致深度神经网络对图像进行错误分类。

但是如果我们能够控制最终的目标类标签是什么呢？例如，是否有可能拍摄一张“狗”的图像，并构建一个对抗性攻击，使卷积神经网络认为该图像是一只“猫”？

答案是肯定的——我们将在下周的教程中讨论完全相同的主题。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***