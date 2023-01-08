# 使用 Keras 和 TensorFlow 进行有针对性的对抗性攻击

> 原文：<https://pyimagesearch.com/2020/10/26/targeted-adversarial-attacks-with-keras-and-tensorflow/>

在本教程中，您将学习如何使用 Keras、TensorFlow 和深度学习来执行有针对性的对抗性攻击和构建有针对性的对抗性图像。

上周的教程涵盖了 [*无针对性的*对抗性学习](https://pyimagesearch.com/2020/10/19/adversarial-images-and-attacks-with-keras-and-tensorflow/)，它的过程是:

*   **步骤#1:** 接受输入图像并使用预训练的 CNN 确定其类别标签
*   **步骤#2:** 构造噪声向量，该噪声向量在被添加到输入图像时故意干扰结果图像，以这样的方式:
    *   **步骤#2a:** 输入图像被预先训练的 CNN*错误分类*
    *   步骤#2b: 然而，对于人眼来说，被扰乱的图像与原始图像是*无法区分的*

有了无目标的对抗性学习，我们就不在乎输入图像的*新*类标签是什么了，假设它被 CNN 错误分类了。例如，下图显示了我们已经应用了对抗性学习来获取被正确分类为*“猪”*的输入，并对其进行扰动，使得图像现在被错误地分类为*“袋熊”:*

在无目标的对抗性学习中，我们无法控制最终的、被扰乱的类别标签是什么。但是如果我们想要控制呢？这可能吗？

**绝对是——为了控制扰动图像的类别标签，我们需要应用*有针对性的*对抗学习。**

本教程的剩余部分将向你展示如何应用有针对性的对抗性学习。

**要学习如何用 Keras 和 TensorFlow 进行有针对性的对抗性学习，*继续阅读。***

## **使用 Keras 和 TensorFlow 进行有针对性的对抗性攻击**

在本教程的第一部分，我们将简要讨论什么是对抗性攻击和对抗性图像。然后我会解释*有针对性的*对抗性攻击和*无针对性的*攻击之间的区别。

接下来，我们将回顾我们的项目目录结构，从那里，我们将实现一个 Python 脚本，它将使用 Keras 和 TensorFlow 应用有针对性的对抗性学习。

我们将讨论我们的结果来结束本教程。

### 什么是对抗性攻击？什么是形象对手？

如果你是对抗性攻击的新手，以前没有听说过对抗性图像，我建议你在阅读本指南之前，先阅读我的博客文章， *[对抗性图像和攻击与 Keras 和 TensorFlow](https://pyimagesearch.com/2020/10/19/adversarial-images-and-attacks-with-keras-and-tensorflow/)* 。

**要点是，敌对的图像是*故意构建的*来*愚弄*预先训练好的模型。**

例如，如果一个预先训练的 CNN 能够正确地分类一个输入图像，一个敌对的攻击试图采取完全相同的图像并且:

1.  扰动它，使图像现在*被错误分类…*
2.  …然而，新的、被打乱的图像看起来与原始图像(至少在人眼看来)一模一样

理解对抗性攻击是如何工作的以及对抗性图像是如何构建的非常重要——了解这一点将有助于你训练你的 CNN，使它们能够抵御这些类型的对抗性攻击(这是我将在未来的教程中涉及的主题)。

### **有针对性的*对抗性攻击与无针对性的*攻击有何不同？****

 ****上面的图 3** 直观地显示了无目标对抗性攻击和有目标攻击之间的区别。

当构建无目标的对抗性攻击时，我们无法控制扰动图像的最终输出类别标签将是什么— **我们唯一的目标是迫使模型对输入图像进行*错误的*分类。**

**图 3 *(上)*** 是*无针对性*对抗性攻击的一个例子。在这里，我们输入一个*“猪”*的图像——对抗性攻击算法然后扰乱输入图像，使得它被错误地分类为*“袋熊”*，但是同样，我们没有指定目标类标签应该是什么(坦率地说，无目标算法不关心，只要输入图像现在被错误地分类)。

另一方面，有针对性的对抗性攻击让我们能够更好地控制被扰乱图像的最终预测标签。

**图 3 *(下)*** 是*针对性*对抗性攻击的一个例子。我们再次输入一个*“pig”*的图像，但是我们也提供了扰动图像的目标类标签(在本例中是一只*“Lakeland terrier”*，一种狗)。

然后，我们的目标对抗性攻击算法能够干扰猪的输入图像，使得它现在被错误地分类为莱克兰梗。

在本教程的剩余部分，你将学习如何进行这样有针对性的对抗性攻击。

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

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！获得我们的 PyImageSearch 教程 **Jupyter 笔记本，**，它运行在**谷歌的 Colab 生态系统*上，在你的浏览器* — *中不需要安装。***

### **项目结构**

在我们开始用 Keras 和 TensorFlow 实现有针对性的对抗性攻击之前，我们首先需要回顾一下我们的项目目录结构。

首先使用本教程的 ***【下载】*** 部分下载源代码和示例图像。从那里，检查目录结构:

```py
$ tree --dirsfirst
.
├── pyimagesearch
│   ├── __init__.py
│   ├── imagenet_class_index.json
│   └── utils.py
├── adversarial.png
├── generate_targeted_adversary.py
├── pig.jpg
└── predict_normal.py

1 directory, 7 files
```

我们的目录结构与上周关于 *[敌对图像以及用 Keras 和 TensorFlow](https://pyimagesearch.com/2020/10/19/adversarial-images-and-attacks-with-keras-and-tensorflow/)* 攻击的指南相同。

`pyimagesearch`模块包含`utils.py`，这是一个帮助实用程序，它加载并解析位于`imagenet_class_index.json`中的 ImageNet 类标签索引。我们在上周的教程中介绍了这个助手函数，今天*不会*在这里介绍它的实现——我建议你阅读我的[以前的教程](https://pyimagesearch.com/2020/10/19/adversarial-images-and-attacks-with-keras-and-tensorflow/)来了解更多细节。

让我们开始实施有针对性的对抗性攻击吧！

### **步骤#1:使用我们预先训练的 CNN** 获得原始类别标签预测

在我们能够执行有针对性的对抗性攻击之前，我们必须首先确定来自预训练 CNN 的预测类别标签是什么。

出于本教程的目的，我们将使用 ResNet 架构，在 ImageNet 数据集上进行了预训练。

**对于任何给定的输入图像，我们需要:**

1.  加载图像
2.  预处理它
3.  通过 ResNet 传递
4.  获得类别标签预测
5.  确定类标签的整数索引

一旦我们有了*预测的*类标签的整数索引，以及*目标*类标签，我们希望网络预测图像是什么；然后我们就可以进行有针对性的对抗性攻击。

让我们从获得猪的以下图像的类别标签预测和索引开始:

为了完成这项任务，我们将在项目目录结构中使用`predict_normal.py`脚本。这个脚本在上周的教程中已经讨论过了，所以我们今天不会在这里讨论它——如果你有兴趣看这个脚本背后的代码，可以参考我之前的教程。

综上所述，首先使用本教程的 ***“下载”*** 部分下载源代码和示例图像。

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

这里你可以看到我们输入的`pig.jpg`图像被归类为*“猪”*，置信度为 **99.97%。**

在我们的下一部分，你将学习如何干扰这张图片，使它被错误地归类为*“莱克兰梗”*(一种狗)。

**但是现在，请注意我们终端输出的第 5 行，它显示了预测标签*【猪】*的 ImageNet 类标签索引是**`341`**——我们将在下一节中需要这个值。**

### **步骤#2:使用 Keras 和 TensorFlow 实施有针对性的对抗性攻击**

我们现在准备实施有针对性的对抗性攻击，并使用 Keras 和 TensorFlow 构建有针对性的对抗性图像。

打开项目目录结构中的`generate_targeted_adversary.py`文件，并插入以下代码:

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

我们通过使用 TensorFlow 的`clip_by_value`方法来完成这个裁剪。我们提供`tensor`作为输入，然后将`-eps`设置为最小限幅值限制，将`eps`设置为正限幅值限制。

当我们构造扰动向量时，将使用该函数，确保我们构造的噪声向量落在可容忍的限度内，并且最重要的是，**不会显著影响输出对抗图像的视觉质量。**

请记住，对立的图像应该与它们的原始输入*相同*(对人眼而言)——通过在可容忍的限度内剪裁张量值，我们能够强制执行这一要求。

接下来，我们需要定义`generate_targeted_adversaries`函数，这是这个 Python 脚本的主要部分:

```py
def generate_targeted_adversaries(model, baseImage, delta, classIdx,
	target, steps=500):
	# iterate over the number of steps
	for step in range(0, steps):
		# record our gradients
		with tf.GradientTape() as tape:
			# explicitly indicate that our perturbation vector should
			# be tracked for gradient updates
			tape.watch(delta)

			# add our perturbation vector to the base image and
			# preprocess the resulting image
			adversary = preprocess_input(baseImage + delta)
```

接下来是应用有针对性的对抗性攻击的梯度下降部分:

```py
			# run this newly constructed image tensor through our
			# model and calculate the loss with respect to the
			# both the *original* class label and the *target*
			# class label
			predictions = model(adversary, training=False)
			originalLoss = -sccLoss(tf.convert_to_tensor([classIdx]),
				predictions)
			targetLoss = sccLoss(tf.convert_to_tensor([target]),
				predictions)
			totalLoss = originalLoss + targetLoss

			# check to see if we are logging the loss value, and if
			# so, display it to our terminal
			if step % 20 == 0:
				print("step: {}, loss: {}...".format(step,
					totalLoss.numpy()))

		# calculate the gradients of loss with respect to the
		# perturbation vector
		gradients = tape.gradient(totalLoss, delta)

		# update the weights, clip the perturbation vector, and
		# update its value
		optimizer.apply_gradients([(gradients, delta)])
		delta.assign_add(clip_eps(delta, eps=EPS))

	# return the perturbation vector
	return delta
```

现在已经定义了所有的函数，我们可以开始解析命令行参数了:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to original input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output adversarial image")
ap.add_argument("-c", "--class-idx", type=int, required=True,
	help="ImageNet class ID of the predicted label")
ap.add_argument("-t", "--target-class-idx", type=int, required=True,
	help="ImageNet class ID of the target adversarial label")
args = vars(ap.parse_args())
```

让我们继续进行一些初始化:

```py
EPS = 2 / 255.0
LR = 5e-3

# load image from disk and preprocess it
print("[INFO] loading image...")
image = cv2.imread(args["input"])
image = preprocess_image(image)
```

**第 82 行**定义了我们的ε(`EPS`)值，用于在构建对抗图像时裁剪张量。`2 / 255.0`的`EPS`值是敌对出版物和教程中使用的标准值。

然后我们在第 84 行**定义我们的学习率。**值`LR = 5e-3`是通过经验调整获得的— *在构建您自己的目标对抗性攻击时，您可能需要更新该值。*

**第 88 行和第 89 行**加载我们的输入`image`，然后使用 ResNet 的预处理帮助函数对其进行预处理。

接下来，我们需要加载 ResNet 模型并初始化我们的损失函数:

```py
# load the pre-trained ResNet50 model for running inference
print("[INFO] loading pre-trained ResNet50 model...")
model = ResNet50(weights="imagenet")

# initialize optimizer and loss function
optimizer = Adam(learning_rate=LR)
sccLoss = SparseCategoricalCrossentropy()

# create a tensor based off the input image and initialize the
# perturbation vector (we will update this vector via training)
baseImage = tf.constant(image, dtype=tf.float32)
delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)
```

如果你想了解更多关于这些变量和初始化的细节，请参考上周的教程，我会在那里详细介绍它们。

构建了所有变量后，我们现在可以应用有针对性的对抗性攻击:

```py
# generate the perturbation vector to create an adversarial example
print("[INFO] generating perturbation...")
deltaUpdated = generate_targeted_adversaries(model, baseImage, delta,
	args["class_idx"], args["target_class_idx"])

# create the adversarial example, swap color channels, and save the
# output image to disk
print("[INFO] creating targeted adversarial example...")
adverImage = (baseImage + deltaUpdated).numpy().squeeze()
adverImage = np.clip(adverImage, 0, 255).astype("uint8")
adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
cv2.imwrite(args["output"], adverImage)
```

然后我们剪裁任何像素值，使所有像素都在范围*【0，255】*内，接着将图像转换为无符号的 8 位整数(这样 OpenCV 就可以对图像进行操作)。

最后的`adverImage`然后被写入磁盘。

问题依然存在——我们是否愚弄了我们最初的 ResNet 模型，做出了不正确的预测？

让我们在下面代码块中回答这个问题:

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

# write the top-most predicted label on the image along with the
# confidence score
text = "{}: {:.2f}%".format(label, confidence)
cv2.putText(adverImage, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
	(0, 255, 0), 2)

# show the output image
cv2.imshow("Output", adverImage)
cv2.waitKey(0)
```

**第 120 行**通过首先构建对抗图像，然后使用 ResNet 的预处理实用程序对其进行预处理，来构建一个`preprocessedImage`。

一旦图像经过预处理，我们就使用我们的`model`对其进行预测。这些预测随后被解码并获得排名第一的预测——类别标签和相应的概率随后被显示到我们的终端上(**第 121-126 行**)。

最后，我们用预测的标签和置信度注释输出图像，然后将输出图像显示到屏幕上。

要审查的代码太多了！花点时间祝贺自己成功实施了有针对性的对抗性攻击。在下一部分，我们将看到我们努力工作的成果。

### **步骤#3:有针对性的对抗性攻击结果**

我们现在准备进行有针对性的对抗性攻击！确保您已经使用了本教程的 ***【下载】*** 部分来下载源代码和示例图像。

接下来，打开`imagenet_class_index.json`文件，确定我们想要“欺骗”网络进行预测的 ImageNet 类标签的整数索引——类标签索引文件的前几行如下所示:

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
```

滚动浏览文件，直到找到要使用的类别标签。

在这种情况下，我选择了索引`189`，它对应于一只*“莱克兰梗”*(一种狗):

```py
...
"189": [
    "n02095570",
    "Lakeland_terrier"
  ],
...
```

从那里，您可以打开一个终端并执行以下命令:

```py
$ python generate_targeted_adversary.py --input pig.jpg --output adversarial.png --class-idx 341 --target-class-idx 189
[INFO] loading image...
[INFO] loading pre-trained ResNet50 model...
[INFO] generating perturbation...
step: 0, loss: 16.111093521118164...
step: 20, loss: 15.760734558105469...
step: 40, loss: 10.959839820861816...
step: 60, loss: 7.728139877319336...
step: 80, loss: 5.327273368835449...
step: 100, loss: 3.629972219467163...
step: 120, loss: 2.3259339332580566...
step: 140, loss: 1.259613037109375...
step: 160, loss: 0.30303144454956055...
step: 180, loss: -0.48499584197998047...
step: 200, loss: -1.158257007598877...
step: 220, loss: -1.759873867034912...
step: 240, loss: -2.321563720703125...
step: 260, loss: -2.910153865814209...
step: 280, loss: -3.470625877380371...
step: 300, loss: -4.021825313568115...
step: 320, loss: -4.589465141296387...
step: 340, loss: -5.136003017425537...
step: 360, loss: -5.707150459289551...
step: 380, loss: -6.300693511962891...
step: 400, loss: -7.014866828918457...
step: 420, loss: -7.820181369781494...
step: 440, loss: -8.733556747436523...
step: 460, loss: -9.780607223510742...
step: 480, loss: -10.977422714233398...
[INFO] creating targeted adversarial example...
[INFO] running inference on the adversarial example...
[INFO] label: Lakeland_terrier confidence: 54.82%
```

在左边的*，*你可以看到我们的原始输入图像，它被正确地归类为*“猪”。*

然后，我们应用了一个有针对性的对抗性攻击*(右)*，扰乱了输入图像，以至于它被错误地分类为一只具有 **68.15%置信度的莱克兰梗(一种狗)！**

作为参考，莱克兰梗看起来一点也不像猪:

在上周关于*无目标*对抗性攻击的教程中，我们看到*无法控制*扰动图像的最终预测类别标签；然而，通过应用*有针对性的*对抗性攻击，我们能够*控制*最终预测什么标签。

## **总结**

在本教程中，您学习了如何使用 Keras、TensorFlow 和深度学习来执行有针对性的对抗性学习。

当应用无目标的对抗学习时，我们的目标是干扰输入图像，使得:

1.  受干扰的图像被我们预先训练的 CNN 错误分类
2.  然而，对于人眼来说，被扰乱的图像与原始图像是相同的

无目标的对抗学习的问题是，我们无法控制扰动的输出类别标签。例如，如果我们有一个*“猪”*的输入图像，并且我们想要扰乱该图像，使得它被错误分类，我们不能控制什么是*新的*类标签。

另一方面，有针对性的对抗性学习允许我们控制新的类别标签将是什么，这非常容易实现，只需要更新我们的损失函数计算。

到目前为止，我们已经讲述了如何*构建*对抗性攻击，但是如果我们想要*防御*攻击呢？这可能吗？

的确是这样——我将在以后的博客文章中讨论防御对抗性攻击。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****