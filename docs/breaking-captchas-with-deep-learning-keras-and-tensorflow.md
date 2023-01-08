# 利用深度学习、Keras 和 TensorFlow 破解验证码

> 原文：<https://pyimagesearch.com/2021/07/14/breaking-captchas-with-deep-learning-keras-and-tensorflow/>

在过去的**、**中，我们已经处理了为我们预先编译和标记的数据集——*，但是如果我们想要着手创建我们自己的**定制数据集**，然后在其上训练 CNN，那会怎么样呢？*在本教程中，我将展示一个*完整的*深度学习案例研究，为您提供以下示例:

1.  下载一组图像。
2.  为你的训练图像添加标签和注释。
3.  在您的自定义数据集上训练 CNN。
4.  评估和测试训练好的 CNN。

我们将要下载的图像数据集是一组验证码图像，用于防止机器人自动注册或登录到给定的网站(或者更糟糕的是，试图强行进入某人的帐户)。

一旦我们下载了一组验证码图片，我们需要手动标记验证码中的每个数字。我们将会发现，*获取*和*标记*一个数据集可以成功一半(如果不是更多的话)。根据您需要多少数据、获取数据的难易程度以及您是否需要标记数据(即，为图像分配一个地面实况标签)，这可能是一个在时间和/或资金方面都很昂贵的过程(如果您付钱给其他人来标记数据)。

因此，只要有可能，我们就尝试使用传统的计算机视觉技术来加快贴标过程。如果我们使用图像处理软件，如 Photoshop 或 GIMP，手动提取验证码图像中的数字来创建我们的训练集，可能需要我们连续工作*天*才能完成任务。

然而，通过应用一些基本的计算机视觉技术，我们可以在不到一个小时的时间内下载并标记我们的训练集。这是我鼓励深度学习从业者也投资于他们的计算机视觉教育的众多原因之一。

**要学习如何用深度学习、Keras、TensorFlow 破解验证码，** ***继续阅读。***

## **利用深度学习、Keras 和 TensorFlow 破解验证码**

我还想提一下，现实世界中的数据集不像基准数据集，如 MNIST、CIFAR-10 和 ImageNet，它们对图像进行了整齐的标记和组织，我们的目标只是在数据上训练一个模型并对其进行评估。这些基准数据集可能具有挑战性，但在现实世界中，*斗争往往是获得(标记的)数据本身* —在许多情况下，标记的数据比通过在数据集上训练网络获得的深度学习模型更有价值*。*

 *例如，如果你正在运营一家公司，负责为美国政府创建一个定制的自动车牌识别(ALPR)系统，你可能会投资*年*来建立一个强大的大规模数据集，同时评估各种识别车牌的深度学习方法。积累如此庞大的标签数据集将会给你带来超越其他公司的竞争优势——在这种情况下，*数据本身*比最终产品更有价值。

你的公司更有可能被收购，仅仅是因为你对这个巨大的、有标签的数据集拥有独家权利。建立一个令人惊叹的深度学习模型来识别车牌只会增加你公司的价值，但同样，*标记的数据*获取和复制成本高昂，所以如果你拥有难以(如果不是不可能)复制的数据集的关键，请不要搞错:你公司的主要资产是数据，而不是深度学习。

让我们看看如何获得图像数据集，对它们进行标记，然后应用深度学习来破解验证码系统。

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

## **用 CNN 破解验证码**

以下是如何考虑打破验证码。记住**负责任的披露**的概念——当涉及计算机安全时，你应该*总是*这样做。

当我们创建一个 Python 脚本来*自动*下载一组我们将用于训练和评估的图像时，这个过程就开始了。

下载完我们的图像后，我们需要使用一点计算机视觉来帮助我们标记图像，使这个过程*比在 GIMP 或 Photoshop 等图片软件中简单地裁剪和标记更容易*和*更快*。一旦我们标记了我们的数据，我们将训练 LeNet 架构-正如我们将发现的那样，我们能够在不到 15 个时期内破解验证码系统并获得 100%的准确性。

### **责任披露说明**

住在美国东北部/中西部，没有电子通行证很难在主要高速公路上行驶。E-ZPass 是一种电子收费系统，用于许多桥梁、州际公路和隧道。旅行者只需购买一个 E-ZPass 应答器，将其放在汽车的挡风玻璃上，就可以在不停车的情况下快速通过收费站，因为他们的 E-ZPass 帐户附带的信用卡会收取任何通行费。

E-ZPass 让过路费变成了一个“愉快”得多的过程(如果有这种东西的话)。而不是在需要进行实际交易的地方没完没了地排队等候(例如，把钱交给收银员，收到找你的钱，拿到报销的打印收据，等等)。)，你可以直接在快车道上飞驰而过，不需要停下来——这在旅行中节省了大量时间，也减少了很多麻烦(尽管你仍然需要支付过路费)。

我花了很多时间在马里兰州和康涅狄格州之间旅行，这是美国 I-95 走廊沿线的两个州。I-95 走廊，尤其是在新泽西州，包含了过多的收费站，所以 E-ZPass 通行证对我来说是一个显而易见的决定。大约一年前，我的 E-ZPass 帐户附带的信用卡到期了，我需要更新它。我去 E-ZPass 纽约网站(我购买 E-ZPass 的州)登录并更新我的信用卡，但我突然停了下来(**图 2** )。

你能发现这个系统的缺陷吗？他们的“验证码”只不过是普通白色背景上的四个数字，这是一个重大的安全风险——即使是具有基本计算机视觉或深度学习经验的人也可以开发一个软件来破解这个系统。

这就是 ***责任披露*** 概念的由来。负责任的披露是一个计算机安全术语，用于描述如何披露漏洞。检测到威胁后，你没有立即将它发布到互联网上让每个人都看到*，而是尝试首先联系利益相关者，以确保他们知道存在问题。然后，风险承担者可以尝试修补软件并解决漏洞。*

简单地忽略漏洞和隐藏问题是一种错误的安全措施，应该避免。在理想的情况下，漏洞在公开披露之前*就被解决了。*

然而，当利益相关者不承认这个问题或者没有在合理的时间内解决这个问题时，就会产生一个道德难题——你会隐藏这个问题，假装它不存在吗？或者你公开它，把更多的注意力放在问题上，以便更快地解决问题？负责任的披露声明你首先把问题带给利益相关者(*负责任的*)——如果问题没有解决，那么你需要披露问题(*披露*)。

为了展示 E-ZPass NY 系统是如何面临风险的，我训练了一个深度学习模型来识别验证码中的数字。然后，我编写了第二个 Python 脚本来(1)自动填充我的登录凭证和(2)破解验证码，允许我的脚本访问我的帐户。

在这种情况下，我只是自动登录到我的帐户。使用这个“功能”，我可以自动更新信用卡，生成我的通行费报告，甚至在我的 E-ZPass 上添加一辆新车。但是一些邪恶的人可能会利用这种方法强行进入客户的账户。

在 我写这篇文章的一年前，我通过电子邮件、电话和推特联系了 E-ZPass 关于这个问题的 ***。他们确认收到了我的信息；然而，尽管进行了多次联系，但没有采取任何措施来解决这个问题。***

在本教程的剩余部分，我将讨论我们如何使用 E-ZPass 系统来获取 captcha 数据集，然后我们将在其上标记和训练深度学习模型。我将*而不是*分享自动登录账户的 Python 代码——这超出了负责任披露的范围，所以请不要向我索要该代码。

请记住，所有的知识都伴随着责任。这些知识*在任何情况下*都不应用于邪恶或不道德的目的。这个案例研究是作为一种方法存在的，以演示如何获取和标记自定义数据集，然后在其上训练深度学习模型。

**我必须声明，我*不对*如何使用该代码负责——将此作为学习的机会，而不是作恶的机会。**

### **验证码破解目录结构**

为了构建验证码破解系统，我们需要更新`pyimagesearch.utils`子模块并包含一个名为`captchahelper.py`的新文件:

```py
|--- pyimagesearch
|    |--- __init__.py
|    |--- datasets
|    |--- nn
|    |--- preprocessing
|    |--- utils
|    |    |--- __init__.py
|    |    |--- captchahelper.py
```

这个文件将存储一个名为`preprocess`的实用函数，以帮助我们在将数字输入到我们的深度神经网络之前处理它们。

我们还将在我们的 *pyimagesearch* 模块之外创建第二个目录，这个目录名为`captcha_breaker`，包含以下文件和子目录:

```py
|--- captcha_breaker
|    |--- dataset/
|    |--- downloads/
|    |--- output/
|    |--- annotate.py
|    |--- download_images.py
|    |--- test_model.py
|    |--- train_model.py
```

目录是我们所有的项目代码存储的地方，用来破解图像验证码。`dataset`目录是我们存储*标记的*数字的地方，我们将手工标记这些数字。我喜欢使用以下目录结构模板来组织我的数据集:

```py
root_directory/class_name/image_filename.jpg
```

因此，我们的`dataset`目录将具有以下结构:

```py
dataset/{1-9}/example.jpg
```

其中`dataset`是根目录，`{1-9}`是可能的数字名称，`example.jpg`是给定数字的一个例子。

`downloads`目录将存储原始验证码。从易通行网站下载的 jpg 文件。在`output`目录中，我们将存储我们训练过的 LeNet 架构。

顾名思义，`download_images.py`脚本将负责实际下载示例验证码并将它们保存到磁盘。一旦我们下载了一组验证码，我们将需要从每张图片中提取数字，并手工标记每个数字——这将由`annotate.py`完成。

`train_model.py`脚本将在标记的数字上训练 LeNet，而`test_model.py`将把 LeNet 应用于 captcha 图像本身。

### **自动下载示例图像**

构建验证码破解程序的第一步是下载验证码图片。

如果您将“https://www . e-zpassny . com/vector/jcaptcha . do”复制并粘贴到您的 web 浏览器中，并多次点击刷新，您会注意到这是一个动态程序，每次刷新都会生成一个新的验证码。因此，为了获得我们的示例 captcha 图像，我们需要请求这个图像几百次并保存结果图像。

要自动获取新的验证码图片并保存到磁盘，我们可以使用`download_images.py`:

```py
# import the necessary packages
import argparse
import requests
import time
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
ap.add_argument("-n", "--num-images", type=int,
	default=500, help="# of images to download")
args = vars(ap.parse_args())
```

**第 2-5 行**导入我们需要的 Python 包。`requests`库使得使用 HTTP 连接变得容易，并且在 Python 生态系统中被大量使用。如果您的系统上尚未安装`requests`，您可以通过以下方式安装:

```py
$ pip install requests
```

然后我们在第 8-13 行解析我们的命令行参数。我们需要一个命令行参数`--output`，它是存储原始验证码图像的输出目录的路径(我们稍后将手工标记图像中的每个数字)。

第二个可选开关`--num-images`，控制我们将要下载的验证码图片的数量。我们将这个值默认为`500`总图像数。由于每个验证码中有四个数字，这个`500`的值将会给我们`500×4 = 2,000`个数字，我们可以用它们来训练我们的网络。

我们的下一个代码块初始化我们将要下载的验证码图片的 URL，以及到目前为止生成的图片总数:

```py
# initialize the URL that contains the captcha images that we will
# be downloading along with the total number of images downloaded
# thus far
url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0
```

我们现在可以下载验证码图片了:

```py
# loop over the number of images to download
for i in range(0, args["num_images"]):
	try:
		# try to grab a new captcha image
		r = requests.get(url, timeout=60)

		# save the image to disk
		p = os.path.sep.join([args["output"], "{}.jpg".format(
			str(total).zfill(5))])
		f = open(p, "wb")
		f.write(r.content)
		f.close()

		# update the counter
		print("[INFO] downloaded: {}".format(p))
		total += 1

	# handle if any exceptions are thrown during the download process
	except:
		print("[INFO] error downloading image...")

	# insert a small sleep to be courteous to the server
	time.sleep(0.1)
```

在**的第 22 行，**我们开始循环我们想要下载的`--num-images`。在**行 25** 上请求下载图像。然后我们将图像保存到磁盘的第 28-32 行。如果下载图像时出现错误，我们在第 39 行和第 40 行的`try/except`块会捕捉到错误，并允许我们的脚本继续运行。最后，我们在**第 43 行**插入一个小 sleep 来礼貌地对待我们请求的 web 服务器。

您可以使用以下命令执行`download_images.py`:

```py
$ python download_images.py --output downloads
```

这个脚本需要一段时间来运行，因为我们已经(1)发出了一个下载图像的网络请求，并且(2)在每次下载后插入了一个 0.1 秒的暂停。

一旦程序执行完毕，您会看到您的`download`目录中充满了图像:

```py
$ ls -l downloads/*.jpg | wc -l
500
```

然而，这些只是*原始验证码图片* —我们需要*提取*和*标记*验证码中的每个数字来创建我们的训练集。为了实现这一点，我们将使用一点 OpenCV 和图像处理技术来使我们的生活更容易。

### **注释和创建我们的数据集**

那么，你如何标记和注释我们的验证码图片呢？我们是否打开 Photoshop 或 GIMP，使用“选择/选取框”工具复制出一个给定的数字，保存到磁盘，然后令人厌烦地重复*？如果我们这样做了，我们可能需要*天*的不间断工作来标记原始验证码图片中的每个数字。*

 *相反，更好的方法是使用 OpenCV 库中的基本图像处理技术来帮助我们。要了解如何更有效地标记数据集，请打开一个新文件，将其命名为`annotate.py`，并插入以下代码:

```py
# import the necessary packages
from imutils import paths
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True,
	help="path to output directory of annotations")
args = vars(ap.parse_args())
```

**第 2-6 行**导入我们需要的 Python 包，而**第 9-14 行**解析我们的命令行参数。该脚本需要两个参数:

*   `--input`:原始验证码图片的输入路径(即`downloads`目录)。
*   `--annot`:输出路径，我们将在这里存储标记的数字(即，`dataset`目录)。

我们的下一个代码块获取`--input`目录中所有图像的路径，并初始化一个名为`counts`的字典，该字典将存储给定数字(键)被标记的总次数(值):

```py
# grab the image paths then initialize the dictionary of character
# counts
imagePaths = list(paths.list_images(args["input"]))
counts = {}
```

实际的注释过程从下面开始:

```py
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# display an update to the user
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))

	try:
		# load the image and convert it to grayscale, then pad the
		# image to ensure digits caught on the border of the image
		# are retained
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8,
			cv2.BORDER_REPLICATE)
```

在第 22 行**，**上，我们开始循环每个单独的`imagePaths`。对于每幅图像，我们从磁盘中加载(**第 31 行**)，将其转换为灰度(**第 32 行**)，并在每个方向上用八个像素填充图像的边界(**第 33 行和第 34 行**)。**图 3** 显示了原始图像(*左*)和填充图像(*右*)之间的差异。

我们执行这个填充*，以防*我们的任何手指碰到图像的边界。如果手指*碰到了边界*，我们将无法从图像中提取它们。因此，为了防止这种情况，我们特意填充了输入图像，使给定的数字*不可能*接触到边界。

我们现在准备通过 Otsu 的阈值方法对输入图像进行二值化:

```py
  		# threshold the image to reveal the digits
		thresh = cv2.threshold(gray, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
```

这个函数调用自动为我们的图像设定阈值，这样我们的图像现在是*二进制*——黑色像素代表*背景*，而白色像素是我们的*前景*，如图**图 4** 所示。

对图像进行阈值处理是我们图像处理流程中的关键步骤，因为我们现在需要找到每个数字的*轮廓*:

```py
		# find contours in the image, keeping only the four largest
		# ones
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
```

**第 42 和 43 行**找到图像中每个手指的轮廓(即轮廓)。以防图像中有“噪声”,我们根据轮廓的面积对其进行分类，只保留四个最大的轮廓(即我们的手指本身)。

给定我们的轮廓，我们可以通过计算边界框来提取每个轮廓:

```py
  		# loop over the contours
		for c in cnts:
			# compute the bounding box for the contour then extract
			# the digit
			(x, y, w, h) = cv2.boundingRect(c)
			roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

			# display the character, making it large enough for us
			# to see, then wait for a keypress
			cv2.imshow("ROI", imutils.resize(roi, width=28))
			key = cv2.waitKey(0)
```

在**第 48 行，**我们在阈值图像中找到的每个轮廓上循环。我们调用`cv2.boundingRect`来计算包围盒( *x，y*)——数字区域的坐标。然后，在**线 52** 上从灰度图像中提取该感兴趣区域(ROI)。我在**图 5** 中加入了从原始验证码图片中提取的数字样本作为蒙太奇。

第 56 行将数字 ROI 显示到我们的屏幕上，调整到足够大以便我们容易看到。然后等待你键盘上的按键——但是明智地选择你的按键！您按下的键将被用作数字的*标签*。

要了解如何通过`cv2.waitKey`调用进行标记，请看下面的代码块:

```py
  			# if the '`' key is pressed, then ignore the character
			if key == ord("`"):
				print("[INFO] ignoring character")
				continue

			# grab the key that was pressed and construct the path
			# the output directory
			key = chr(key).upper()
			dirPath = os.path.sep.join([args["annot"], key])

			# if the output directory does not exist, create it
			if not os.path.exists(dirPath):
				os.makedirs(dirPath)
```

如果按下波浪号键“`”(波浪号)，我们将忽略该字符(**行 60 和 62** )。如果我们的脚本意外地在输入图像中检测到“噪声”(即除了数字之外的任何东西)，或者如果我们不确定数字是什么，就可能需要忽略一个字符。否则，我们假设按下的键是数字的*标签*(**第 66 行**)，并使用该键构建到我们的输出标签(**第 67 行**)的目录路径。

例如，如果我按下键盘上的`7`键，`dirPath`将会是:

```py
dataset/7
```

因此，所有包含数字“7”的图像将被存储在`dataset/7`子目录中。**第 70 行和第 71 行**检查`dirPath`目录是否不存在——如果不存在，我们创建它。

一旦我们确保`dirPath`正确存在，我们只需将示例数字写入文件:

```py
  			# write the labeled character to file
			count = counts.get(key, 1)
			p = os.path.sep.join([dirPath, "{}.png".format(
				str(count).zfill(6))])
			cv2.imwrite(p, roi)

			# increment the count for the current key
			counts[key] = count + 1
```

**第 74 行**为当前数字获取目前为止写入磁盘的示例总数。然后，我们使用`dirPath`构建示例数字的输出路径。在执行**第 75 行和第 76 行**之后，我们的输出路径`p`可能看起来像:

```py
datasets/7/000001.png
```

同样，请注意所有包含数字 7 的示例 ROI 将如何存储在`datasets/7`子目录中——这是在标记图像时组织数据集的一种简单、方便的方式。

如果在处理图像时出现错误，我们的最后一个代码块处理我们是否想从脚本中`control-c`退出*或*:

```py
  	# we are trying to control-c out of the script, so break from the
	# loop (you still need to press a key for the active window to
	# trigger this)
	except KeyboardInterrupt:
		print("[INFO] manually leaving script")
		break

	# an unknown error has occurred for this particular image
	except:
		print("[INFO] skipping image...")
```

如果我们希望`control-c`提前退出脚本，**第 85 行**会检测到这一点，并允许我们的 Python 程序优雅地退出。**第 90 行**捕捉*所有其他错误*并简单地忽略它们，允许我们继续贴标过程。

当标记一个数据集时，你最不希望发生的事情是由于图像编码问题导致的随机错误，从而导致你的整个程序崩溃。如果发生这种情况，您必须重新开始标记过程。显然，您可以构建额外的逻辑来检测您离开的位置。

要标记您从 E-ZPass NY 网站下载的图像，只需执行以下命令:

```py
$ python annotate.py --input downloads --annot dataset
```

在这里，你可以看到数字 7 显示在我的屏幕上的**图 6** 。

然后，我按下键盘上的`7`键来标记它，然后该数字被写入到`dataset/7`子目录中的文件中。

然后,`annotate.py`脚本前进到下一个数字让我标记。然后，您可以对原始 captcha 图像中的所有数字进行标记。您将很快意识到标注数据集可能是一个非常繁琐、耗时的过程。给所有 2000 个数字贴上标签应该花不到半个小时——但你可能会在头五分钟内感到厌倦。

记住，实际上*获得*你的标签数据集是成功的一半。从那里可以开始实际的工作。幸运的是，我已经为你标记了数字！如果您查看本教程附带下载中包含的`dataset`目录，您会发现整个数据集已经准备就绪:

```py
$ ls dataset/
1  2  3  4  5  6  7  8  9
$ ls -l dataset/1/*.png | wc -l
232
```

在这里，您可以看到九个子目录，每个子目录对应一个我们希望识别的数字。在每个子目录中，都有特定数字的示例图像。现在我们已经有了标记数据集，我们可以继续使用 LeNet 架构来训练我们的验证码破解程序。

### **数字预处理**

正如我们所知，我们的卷积神经网络需要在训练期间传递一个具有固定宽度和高度的图像。然而，我们标记的数字图像大小不一——有些比宽高，有些比高宽。因此，我们需要一种方法来填充输入图像并将其调整到固定的大小*，而不使*扭曲它们的纵横比。

我们可以通过在`captchahelper.py`中定义一个`preprocess`函数来调整图像的大小和填充图像，同时保持纵横比:

```py
# import the necessary packages
import imutils
import cv2

def preprocess(image, width, height):
	# grab the dimensions of the image, then initialize
	# the padding values
	(h, w) = image.shape[:2]

	# if the width is greater than the height then resize along
	# the width
	if w > h:
		image = imutils.resize(image, width=width)

	# otherwise, the height is greater than the width so resize
	# along the height
	else:
		image = imutils.resize(image, height=height)
```

我们的`preprocess`函数需要三个参数:

1.  `image`:我们要填充和调整大小的输入图像。
2.  `width`:图像的目标输出宽度。
3.  `height`:图像的目标输出高度。

在**第 12 和 13 行，**我们检查宽度是否大于高度，如果是，我们沿着较大的维度(宽度)调整图像的大小，否则，如果高度大于宽度，我们沿着高度调整大小(**第 17 和 18 行**)，这意味着宽度或高度(取决于输入图像的维度)是固定的。

然而，相反的维度比它应该的要小。要解决这个问题，我们可以沿着较短的维度“填充”图像，以获得固定的大小:

```py
  	# determine the padding values for the width and height to
	# obtain the target dimensions
	padW = int((width - image.shape[1]) / 2.0)
	padH = int((height - image.shape[0]) / 2.0)

	# pad the image then apply one more resizing to handle any
	# rounding issues
	image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
		cv2.BORDER_REPLICATE)
	image = cv2.resize(image, (width, height))

	# return the pre-processed image
	return image
```

**第 22 行和第 23 行**计算达到目标`width`和`height`所需的填充量。**第 27 行和第 28 行**将填充应用于图像。应用这个填充应该把我们的图像带到我们的目标`width`和`height`；然而，在某些情况下，我们可能会在给定的维度上偏离一个像素。解决这种差异的最简单的方法是简单地调用`cv2.resize` ( **Line 29** )来确保所有图像的宽度和高度都相同。

我们没有立即*调用函数顶部的`cv2.resize`的原因是，我们首先需要考虑输入图像的纵横比，并尝试首先正确填充它。如果我们不保持图像的长宽比，那么我们的数字将变得扭曲。*

 *### **训练验证码破解者**

既然已经定义了我们的`preprocess`函数，我们就可以继续在图像 captcha 数据集上训练 LeNet 了。打开`train_model.py`文件并插入以下代码:

```py
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from pyimagesearch.nn.conv import LeNet
from pyimagesearch.utils.captchahelper import preprocess
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
```

**第 2-14 行**导入我们需要的 Python 包。请注意，我们将使用 SGD 优化器和 LeNet 架构来训练数字模型。在通过我们的网络之前，我们还将对每个数字使用我们新定义的`preprocess`函数。

接下来，让我们回顾一下命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
args = vars(ap.parse_args())
```

`train_model.py`脚本需要两个命令行参数:

1.  `--dataset`:带标签的验证码数字的输入数据集的路径(即磁盘上的`dataset`目录)。
2.  在这里，我们提供了训练后保存我们的序列化 LeNet 权重的路径。

我们现在可以从磁盘加载数据和相应的标签:

```py
# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in paths.list_images(args["dataset"]):
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = preprocess(image, 28, 28)
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
```

在**的第 25 和 26 行，**我们分别初始化我们的`data`和`labels`列表。然后，我们在第 29 行的**标签中循环每个图像。对于数据集中的每个图像，我们从磁盘中加载它，将其转换为灰度，并对其进行预处理，使其宽度为 28 像素，高度为 28 像素(**第 31-35 行**)。然后图像被转换成 Keras 兼容的数组并添加到`data`列表中(**第 34 行和第 35 行**)。**

以下列格式组织数据集目录结构的主要好处之一是:

```py
root_directory/class_label/image_filename.jpg
```

您可以通过从文件名中抓取倒数第二个组件来轻松提取类标签(**第 39 行**)。例如，给定输入路径`dataset/7/000001.png`,`label`将是`7`，然后将其添加到`labels`列表中(**第 40 行**)。

我们的下一个代码块处理将原始像素亮度值归一化到范围`[0, 1]`，随后构建训练和测试分割，并对标签进行一键编码:

```py
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer().fit(trainY)
trainY = lb.transform(trainY)
testY = lb.transform(testY)
```

然后，我们可以初始化 LeNet 模型和 SGD 优化器:

```py
# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=9)
opt = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
```

我们的输入图像将有 28 像素的宽度，28 像素的高度，和一个单通道。我们正在识别的共有 9 个数字类(没有`0`类)。

给定初始化的模型和优化器，我们可以训练网络 15 个时期，评估它，并将其序列化到磁盘:

```py
# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY,  validation_data=(testX, testY),
	batch_size=32, epochs=15, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])
```

我们的最后一个代码块将处理绘制训练集和测试集随时间的准确性和损失:

```py
# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
```

要在我们的自定义 captcha 数据集上使用 SGD 优化器来训练 LeNet 架构，只需执行以下命令:

```py
$ python train_model.py --dataset dataset --model output/lenet.hdf5
[INFO] compiling model...
[INFO] training network...
Train on 1509 samples, validate on 503 samples
Epoch 1/15
0s - loss: 2.1606 - acc: 0.1895 - val_loss: 2.1553 - val_acc: 0.2266
Epoch 2/15
0s - loss: 2.0877 - acc: 0.3565 - val_loss: 2.0874 - val_acc: 0.1769
Epoch 3/15
0s - loss: 1.9540 - acc: 0.5003 - val_loss: 1.8878 - val_acc: 0.3917
...
Epoch 15/15
0s - loss: 0.0152 - acc: 0.9993 - val_loss: 0.0261 - val_acc: 0.9980
[INFO] evaluating network...
             precision    recall  f1-score   support

          1       1.00      1.00      1.00        45
          2       1.00      1.00      1.00        55
          3       1.00      1.00      1.00        63
          4       1.00      0.98      0.99        52
          5       0.98      1.00      0.99        51
          6       1.00      1.00      1.00        70
          7       1.00      1.00      1.00        50
          8       1.00      1.00      1.00        54
          9       1.00      1.00      1.00        63

avg / total       1.00      1.00      1.00       503

[INFO] serializing network...
```

正如我们所看到的，在仅仅 15 个时期之后，我们的网络在训练集和验证集上都获得了 100%的分类准确率。这也不是过度拟合的情况——当我们研究图 7 中的训练和验证曲线时，我们可以看到，在第 5 个时期，验证和训练损失/精度彼此匹配。

如果您检查`output`目录，您还会看到序列化的`lenet.hdf5`文件:

```py
$ ls -l output/
total 9844
-rw-rw-r-- 1 adrian adrian 10076992 May  3 12:56 lenet.hdf5
```

然后我们可以在新的输入图像上使用这个模型。

### **测试验证码破解程序**

现在我们的验证码破解程序已经训练好了，让我们在一些示例图片上测试一下。打开`test_model.py`文件并插入以下代码:

```py
# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from pyimagesearch.utils.captchahelper import preprocess
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
```

像往常一样，我们的 Python 脚本从导入 Python 包开始。我们将再次使用`preprocess`函数为分类准备数字。

接下来，我们将解析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-m", "--model", required=True,
	help="path to input model")
args = vars(ap.parse_args())
```

`--input`开关控制我们希望破解的输入验证码图像的路径。我们可以从 E-ZPass NY 网站下载一组新的验证码，但为了简单起见，我们将从现有的原始验证码文件中提取图像样本。`--model`参数只是驻留在磁盘上的序列化权重的路径。

我们现在可以加载我们预先训练好的 CNN，随机抽取 10 张验证码图片进行分类:

```py
# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# randomly sample a few of the input images
imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths, size=(10,),
	replace=False)
```

有趣的部分来了——破解验证码:

```py
# loop over the image paths
for imagePath in imagePaths:
	# load the image and convert it to grayscale, then pad the image
	# to ensure digits caught near the border of the image are
	# retained
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20,
		cv2.BORDER_REPLICATE)

	# threshold the image to reveal the digits
	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
```

在**的第 30 行，**我们开始循环每一个被采样的`imagePaths`。就像在`annotate.py`的例子中，我们需要提取验证码中的每个数字。这种提取是通过从磁盘加载图像，将其转换为灰度，并填充边界以使手指不能接触到图像的边界来完成的(**第 34-37 行**)。我们在这里添加*额外的填充*，这样我们就有足够的空间让*在图像上绘制*和*可视化*正确的预测。

**第 40 行和第 41 行**对图像进行阈值处理，使得数字显示为*白色前景*对*黑色背景*。

我们现在需要找到`thresh`图像中手指的轮廓:

```py
  	# find contours in the image, keeping only the four largest ones,
	# then sort them from left-to-right
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
	cnts = contours.sort_contours(cnts)[0]

	# initialize the output image as a "grayscale" image with 3
	# channels along with the output predictions
	output = cv2.merge([gray] * 3)
	predictions = []
```

我们可以通过调用`thresh`图像上的`cv2.findContours`来找到数字。这个函数返回一个( *x，y* )坐标列表，指定每个数字的*轮廓*。

然后，我们执行两个阶段的排序。第一阶段根据轮廓的*大小*对轮廓进行分类，只保留最大的四个轮廓。我们(正确地)假设具有最大尺寸的四个轮廓是我们想要识别的数字。然而，这些轮廓上没有保证的*空间排序*——我们希望识别的第三个数字可能在`cnts`列表中排在第一位。因为我们从左到右阅读数字，我们需要从左到右排序轮廓。这是通过`sort_contours`功能([http://pyimg.co/sbm9p](http://pyimg.co/sbm9p))完成的。

**第 53 行**获取我们的`gray`图像，并通过将灰度通道复制三次(红色、绿色和蓝色通道各一次)将其转换为三通道图像。然后我们通过 CNN 在**第 54 行**初始化我们的`predictions`列表。

鉴于验证码中数字的轮廓，我们现在可以破解它:

```py
  	# loop over the contours
	for c in cnts:
		# compute the bounding box for the contour then extract the
		# digit
		(x, y, w, h) = cv2.boundingRect(c)
		roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

		# pre-process the ROI and then classify it
		roi = preprocess(roi, 28, 28)
		roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
		pred = model.predict(roi).argmax(axis=1)[0] + 1
		predictions.append(str(pred))

		# draw the prediction on the output image
		cv2.rectangle(output, (x - 2, y - 2),
			(x + w + 4, y + h + 4), (0, 255, 0), 1)
		cv2.putText(output, str(pred), (x - 5, y - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
```

在**第 57 行，**我们循环遍历数字的每个轮廓(已经从左到右排序)。然后我们在**第 60 和 61 行**提取手指的 ROI，接着在**第 64 和 65 行对其进行预处理。**

**第 66 行**调用我们`model`的`.predict`方法。由`.predict`返回的*概率最大*的索引将是我们的类标签。我们将`1`加到这个值上，因为索引值从零开始；然而，没有零类，只有数字 1-9 的类。该预测随后被添加到第 67 行的`predictions`列表中。

**第 70 行和第 71 行**在当前数字周围绘制一个边界框，而**第 72 行和第 73 行**在`output`图像本身上绘制预测的数字。

我们的最后一个代码块处理将破解的验证码作为字符串写入我们的终端，并显示`output`图像:

```py
  	# show the output image
	print("[INFO] captcha: {}".format("".join(predictions)))
	cv2.imshow("Output", output)
	cv2.waitKey()
```

要查看我们的验证码破解程序，只需执行以下命令:

```py
$ python test_model.py --input downloads --model output/lenet.hdf5
Using TensorFlow backend.
[INFO] loading pre-trained network...
[INFO] captcha: 2696
[INFO] captcha: 2337
[INFO] captcha: 2571
[INFO] captcha: 8648
```

在图 8 的**、**中，我已经包含了从我的`test_model.py`运行中生成的四个样本。在*的每一个案例中，*我们都正确地预测了数字串，并使用基于少量训练数据训练的简单网络架构破解了图像验证码。

## **总结**

在本教程中，我们学习了如何:

1.  收集原始图像数据集。
2.  为我们的培训图像添加标签和注释。
3.  在我们的标记数据集上训练一个定制的卷积神经网络。
4.  在示例图像上测试和评估我们的模型。

为了做到这一点，我们从纽约的 E-ZPass 网站上搜集了 500 张验证码图片。然后，我们编写了一个 Python 脚本来帮助我们完成标记过程，使我们能够快速标记整个数据集，并将结果图像存储在一个有组织的目录结构中。

在我们的数据集被标记后，我们使用分类交叉熵损失在数据集上使用 SGD 优化器来训练 LeNet 架构，结果模型在零过拟合的测试集上获得了 100%的准确性。最后，我们将预测数字的结果可视化，以确认我们已经成功地设计了一种破解验证码的方法。

我想再次提醒你，本教程仅作为如何获取图像数据集并对其进行标记的*示例*。在*任何情况下*你都不应该出于邪恶的原因使用这个数据集或结果模型。如果你发现计算机视觉或深度学习可以被用来利用漏洞，一定要练习*负责任的披露*，并尝试向适当的利益相关者报告这个问题；不这样做是不道德的(滥用这一准则也是不道德的，从法律上讲，我不能对此负责)。

其次，本教程(下一个关于深度学习的微笑检测的教程也是如此)利用了计算机视觉和 OpenCV 库来帮助构建一个完整的应用程序。如果你打算成为一名认真的深度学习实践者，我*强烈建议*学习图像处理和 OpenCV 库的基础知识——即使对这些概念有一个基本的了解也能让你:

1.  欣赏更高层次的深度学习。
2.  开发更强大的应用程序，使用深度学习进行图像分类
3.  利用图像处理技术更快地实现您的目标。

在上面的**注释和创建我们的数据集**部分，我们可以找到一个使用基本图像处理技术的很好的例子，在那里我们能够快速注释和标记我们的数据集。如果不使用简单的计算机视觉技术，我们将不得不使用图像编辑软件(如 Photoshop 或 GIMP)手工裁剪并保存示例数字到磁盘。相反，我们能够编写一个快速而肮脏的应用程序，*自动*从验证码中提取每个数字——我们所要做的就是按下键盘上正确的键来标记图像。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*******