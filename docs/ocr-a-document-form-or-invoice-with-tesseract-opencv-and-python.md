# 使用 Tesseract、OpenCV 和 Python 对文档、表单或发票进行 OCR

> 原文：<https://pyimagesearch.com/2020/09/07/ocr-a-document-form-or-invoice-with-tesseract-opencv-and-python/>

在本教程中，您将学习如何使用 Tesseract、OpenCV 和 Python 对文档、表单或发票进行 OCR。

上周，我们讨论了如何接受输入图像并将其与模板图像对齐，如下所示:

在左边的*，*是我们的模板图像(例如，来自美国国税局的表单)。*中间的*图是我们希望与模板对齐的输入图像(从而允许我们将两幅图像中的字段匹配在一起)。最后，*右边的*显示了将两幅图像对齐的输出。

此时，我们可以将表单中的文本字段与模板中的每个对应字段相关联，**这意味着我们知道输入图像的哪些位置映射到姓名、地址、EIN 等。模板的字段:**

了解字段的位置和内容使我们能够对每个单独的字段进行 OCR，并跟踪它们以进行进一步的处理，如自动数据库输入。

但这也带来了问题:

*   我们如何着手实现这个文档 OCR 管道呢？
*   我们需要使用什么 OCR 算法？
*   这个 OCR 应用程序会有多复杂？

正如您将看到的，我们将能够用不到 150 行代码实现我们的整个文档 OCR 管道！

***注:**本教程是我即将出版的书 [OCR with OpenCV、Tesseract 和 Python](https://pyimagesearch.com/ocr-with-opencv-tesseract-and-python/) 中的一章的一部分。*

**要了解如何使用 OpenCV、Tesseract 和 Python 对文档、表格或发票进行 OCR，请继续阅读。**

## **使用 Tesseract、OpenCV 和 Python 对文档、表格或发票进行 OCR**

在本教程的第一部分，我们将简要讨论为什么我们可能想要 OCR 文档、表单、发票或任何类型的物理文档。

从那里，我们将回顾实现文档 OCR 管道所需的步骤。然后，我们将使用 OpenCV 和 Tesseract 在 Python 脚本中实现每个单独的步骤。

最后，我们将回顾对示例图像应用图像对齐和 OCR 的结果。

### 为什么要在表格、发票和文件上使用 OCR？

尽管生活在数字时代，我们仍然强烈依赖于*物理*纸面痕迹、*尤其是*大型组织，如政府、企业公司和大学/学院。

对实体文件跟踪的需求，加上几乎每份文件都需要组织、分类，甚至与组织中的多人*共享*这一事实，要求我们也*数字化*文件上的信息，并将其保存在我们的数据库中。

这些大型组织雇佣数据输入团队，他们的唯一目的是获取这些物理文档，手动重新键入信息，然后将其保存到系统中。

**光学字符识别算法可以*自动*数字化这些文档，提取信息，并通过管道将它们输入数据库进行存储，**减少了对庞大、昂贵甚至容易出错的手动输入团队的需求。

在本教程的其余部分，您将学习如何使用 OpenCV 和 Tesseract 实现一个基本的文档 OCR 管道。

### **使用 OpenCV 和 Tesseract 实现文档 OCR 管道的步骤**

用 OpenCV 和 Tesseract 实现文档 OCR 管道是一个多步骤的过程。在这一节中，我们将发现创建 OCR 表单管道所需的五个步骤。

**步骤#1** 涉及定义输入图像文档中字段的位置。我们可以通过在我们最喜欢的图像编辑软件中打开我们的模板图像来做到这一点，如 Photoshop、GIMP 或任何内置于操作系统中的照片应用程序。从那里，我们手动检查图像并确定边界框 *(x，y)*——我们想要进行 OCR 的每个字段的坐标，如图**图 4:** 所示

然后，我们接受一个包含要进行 OCR 的文档的输入图像(**步骤#2** )，并将其提交给我们的 OCR 管道(**图 5** ):

然后我们可以(**步骤#3** )应用**自动图像对齐/配准**来将输入图像与模板表单对齐(**图 6** )。

**步骤#4** 遍历所有文本字段位置(我们在**步骤#1** 中定义)，提取 ROI，并对 ROI 应用 OCR。在这个步骤中，我们能够 **OCR 文本本身*并且*将它与原始模板文档中的文本字段**相关联，如图**图 7:** 所示

最后的**步骤#5** 是显示我们输出的 OCR 文档，如图**图 8:** 所示

对于真实世界的用例，作为**步骤#5** 的替代方案，您可能希望将信息直接传输到会计数据库中。

我们将学习如何开发一个 Python 脚本，通过使用 OpenCV 和 Tesseract 创建 OCR 文档管道来完成本章中的第 1 步到第 5 步。

### **项目结构**

如果你想跟随今天的教程，找到 ***“下载”*** 部分并获取代码和图像存档。使用你最喜欢的解压工具解压文件。从那里，打开文件夹，您将看到以下内容:

```py
$ tree --dirsfirst
.
├── pyimagesearch
│   ├── alignment
│   │   ├── __init__.py
│   │   └── align_images.py
│   └── __init__.py
├── scans
│   ├── scan_01.jpg
│   └── scan_02.jpg
├── form_w4.png
└── ocr_form.py

3 directories, 7 files
```

*   ``scans/scan_01.jpg`` :一个例子 IRS W-4 文件，已经填了我的真实姓名，但是假的税务数据。
*   `scans/scan_02.jpg`:一个类似的 IRS W-4 文件的例子，其中填写了虚假的税务信息。
*   ``form_w4.png`` :官方 2020 IRS W-4 表格模板。这个空表单没有输入任何信息。我们需要它和现场位置，这样我们就可以排列扫描，并最终从扫描中提取信息。我们将使用外部照片编辑/预览应用程序手动确定字段位置。

我们只需要回顾一个 Python 驱动脚本:`ocr_form.py`。这个表单解析器依赖于两个辅助函数:

*   `align_images`:包含在`alignment`子模块内，上周[首次推出](https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/)。本周我们不会再复习这个方法，所以如果你错过了，一定要参考我之前的教程！
*   ``cleanup_text`` :这个函数出现在我们的驱动程序脚本的顶部，简单地消除 OCR 检测到的非 ASCII 字符(我将在下一节分享更多关于这个函数的内容)。

如果您准备好了，就直接进入下一个实现部分吧！

### **使用 OpenCV 和 Tesseract 实现我们的文档 OCR 脚本**

我们现在准备使用 OpenCV 和 Tesseract 实现我们的文档 OCR Python 脚本。

打开一个新文件，将其命名为`ocr_form.py`，并插入以下代码:

```py
# import the necessary packages
from pyimagesearch.alignment import align_images
from collections import namedtuple
import pytesseract
import argparse
import imutils
import cv2
```

```py
def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()
```

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image that we'll align to template")
ap.add_argument("-t", "--template", required=True,
	help="path to input template image")
args = vars(ap.parse_args())
```

我们将图像与模板对齐，然后根据需要对各个字段进行 OCR。

现在，我们没有创建一个“智能表单 OCR 系统”,在这个系统中，所有的文本都被识别，字段都是基于正则表达式模式设计的。这当然是可行的——这是我即将出版的 OCR 书中介绍的一种先进方法。

相反，为了保持本教程的轻量级，我已经为我们关心的每个字段手动定义了`OCR_Locations`。好处是我们能够给每个字段一个名称，并指定精确的 *(x，y)*-坐标作为字段的边界。现在让我们在**步骤#1** 中定义文本字段的位置:

```py
# create a named tuple which we can use to create locations of the
# input document which we wish to OCR
OCRLocation = namedtuple("OCRLocation", ["id", "bbox",
	"filter_keywords"])

# define the locations of each area of the document we wish to OCR
OCR_LOCATIONS = [
	OCRLocation("step1_first_name", (265, 237, 751, 106),
		["middle", "initial", "first", "name"]),
	OCRLocation("step1_last_name", (1020, 237, 835, 106),
		["last", "name"]),
	OCRLocation("step1_address", (265, 336, 1588, 106),
		["address"]),
	OCRLocation("step1_city_state_zip", (265, 436, 1588, 106),
		["city", "zip", "town", "state"]),
	OCRLocation("step5_employee_signature", (319, 2516, 1487, 156),
		["employee", "signature", "form", "valid", "unless",
		 	"you", "sign"]),
	OCRLocation("step5_date", (1804, 2516, 504, 156), ["date"]),
	OCRLocation("employee_name_address", (265, 2706, 1224, 180),
		["employer", "name", "address"]),
	OCRLocation("employee_ein", (1831, 2706, 448, 180),
		["employer", "identification", "number", "ein"]),
]
```

这里，**行 24 和 25** 创建了一个命名元组，由以下内容组成:

*   ``name = "OCRLocation"`` :我们元组的名称。
*   ``"id"`` :该字段的简短描述，便于参考。使用此字段描述表单字段的实际内容。例如，它是邮政编码字段吗？
*   ``"bbox"`` :列表形式字段的边框坐标，使用如下顺序:`[x, y, w, h]`。在这种情况下， *x* 和 *y* 是左上坐标， *w* 和 *h* 是宽度和高度。
*   `"filter_keywords"`:我们不希望在 OCR 中考虑的单词列表，如**图 12** 所示的表单域指令。

**第 28-45 行**定义了官方 2020 IRS W-4 税表的 ***八个*字段，如图**图 9:****

```py
# load the input image and template from disk
print("[INFO] loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])

# align the images
print("[INFO] aligning images...")
aligned = align_images(image, template)
```

```py
$ convert /path/to/taxes/2020/forms/form_w4.pdf ./form_w4.png
```

ImageMagick 足够智能，能够根据文件扩展名和文件本身识别出您想要将 PDF 转换为 PNG 图像。如果您愿意，可以很容易地修改命令来生成 JPG。

你有许多表格吗？只需使用 ImageMagick 的`mogrify`命令，它支持通配符([参考文档](http://www.imagemagick.org/Usage/basics/#mogrify))。

假设您的文档是 PNG 或 JPG 格式，您可以像我们今天的教程一样，在 OpenCV 和 PyTesseract 中使用它！

注意我们的输入图像*(左)*是如何与模板文档*(右)对齐的。*

下一步(**步骤#4** )是循环遍历我们的每个`OCR_LOCATIONS`和**应用光学字符识别到每个文本字段**使用宇宙魔方和宇宙魔方的力量:

```py
# initialize a results list to store the document OCR parsing results
print("[INFO] OCR'ing document...")
parsingResults = []

# loop over the locations of the document we are going to OCR
for loc in OCR_LOCATIONS:
	# extract the OCR ROI from the aligned image
	(x, y, w, h) = loc.bbox
	roi = aligned[y:y + h, x:x + w]

	# OCR the ROI using Tesseract
	rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	text = pytesseract.image_to_string(rgb)
```

首先，我们初始化`parsingResults`列表来存储每个文本字段的 OCR 结果(**第 58 行**)。从那里，我们继续循环每个`OCR_LOCATIONS`(从**行 61** 开始)，这是我们之前手动定义的。

```py
	# break the text into lines and loop over them
	for line in text.split("\n"):
		# if the line is empty, ignore it
		if len(line) == 0:
			continue

		# convert the line to lowercase and then check to see if the
		# line contains any of the filter keywords (these keywords
		# are part of the *form itself* and should be ignored)
		lower = line.lower()
		count = sum([lower.count(x) for x in loc.filter_keywords])

		# if the count is zero then we know we are *not* examining a
		# text field that is part of the document itself (ex., info,
		# on the field, an example, help text, etc.)
		if count == 0:
			# update our parsing results dictionary with the OCR'd
			# text if the line is *not* empty
			parsingResults.append((loc, line))
```

虽然我已经在这个字段中填写了我的名字，*“Adrian”，*文本*“(a)名字和中间名首字母”*仍将由 Tesseract 进行 OCR 识别——***上面的代码会自动过滤掉字段中的说明文本*，确保只返回人工输入的文本。**

我们快到了，坚持住！让我们继续后处理我们的`parsingResults`来清理它们:

```py
# initialize a dictionary to store our final OCR results
results = {}

# loop over the results of parsing the document
for (loc, line) in parsingResults:
	# grab any existing OCR result for the current ID of the document
	r = results.get(loc.id, None)

	# if the result is None, initialize it using the text and location
	# namedtuple (converting it to a dictionary as namedtuples are not
	# hashable)
	if r is None:
		results[loc.id] = (line, loc._asdict())

	# otherwise, there exists an OCR result for the current area of the
	# document, so we should append our existing line
	else:
		# unpack the existing OCR result and append the line to the
		# existing text
		(existingText, loc) = r
		text = "{}\n{}".format(existingText, line)

		# update our results dictionary
		results[loc["id"]] = (text, loc)
```

我们最终的`results`字典(**第 91 行**)将很快保存清理后的解析结果，包括文本位置的唯一 ID(键)和经过 OCR 处理的文本的二元组及其位置(值)。让我们通过在**行 94** 上循环我们的`parsingResults`来开始填充我们的`results`。我们的循环完成三项任务:

*   我们获取当前文本字段 ID 的任何现有结果。
*   如果没有当前结果，我们简单地将文本`line`和文本`loc`(位置)存储在`results`字典中。
*   否则，我们将把`line`附加到由新行分隔的任何`existingText`中，并更新`results`字典。

我们终于准备好执行**步骤# 5**——可视化我们的 OCR `results`:

```py
# loop over the results
for (locID, result) in results.items():
	# unpack the result tuple
	(text, loc) = result

	# display the OCR result to our terminal
	print(loc["id"])
	print("=" * len(loc["id"]))
	print("{}\n\n".format(text))

	# extract the bounding box coordinates of the OCR location and
	# then strip out non-ASCII text so we can draw the text on the
	# output image using OpenCV
	(x, y, w, h) = loc["bbox"]
	clean = cleanup_text(text)

	# draw a bounding box around the text
	cv2.rectangle(aligned, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# loop over all lines in the text
	for (i, line) in enumerate(text.split("\n")):
		# draw the line on the output image
		startY = y + (i * 70) + 40
		cv2.putText(aligned, line, (x, startY),
			cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
```

```py
# show the input and output images, resizing it such that they fit
# on our screen
cv2.imshow("Input", imutils.resize(image, width=700))
cv2.imshow("Output", imutils.resize(aligned, width=700))
cv2.waitKey(0)
```

如您所见，**第 143 行和第 144** 行首先应用方面感知的大小调整，因为在向用户显示结果和原始数据之前，高分辨率扫描往往不适合普通的计算机屏幕。要停止程序，只需在其中一个窗口处于焦点时按任意键。

用 Python、OpenCV 和 Tesseract 实现您的自动化 OCR 系统做得很好！在下一部分，我们将对它进行测试。

### **使用 OpenCV 和 Tesseract 的 OCR 结果**

我们现在准备使用 OpenCV 和 Tesseract 对文档进行 OCR。

确保使用本教程的 ***“下载”*** 部分下载与本文相关的源代码和示例图片。

从那里，打开一个终端，并执行以下命令:

```py
$ python ocr_form.py --image scans/scan_01.jpg --template form_w4.png
[INFO] loading images...
[INFO] aligning images...
[INFO] OCR'ing document...
step1_first_name
================
Adrian

step1_last_name
===============
Rosebrock

step1_address
=============
PO Box 17598 #17900

step1_city_state_zip
====================
Baltimore, MD 21297-1598

step5_employee_signature
========================
Adrian Rosebrock

step5_date
==========
2020/06/10

employee_name_address
=====================
PylmageSearch
PO BOX 1234
Philadelphia, PA 19019

employee_ein
============
12-3456789
```

这里，我们有我们的输入图像及其相应的模板:

这是图像对齐和文档 OCR 管道的输出:

请注意我们是如何成功地将输入图像与文档模板对齐，本地化每个字段，然后对每个单独的字段进行 OCR 的。

我们的实现还*忽略了*属于文档本身一部分的字段*中的任何一行文本。*

例如，名字段提供说明文本*(a)名和中间名首字母"*；然而，我们的 OCR 管道和关键字过滤过程能够检测到这是文档本身的一部分(即，不是某个*人*输入的内容)，然后简单地忽略它。

总的来说，我们已经能够成功地对文档进行 OCR 了！

让我们尝试另一个示例图像，这次视角略有不同:

```py
$ python ocr_form.py --image scans/scan_02.jpg --template form_w4.png
[INFO] loading images...
[INFO] aligning images...
[INFO] OCR'ing document...
step1_first_name
================
Adrian

step1_last_name
===============
Rosebrock

step1_address
=============
PO Box 17598 #17900

step1_city_state_zip
====================
Baltimore, MD 21297-1598

step5_employee_signature
========================
Adrian Rosebrock

step5_date
==========
2020/06/10

employee_name_address
=====================
PyimageSearch
PO BOX 1234
Philadelphia, PA 19019

employee_ein
============
12-3456789
```

同样，这是我们的输入图像及其模板:

下图包含我们的输出，您可以看到图像已与模板对齐，OCR 已成功应用于每个字段:

**同样，我们已经能够成功地将输入图像与模板文档对齐，然后对每个单独的字段进行 OCR！**

## **总结**

在本教程中，您学习了如何使用 OpenCV 和 Tesseract 对文档、表单或发票进行 OCR。

我们的方法依赖于[图像对齐](https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/)，这是一个接受输入图像和模板图像的过程，然后将它们对齐，以便它们可以整齐地“叠加”在彼此之上。在光学字符识别的环境中，图像对齐允许我们将模板中的每个文本字段与我们的输入图像对齐，这意味着一旦我们对文档进行了 OCR，我们就可以将 OCR 文本与每个字段相关联(例如、姓名、地址等。).

应用图像对齐后，我们使用 Tesseract 来识别输入图像中预先选择的文本字段，同时过滤掉不相关的说明信息。

我希望你喜欢这个教程——更重要的是，我希望你可以在自己的项目中应用图像对齐和 OCR 时使用它。

如果你想了解更多关于光学字符识别的知识，一定要看看我的书 *[OCR with OpenCV，Tesseract，和 Python](http://pyimg.co/ocrigg) 。*

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***