# 火炬中心系列# 5:MiDaS——深度估计模型

> 原文：<https://pyimagesearch.com/2022/01/17/torch-hub-series-5-midas-model-on-depth-estimation/>

在本系列的前一部分中，我们讨论了一些最新的对象检测模型；YOLOv5 和 SSD。在今天的教程中，我们将讨论 MiDaS，这是一种帮助图像深度估计的巧妙尝试。

通过本教程，我们将创建一个关于 MiDaS 背后思想的基本直觉，并学习如何将其用作深度估计推理工具。

本课是关于火炬中心的 6 部分系列的第 5 部分:

1.  [*火炬中心系列# 1:*](https://pyimagesearch.com/2021/12/20/torch-hub-series-1-introduction-to-torch-hub/)火炬中心介绍
2.  [*火炬枢纽系列#2: VGG 和雷斯内特*](https://pyimagesearch.com/2021/12/27/torch-hub-series-2-vgg-and-resnet/)
3.  [*火炬轮毂系列#3: YOLO v5 和 SSD*——*实物检测上的型号*](https://pyimagesearch.com/2022/01/03/torch-hub-series-3-yolov5-and-ssd-models-on-object-detection/)
4.  [*火炬轮毂系列# 4:*—*甘上模型*](https://pyimagesearch.com/2022/01/10/torch-hub-series-4-pgan-model-on-gan/)
5.  *火炬轮毂系列# 5:MiDaS*——*深度估计模型*(本教程)
6.  *火炬中枢系列#6:图像分割*

**要了解如何使用 MiDaS 对您的数据进行自定义，** ***只要继续阅读。***

## **火炬轮毂系列# 5:MiDaS——深度估计模型**

### **简介**

首先，让我们了解什么是深度估计，或者为什么它很重要。图像的深度估计从 2D 图像本身预测对象的顺序(如果图像以 3D 格式扩展)。这无疑是一项艰巨的任务，因为获得这个领域的带注释的数据和数据集本身就是一项艰巨的任务。深度估计的用途非常广泛，最引人注目的是在自动驾驶汽车领域，估计汽车周围物体的距离有助于导航(**图 1** )。

迈达斯背后的研究人员以非常简单的方式解释了他们的动机。他们坚定地断言，在处理包含现实生活问题的问题陈述时，单一数据集上的训练模型将是不健壮的。当实时使用的模型被创建时，它们应该足够健壮以处理尽可能多的情况和异常值。

牢记这一点，MiDaS 的创造者决定在多个数据集上训练他们的模型。这包括具有不同类型标签和目标函数的数据集。为了实现这一点，他们设计了一种方法，在与所有地面实况表示兼容的适当输出空间中进行计算。

这个想法在理论上非常巧妙，但作者必须仔细设计损失函数，并考虑使用多个数据集所带来的挑战。由于这些数据集在不同程度上具有不同的深度估计表示，正如论文作者所述，出现了固有的比例模糊和移位模糊。

现在，因为所有的数据集都可能遵循彼此不同的分布。因此，这些问题在意料之中。然而，作者对每个挑战都提出了解决方案。最终产品是一个强大的深度估计器，既高效又准确。在**图 2** 中，我们看到了论文中显示的一些结果。

跨数据集学习的想法并不新鲜，但是将基础事实放到一个公共输出空间中所带来的复杂性是非常难以克服的。然而，这篇论文详尽地解释了每一步，从直觉到所用损失的数学定义。

让我们看看如何使用 MiDaS 模型来找到自定义图像的反向深度。

### **配置**您的开发环境****

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果你需要帮助为 OpenCV 配置开发环境，我*强烈推荐*阅读我的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让你启动并运行。

### **在配置开发环境时遇到了**个问题？****

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目**结构****

我们首先需要回顾我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

从这里，看一下目录结构:

```py
!tree .
.
├── midas_inference.py
├── output
│   └── midas_output
│       └── output.png
└── pyimagesearch
    ├── config.py
    └── data_utils.py
```

在`pyimagesearch`目录中，我们有 2 个脚本:

*   `config.py`:包含项目的端到端配置管道
*   `data_utils.py`:包含了我们将在项目中使用的两个数据实用函数

在父目录中，我们有一个脚本:

*   `midas_inference.py`:根据预训练的 MiDaS 模型进行推断

最后，我们有`output`目录，它将存放从运行脚本中获得的结果图。

### **下载数据集**

由于其紧凑性，我们将再次使用来自 Kaggle 的[狗&猫图像](https://www.kaggle.com/chetankv/dogs-cats-images)数据集。

```py
$ mkdir ~/.kaggle
$ cp <path to your kaggle.json> ~/.kaggle/
$ chmod 600 ~/.kaggle/kaggle.json
$ kaggle datasets download -d chetankv/dogs-cats-images
$ unzip -qq dogs-cats-images.zip
$ rm -rf "/content/dog vs cat"
```

正如在本系列的前几篇文章中所解释的，您需要自己独特的`kaggle.json`文件来连接 Kaggle API ( **第 2 行**)。**第 3 行**上的`chmod 600`命令将允许你的脚本完全访问读写文件。

下面的`kaggle datasets download`命令(**第 4 行**)允许您下载他们网站上托管的任何数据集。最后，我们有 unzip 命令和一个用于不必要添加的辅助 delete 命令(**第 5 行和第 6 行**)。

让我们转到配置管道。

### **配置先决条件**

在`pyimagesearch`目录中，您会发现一个名为`config.py`的脚本。这个脚本将包含我们项目的完整的端到端配置管道。

```py
# import the necessary packages
import torch
import os

# define the root directory followed by the test dataset paths
BASE_PATH = "dataset"
TEST_PATH = os.path.join(BASE_PATH, "test_set")

# specify image size and batch size
IMAGE_SIZE = 384
PRED_BATCH_SIZE = 4

# determine the device type 
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# define paths to save output 
OUTPUT_PATH = "output"
MIDAS_OUTPUT = os.path.join(OUTPUT_PATH, "midas_output")
```

首先，我们将`BASE_PATH`变量作为数据集目录的指针(**第 6 行**)。我们没有对我们的模型做任何额外的修改，所以我们将只使用测试集(**第 7 行**)。

在**第 10 行**上，我们有一个名为`IMAGE_SIZE`的变量，设置为`384`，作为我们 MiDaS 模型输入的指令。预测批量大小设置为`4` ( **第 11 行**)，但是鼓励读者尝试不同的大小。

建议您为今天的项目准备一个兼容 CUDA 的设备( **Line 14** )，但是由于我们不打算进行任何繁重的训练，CPU 应该也能正常工作。

最后，我们创建了路径来保存从模型推断中获得的输出(**第 17 行和第 18** )。

在今天的任务中，我们将只使用一个助手函数来帮助我们的管道。为此，我们将转到`pyimagesearch`目录中的第二个脚本`data_utils.py`。

```py
# import the necessary packages
from torch.utils.data import DataLoader

def get_dataloader(dataset, batchSize, shuffle=True):
	# create a dataloader and return it
	dataLoader= DataLoader(dataset, batch_size=batchSize,
		shuffle=shuffle)
	return dataLoader
```

在第 4 行的**上，我们有`get_dataloader`函数，它接受数据集、批量大小和随机变量作为它的参数。这个函数返回一个类似 PyTorch Dataloader 实例的生成器，它将帮助我们处理大量数据(**第 6 行**)。**

这就是我们的公用事业。让我们继续推理脚本。

### **使用 MiDaS 进行反向深度估计**

这个时候，你的脑海里可能会蹦出一个很符合逻辑的问题；为什么我们要花如此大的力气从一堆图片中得出结论呢？

我们在这里选择的方法是一种处理大型数据集的完全可靠的方法，即使您选择训练模型以便稍后进行微调，管道也是有用的。我们还考虑在不调用 MiDaS 储存库的预制功能的情况下尽可能准备数据。

```py
# import necessary packages
from pyimagesearch.data_utils import get_dataloader
from pyimagesearch import config
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torch
import os

# create the test dataset with a test transform pipeline and
# initialize the test data loader
testTransform = Compose([
	Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), ToTensor()])
testDataset = ImageFolder(config.TEST_PATH, testTransform)
testLoader = get_dataloader(testDataset, config.PRED_BATCH_SIZE)
```

如前所述，由于我们将只使用测试集，我们已经创建了一个 PyTorch 测试转换实例，我们正在对图像进行整形，并将它们转换为张量(**行 12 和 13** )。

如果你的数据集的格式与我们在项目中使用的格式相同(例如，文件夹下的图像命名为 labels)，那么我们可以使用`ImageFolder`函数创建一个 PyTorch 数据集实例(**第 14 行**)。最后，我们使用之前定义的`get_dataloader`函数来生成一个数据加载器实例(**第 15 行**)。

```py
# initialize the midas model using torch hub
modelType = "DPT_Large" 
midas = torch.hub.load("intel-isl/MiDaS", modelType)

# flash the model to the device and set it to eval mode
midas.to(device)
midas.eval()
```

接下来，我们使用`torch.hub.load`函数在本地运行时加载 MiDaS 模型(**第 18 行和第 19 行**)。同样，几个可用的选择可以在这里调用，所有这些都可以在这里找到[。最后，我们将模型加载到我们的设备中，并将其设置为评估模式(**第 22 行和第 23 行**)。](https://github.com/isl-org/MiDaS)

```py
# initialize iterable variable
sweeper = iter(testLoader)

# grab a batch of test data send the images to the device
print("[INFO] getting the test data...")
batch = next(sweeper)
(images, _) = (batch[0], batch[1])
images = images.to(config.DEVICE) 

# turn off auto grad
with torch.no_grad():
	# get predictions from input
	prediction = midas(images)

	# unsqueeze the predictions batchwise
	prediction = torch.nn.functional.interpolate(
		prediction.unsqueeze(1), size=[384,384], mode="bicubic",
		align_corners=False).squeeze()

# store the predictions in a numpy array
output = prediction.cpu().numpy()
```

第 26 行**上的`sweeper`变量将作为`testLoader`的可迭代变量。每次我们运行第 30** 行**上的命令，我们将从`testLoader`获得新的一批数据。在**第 31 行**上，我们将该批产品拆包成图像和标签，仅保留图像。**

在将图像加载到我们的设备(**第 32 行**)后，我们关闭自动渐变并让图像通过模型(**第 35-37 行**)。在**第 40 行**上，我们使用一个漂亮的效用函数`torch.nn.functional.interpolate`将我们的预测解包成一个有效的 3 通道图像格式。

最后，我们将重新格式化的预测存储到 numpy 格式的输出变量中(**第 45 行**)。

```py
# define row and column variables
rows = config.PRED_BATCH_SIZE
cols = 2

# define axes for subplots
axes = []
fig=plt.figure(figsize=(10, 20))

# loop over the rows and columns
for totalRange in range(rows*cols):
	axes.append(fig.add_subplot(rows, cols, totalRange+1))

	# set up conditions for side by side plotting 
	# of ground truth and predictions
	if totalRange % 2 == 0:
		plt.imshow(images[totalRange//2]
			.permute((1, 2, 0)).cpu().detach().numpy())
	else :
		plt.imshow(output[totalRange//2])
fig.tight_layout()

# build the midas output directory if not already present
if not os.path.exists(config.MIDAS_OUTPUT):
	os.makedirs(config.MIDAS_OUTPUT)

# save plots to output directory
print("[INFO] saving the inference...")
outputFileName = os.path.join(config.MIDAS_OUTPUT, "output.png")
plt.savefig(outputFileName)
```

为了绘制我们的结果，我们首先定义行和列变量来定义网格格式(**行 48 和 49** )。在**的第 52 行和第 53 行**，我们定义了支线剧情列表和人物大小。

在行和列上循环，我们定义了一种方法，其中逆深度估计和地面真实图像并排绘制(**行 56-66** )。

最后，我们将图形保存到我们想要的路径中(**第 69 行和第 75 行**)。

我们的推理脚本完成后，让我们检查一些结果。

### **MiDaS 推断结果**

在我们的数据集中，大多数图像的最前面都有一只猫或一只狗。可能没有足够的背景图像，但 MiDaS 模型应该会给我们一个输出，明确地描绘出前景中的猫或狗。这正是在我们的推理图像中发生的事情(**图 4-7** )。

虽然 MiDaS 的惊人能力可以在所有的推理图像中看到，但我们可以深入研究它们，并得出一些更多的观察结果。

在**图 4** 中，不仅猫被描绘在前景中，而且它的头部比身体更靠近相机(通过颜色的变化来显示)。在**图 5** 中，由于场地主要覆盖图像，所以狗和场地有明显的区别。在**图 6** 和**图 7** 中，猫的头部被描绘得比身体更近。

## **总结**

把握迈达斯在当今世界的重要性，对我们来说是重要的一步。想象一下，一个完美的深度估计器对自动驾驶汽车会有多大的帮助。因为自动驾驶汽车几乎完全依赖于激光雷达(光探测和测距)、相机、声纳(声音导航和测距)等实用工具。拥有对其周围环境进行深度估计的完全可靠的系统将使旅行更加安全，并减轻其他传感器系统的负担。

关于 MiDaS 要注意的第二件最重要的事情是跨数据集混合的使用。虽然它最近获得了动力，但完美地执行它需要相当多的时间和计划。此外，MiDaS 是在一个对现实世界问题有重大影响的领域中做到这一点的。

我希望这篇教程能够激发你对自治系统和深度估计的兴趣。请随意使用您的自定义数据集进行尝试，并分享结果。

### **引文**信息

**Chakraborty，D.** “火炬中心系列# 5:MiDaS-深度估计模型”， *PyImageSearch* ，2022 年，[https://PyImageSearch . com/2022/01/17/Torch-Hub-Series-5-MiDaS-深度估计模型/](https://pyimagesearch.com/2022/01/17/torch-hub-series-5-midas-model-on-depth-estimation/)

```py
@article{Chakraborty_2022_THS5,
  author = {Devjyoti Chakraborty},
  title = {Torch Hub Series \#5: {MiDaS} — Model on Depth Estimation},
  journal = {PyImageSearch},
  year = {2022},
  note = {https://pyimagesearch.com/2022/01/17/torch-hub-series-5-midas-model-on-depth-estimation/},
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****