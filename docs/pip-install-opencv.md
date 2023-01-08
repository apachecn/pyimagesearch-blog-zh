# pip 安装 OpenCV

> 原文：<https://pyimagesearch.com/2018/09/19/pip-install-opencv/>

在本教程中，您将学习如何在 Ubuntu、macOS 和 Raspberry Pi 上 pip 安装 OpenCV。

在以前的 OpenCV 安装教程中，我推荐过从源代码编译；然而，在过去的一年里，通过 pip 安装 OpenCV 成为可能，pip 是 Python 自己的包管理器。

虽然从源代码安装可以让您最大程度地控制 OpenCV 配置，但这也是最困难和最耗时的。

如果你正在寻找在你的系统上安装 OpenCV 的最快的方法，你想使用 pip 来安装 OpenCV *(但是在这个过程中有一些事情可能会使你出错，所以请确保你阅读了本指南的其余部分)*。

***2019-11-21 更新:*** *由于在运行 BusterOS 的 Raspberry Pi 4 上使用这种 pip 安装方式存在 OpenCV 的兼容性问题，已经对这篇博文发布了更新。一定要在搜索“2019-11-21 更新”时通过`ctrl + f`找到更新。*

**要了解如何在你的系统上 pip 安装 OpenCV，*继续阅读*。**

## pip 安装 OpenCV

在本教程的剩余部分，我将简要描述可以通过 Python 的包管理器 pip 安装的 OpenCV 包。

从那里，我将演示如何在 Ubuntu、macOS 和 Raspberry Pi 上 pip 安装 OpenCV。

最后，我将回顾一下使用 pip 安装 OpenCV 时可能会遇到的一些常见问题。

在我们开始之前，我想指出这个 OpenCV 安装方法的一个重要注意事项。

我们今天讨论的 OpenCV 的 PyPi/PiWheels 托管版本**不包括“非自由”算法**，例如 SIFT、SURF 和其他专利算法。如果你需要一个快速的环境，不需要运行包含非自由算法的程序，这是一个安装 OpenCV 的好方法——如果不是这样，你需要完成 OpenCV 的完整编译。

### 两个 pip OpenCV 包:`opencv-python`和`opencv-contrib-python`

在我们开始之前，我想提醒你，我今天来这里的方法是 ***非官方的*预建的 OpenCV 包**，可以通过 pip 安装——它们是*而不是*OpenCV.org[发布的官方 OpenCV 包](http://opencv.org/)。

仅仅因为它们不是官方的软件包，并不意味着你应该对使用它们感到不舒服，但是对你来说重要的是要明白它们并没有得到 OpenCV.org 官方团队的直接认可和支持。

综上所述——在 PyPI 库上有四个 OpenCV 包可以通过 pip 安装:

1.  **[opencv-python](https://pypi.org/project/opencv-python/) :** 这个库包含了 ***正好是 opencv 库的主要模块*** 。如果你是 PyImageSearch 阅读器，你*不想*安装这个包。
2.  **[opencv-contrib-python](https://pypi.org/project/opencv-contrib-python/):**OpenCV-contrib-python 库包含**两个*主模块*以及 *contrib 模块***——这是我**推荐您安装的**库，因为它包含了所有 OpenCV 功能。
3.  **[opencv-python-headless](https://pypi.org/project/opencv-python-headless/):**与 opencv-python 相同，但没有 GUI 功能。对无头系统有用。
4.  **[opencv-contrib-python-headless](https://pypi.org/project/opencv-contrib-python-headless/):**与 opencv-contrib-python 相同，但没有 GUI 功能。对无头系统有用。

同样，*在绝大多数情况下*你会想在你的系统上安装`opencv-contrib-python`。

你 ***不要**想把`opencv-python`和`opencv-contrib-python`都装*——挑**其中一个**。

### 如何在 Ubuntu 上 pip 安装 OpenCV

在 Ubuntu 上用 pip 安装 OpenCV 有两种选择:

1.  安装到您的系统中`site-packages`
2.  安装到虚拟环境的`site-packages` ***(首选)***

#### 首先，在 Ubuntu 上安装一些 OpenCV 依赖项

我们需要用 [apt-get](https://help.ubuntu.com/community/AptGet/Howto) 包管理器刷新/升级预安装的包/库:

```py
$ sudo apt-get update
$ sudo apt-get upgrade
```

然后安装两个必需的包:

```py
$ sudo apt-get install python3-dev
$ sudo apt-get install libgl1-mesa-glx
```

#### 接下来，安装 pip

如果您没有 pip，您需要先获得它:

```py
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python3 get-pip.py

```

#### 选项 A:用 pip 把 OpenCV 安装到你的 Ubuntu 系统上

我不推荐这种方法，除非您有不希望隔离、独立的 Python 环境的特殊用例。

让我们在我们的系统上安装 opencv-contrib-python:

```py
$ sudo pip install opencv-contrib-python

```

几秒钟之内，OpenCV 就可以放入您系统的站点包中了！

#### 选项 B:用 pip 将 Ubuntu 上的 OpenCV 安装到虚拟环境中

Python 虚拟环境有巨大的好处。

主要的好处是，您可以在您的系统上开发多个带有独立包的项目(许多都有版本依赖)，而不必把您的系统搅浑。您还可以随时添加和删除虚拟环境。

简而言之:Python 虚拟环境是 Python 开发的最佳实践。很有可能，你应该赶时髦。

我选择的工具是`virtualenv`和`virtualenvwrapper`，但是您也可以选择其他工具，比如 venv 或 Anaconda(简称 conda)。

下面介绍如何安装`virtualenv`和`virtualenvwrapper`，它们都将驻留在您的 ***系统*** `site-packages`中，并管理每个项目的 ***虚拟环境*** 站点包:

```py
$ pip install virtualenv virtualenvwrapper

```

在我们继续之前，您首先需要在您的`~/.bashrc`个人资料中添加一些行。使用`nano`、`vim`或`emacs`打开文件，并将这些行附加到末尾:

```py
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.local/bin/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.local/bin/virtualenv
source $HOME/.local/bin/virtualenvwrapper.sh
```

保存文件。然后在您的终端中“搜索它”:

```py
$ source ~/.bashrc

```

您将看到一些设置 virtualenvwrapper 的终端输出。您现在可以访问新的终端命令:

*   **用`mkvirtualenv`创造**的环境。
*   **用`workon`激活**一个环境(或**切换**到不同的环境)。
*   **用`deactivate`停用**一个环境。
*   **用`rmvirtualenv`去掉**一个环境。
*   *务必[阅读文件](https://virtualenvwrapper.readthedocs.io/en/latest/)！*

让我们**为 OpenCV 创建**一个 Python 3 虚拟环境，名为 CV:

```py
$ mkvirtualenv cv -p python3

```

现在有了魔棒(pip)，您可以在几秒钟内将 OpenCV 安装到您的新环境中:

```py
$ pip install opencv-contrib-python

```

### 如何在 macOS 上 pip 安装 OpenCV

MacOS 的 pip 类似于 Ubuntu 安装 OpenCV。

同样，在带有 pip 的 macOS 上安装 OpenCV 有两种选择:

1.  安装到您的系统中`site-packages`
2.  安装到虚拟环境的`site-packages` ***(首选)***

#### 安装 pip

如果您没有 pip，您需要先获得它:

```py
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python3 get-pip.py

```

#### 选项 A:使用 pip 将 OpenCV 安装到 macOS 系统中

不要这样。

为什么？我实际上建议你去**选项 B** 使用虚拟环境。

好吧，如果你坚持在你的 macOS 系统上安装，那么就像 pip 通过以下方式安装 OpenCV 一样简单:

```py
$ sudo pip install opencv-contrib-python

```

几秒钟之内，OpenCV 就可以放入系统的站点包中了。

#### 选项 B:使用 pip 将 macOS 上的 OpenCV 安装到虚拟环境中

就像用 pip 管理软件包一样轻而易举。

…在虚拟环境中，管理项目及其依赖关系轻而易举。

如果你对计算机视觉开发(或任何这方面的开发)很认真，你应该使用 Python 虚拟环境。

我不在乎你使用什么系统(是`virtualenv`、`venv`还是`conda` /Anaconda】，只要学会使用一个并坚持下去。

下面介绍如何安装 virtualenv 和 virtualenvwrapper，它们都将驻留在您的 ***系统*** 站点包中，并管理每个项目的 ***虚拟环境*** 站点包:

```py
$ pip install virtualenv virtualenvwrapper

```

从那里，你需要添加下面几行到你的`~/.bash_profile`(注意 macOS 的文件名是`.bash_profile`，Ubuntu 的文件名是`.bashrc`。

使用`nano`、`vim`或`emacs`(大多数系统都有`nano`)打开文件:

```py
$ nano ~/.bash_profile

```

…并将这几行附加到末尾:

```py
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python3
source /usr/local/bin/virtualenvwrapper.sh

```

保存文件——如果您使用的是`nano`,快捷键会列在窗口底部。

然后在您的终端中“搜索它”:

```py
$ source ~/.bash_profile

```

您将看到几行终端输出，表明 virtualenvwrapper 已经设置好了。您现在可以访问新的终端命令:

*   创建一个新的虚拟环境。
*   `workon`:激活/切换到虚拟环境。请记住，您可以拥有任意多的环境。
*   跳出虚拟环境，你将与你的系统一起工作。
*   `rmvirtualenv`:删除虚拟环境。
*   *务必[阅读文件](https://virtualenvwrapper.readthedocs.io/en/latest/)！*

让我们**为 OpenCV 创建**一个 Python 3 虚拟环境，名为 CV:

```py
$ mkvirtualenv cv -p python3

```

现在，使用 pip，只需一眨眼的时间，您就可以在几秒钟内将 OpenCV 安装到您的新环境中:

```py
$ pip install opencv-contrib-python

```

### 如何在树莓 Pi 上安装 OpenCV

在这篇文章的前面，我提到过安装 OpenCV 的一个缺点是你对编译本身没有任何控制——二进制文件是为你预先构建的，这虽然很好，但也意味着你不能包含任何额外的优化。

对于覆盆子酱，我们很幸运。

树莓派社区运营的[piwheels.org](https://www.piwheels.org/)的[达夫·琼斯](https://github.com/waveform80)(`picamera`Python 模块的创建者)和[本·纳托尔](https://github.com/bennuttall)，一个为树莓派提供 ARM wheels(即预编译二进制包)的 Python 包仓库。

使用 PiWheels，您将能够在几秒钟内 pip 安装 OpenCV(对于其他需要很长时间编译的 Python 库也是如此，包括 NumPy、SciPy、scikit-learn 等。).

那么，如何指示 pip 命令使用 PiWheels 呢？

简短的回答是**“没什么！”**

如果您正在使用 Raspbian Stretch，您会很高兴地知道，pip 命令将在检查 PyPI 之前检查 PIWheels 的预编译二进制文件*，从而使您的 Pi 节省大量 CPU 周期(以及大量安装时间)。*

此外，当 Ben 和 Dave 为 PiWheels 编写 OpenCV 二进制文件时，他们问我应该使用哪些指令— [,我向他们推荐了针对 Raspberry Pi 的优化 OpenCV 安装程序](https://pyimagesearch.com/2017/10/09/optimizing-opencv-on-the-raspberry-pi/) —这正是他们遵循的指令！

如果您最终使用 pip 在您的 Raspberry Pi 上安装 OpenCV，请放心，您使用的是优化版本。

让我们开始学习如何在我们的 Raspberry Pi 上安装 OpenCV。

#### 在您的 Raspberry Pi 上安装先决条件

Raspberry Pi 要求您在开始之前安装几个系统包:

```py
$ sudo apt-get install libhdf5-dev libhdf5-serial-dev libhdf5-103
$ sudo apt-get install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
$ sudo apt-get install libatlas-base-dev
$ sudo apt-get install libjasper-dev

```

#### 在你的树莓派上安装 pip

Python 包管理器“pip”可以通过 wget 获得:

```py
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python3 get-pip.py

```

现在你有两个选择:

1.  在您的 Raspberry Pi 上安装 OpenCV 到您的全局 Python `site-packages`
2.  将 OpenCV 安装到您的 Raspberry Pi 上的虚拟环境中

#### 选项 A:用 pip 将 OpenCV 安装到您的 Raspberry Pi 系统中

如果您希望能够在隔离环境中使用不同版本的 OpenCV，我不会推荐这个选项。

但是很多人部署他们的 Raspberry Pis 仅仅是为了一个目的/项目，并不需要虚拟环境。

也就是说，如果你后来改变主意，想要使用虚拟环境，那么清理起来会很混乱，所以我建议跳过这个选项，遵循**选项 B** 。

要在您的 Raspberry Pi 系统上 pip 安装 OpenCV，请确保像这样使用 sudo:

```py
$ sudo pip install opencv-contrib-python==4.1.0.25

```

***2019-11-21 更新:**读者反映，通过 pip 安装的 OpenCV 4 的某些版本在树莓 Pi 上无法正常工作。如果您没有使用上面代码块中提到的 OpenCV 的特定版本，当您从 Python 中`import cv2`时，您可能会遇到`**"undefined symbol: __atomic_fetch_add8"**` for `libatomic`错误。*

几秒钟之内，OpenCV 就可以和你已经安装的其他包一起放入你的 Raspberry Pi 的站点包中。

#### 选项 B:在 Raspberry Pi 上用 pip 将 OpenCV 安装到虚拟环境中

如果您的 Raspberry Pi 有多种用途(或者如果您像我一样，一直在为博客帖子测试各种软件版本之间的代码兼容性),那么虚拟环境绝对是一个不错的选择。).

下面是如何安装 virtualenv 和 virtualenvwrapper，我用来完成它的工具:

```py
$ pip install virtualenv virtualenvwrapper

```

然后，您需要在您的`~/.bashrc`中添加以下几行。使用`nano`、`vim`或`emacs`打开文件，并将这些行附加到末尾:

```py
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.local/bin/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.local/bin/virtualenv
source $HOME/.local/bin/virtualenvwrapper.sh
```

保存文件。然后在您的终端中“搜索它”:

```py
$ source ~/.bashrc

```

将打印终端输出，指示 virtualenvwrapper 准备就绪。请务必检查它是否有错误。

您现在可以访问新的终端命令:

*   **用`mkvirtualenv`创造**的环境。
*   **用`workon`激活**一个环境(或**切换**到不同的环境)。
*   **用`deactivate`停用**一个环境。
*   **用`rmvirtualenv`去掉**一个环境。
*   *务必[阅读文件](https://virtualenvwrapper.readthedocs.io/en/latest/)！*

要**创建**一个 Python 3 虚拟环境，其中将包含 OpenCV 和您安装的其他包，只需使用 mkvirtualenv 和下面的命令:

```py
$ mkvirtualenv cv -p python3

```

现在您有了一个名为`cv`的虚拟环境。您可以通过以下方式随时激活它:

```py
$ workon cv

```

现在手腕一翻，你就可以 **pip 安装 OpenCV** 到`cv`:

```py
$ pip install opencv-contrib-python==4.1.0.25

```

***2019-11-21 更新:**读者反映，通过 pip 安装的 OpenCV 4 的某些版本在树莓 Pi 上无法正常工作。如果您没有使用上面代码块中提到的 OpenCV 的特定版本，当您从 Python 中`import cv2`时，您可能会遇到`**"undefined symbol: __atomic_fetch_add8"**` for `libatomic`错误。*

这就是 PiWheels 的全部内容！

我打赌你在用 **PiCamera** 作为你的成像传感器。您可以使用以下命令安装 Python 模块(注意引号):

```py
$ pip install "picamera[array]"

```

### 测试 OpenCV 的 pip 安装

你知道 OpenCV 的 3.3+有一个可以运行深度学习模型的 DNN 模块吗？

您可能会感到惊讶，但是您的 OpenCV 版本现在可以开箱即用，几乎不需要额外的软件。

我们将使用 MobileNet 单镜头检测器在视频中执行[对象检测。](https://pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/)

以下是您需要首先安装的内容(假设有一个`cv`虚拟环境):

```py
$ workon cv
$ pip install imutils
$ pip install "picamera[array]" # if you're using a Raspberry Pi

```

现在，打开一个 Python shell，仔细检查您是否准备好了所有软件:

```py
$ workon cv
$ python
Python 3.6.3 (default, Oct  4 2017, 06:09:15) 
[GCC 4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.37)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>> cv2.__version__
'4.0.1'
>>> import imutils
>>>

```

树莓派将展示不同版本的 Python 3，这是意料之中的。

现在是下载代码的时候了。

一定要用这篇博文的 ***【下载】*** 部分下载源代码+预先训练好的 MobileNet SSD 神经网络。

从那里，执行以下命令:

```py
$ python real_time_object_detection.py \
	--prototxt MobileNetSSD_deploy.prototxt.txt \
	--model MobileNetSSD_deploy.caffemodel
[INFO] loading model...
[INFO] starting video stream...
[INFO] elapsed time: 55.07
[INFO] approx. FPS: 6.54

```

[![](img/e2bc65acb4f49c9cb4bbfa6b2165bc4e.png)](https://pyimagesearch.com/wp-content/uploads/2017/09/real_time_object_detection_animation.gif)

**Figure 1:** A short clip of Real-time object detection with deep learning and OpenCV

我用的是 Macbook Pro。在笔记本电脑上使用 CPU 时，6 FPS 的帧速率已经相当不错了。

树莓 pi 是资源受限的，因此我们可以利用一些技巧来制造高 FPS 的假象。如果你在 **Raspberry Pi** 上，执行以下命令:

```py
$ python pi_object_detection.py \
	--prototxt MobileNetSSD_deploy.prototxt.txt \
	--model MobileNetSSD_deploy.caffemodel
[INFO] loading model...
[INFO] starting process...
[INFO] starting video stream...
[INFO] elapsed time: 48.55
[INFO] approx. FPS: 27.83

```

<https://www.youtube.com/embed/Ob_FrW7yuzw?feature=oembed>