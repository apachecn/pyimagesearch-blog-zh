# 在系统上安装 Tesseract、PyTesseract 和 Python OCR 包

> 原文：<https://pyimagesearch.com/2021/08/16/installing-tesseract-pytesseract-and-python-ocr-packages-on-your-system/>

在本教程中，我们将*配置*我们的 OCR 开发环境。一旦您的机器配置完毕，我们将开始编写执行 OCR 的 Python 代码，为您开发自己的 OCR 应用程序铺平道路。

**要了解如何配置你的开发环境，** ***继续阅读。***

## **学习目标**

在本教程中，您将:

1.  了解如何在您的计算机上安装 Tesseract OCR 引擎
2.  了解如何创建 Python 虚拟环境(Python 开发中的最佳实践)
3.  安装运行本教程中的示例所需的必要 Python 包(并开发您自己的 OCR 项目)

## **OCR 开发环境配置**

在本教程的第一部分，您将学习如何在您的系统上安装 Tesseract OCR 引擎。从这里，您将学习如何创建一个 Python 虚拟环境，然后安装 OpenCV、PyTesseract 和 OCR、计算机视觉和深度学习所需的所有其他必要的 Python 库。

### **安装说明注释**

宇宙魔方 OCR 引擎已经存在了 30 多年。Tesseract OCR 的安装说明相当稳定。因此，我已经包括了这些步骤。

也就是说，让我们在您的系统上安装 Tesseract OCR 引擎！

### **安装宇宙魔方**

在本教程中，你将学习如何在你的机器上安装宇宙魔方。

#### **在 macOS 上安装宇宙魔方**

如果您使用[家酿](https://brew.sh)包管理器，在 macOS 上安装 Tesseract OCR 引擎相当简单。

如果您的系统上尚未安装 Homebrew，请使用上面的链接进行安装。

从那里，你需要做的就是使用`brew`命令来安装宇宙魔方:

```py
 $ brew install tesseract
```

如果上面的命令没有出现错误，那么您现在应该已经在 macOS 机器上安装了 Tesseract。

#### **在 Ubuntu 上安装宇宙魔方**

在 Ubuntu 18.04 上安装 Tesseract 很容易——我们需要做的就是利用`apt-get`:

```py
 $ sudo apt install tesseract-ocr
```

`apt-get`包管理器将自动安装宇宙魔方所需的任何必备库或包。

#### **在 Windows 上安装宇宙魔方**

请注意，PyImageSearch 团队和我*并不*正式支持 Windows，除了使用我们预配置的 Jupyter/Colab 笔记本的客户，这些客户可以在 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)找到。这些笔记本电脑可以在所有环境下运行，包括 macOS、Linux 和 Windows。

相反，我们建议使用基于 Unix 的机器，如 Linux/Ubuntu 或 macOS ，这两种机器都更适合开发计算机视觉、深度学习和 OCR 项目。

也就是说，如果你想在 Windows 上安装宇宙魔方，我们建议你遵循官方的 Windows 安装说明，这些说明是由[宇宙魔方团队](https://github.com/tesseract-ocr/tessdoc)提供的。

#### **验证您的宇宙魔方安装**

假设您能够在您的操作系统上安装 Tesseract，您可以使用`tesseract`命令验证 Tesseract 是否已安装:

```py
 $ tesseract -v
 tesseract 4.1.1
  leptonica-1.79.0
   libgif 5.2.1 : libjpeg 9d : libpng 1.6.37 : libtiff 4.1.0 : zlib 1.2.11 : libwebp 1.1.0 : libopenjp2 2.3.1
  Found AVX2
  Found AVX
  Found FMA
  Found SSE
```

您的输出应该与我的相似。

### **为 OCR 创建 Python 虚拟环境**

Python 虚拟环境是 Python 开发的最佳实践，我们建议使用它们来获得更可靠的开发环境。

在我们的[*pip Install OpenCV*](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)*教程中可以找到为 Python 虚拟环境安装必要的包，以及创建您的第一个 Python 虚拟环境。我们建议您按照该教程创建您的第一个 Python 虚拟环境。*

 *### **安装 OpenCV 和 PyTesseract】**

既然您已经创建了 Python 虚拟环境并做好了准备，我们可以安装 OpenCV 和 PyTesseract，这是与 Tesseract OCR 引擎接口的 Python 包。

这两者都可以使用以下命令进行安装:

```py
 $ workon <name_of_your_env> # required if using virtual envs
 $ pip install numpy opencv-contrib-python
 $ pip install pytesseract
```

接下来，我们将安装 OCR、计算机视觉、深度学习和机器学习所需的其他 Python 包。

### **安装其他计算机视觉、深度学习和机器学习库**

现在让我们安装一些其他支持计算机视觉和机器学习/深度学习的软件包，我们将在本教程的剩余部分中用到它们:

```py
 $ pip install pillow scipy
 $ pip install scikit-learn scikit-image
 $ pip install imutils matplotlib
 $ pip install requests beautifulsoup4
 $ pip install h5py tensorflow textblob
```

## **总结**

在本教程中，您学习了如何在您的计算机上安装 Tesseract OCR 引擎。您还学习了如何安装执行 OCR、计算机视觉和图像处理所需的 Python 包。

现在您的开发环境已经配置好了，我们将在下一个教程中编写 OCR 代码！*