# 用 OpenCV 和 Python 生成电影条形码

> 原文：<https://pyimagesearch.com/2017/01/16/generating-movie-barcodes-with-opencv-and-python/>

![movie_barcode_header](img/d279d30f96d25f12d8259f45cefbb360.png)

在上周的博文中，我演示了如何计算视频文件中的帧数。

今天，我们将使用这些知识来帮助我们完成一项计算机视觉和图像处理任务— *可视化电影条形码*，类似于本文顶部的任务。

几年前，我第一次意识到电影条形码，是从这个软件开始的，这个软件被用来为 2013 年 T2 布鲁克林电影节制作海报和预告片。

自从我开始 PyImageSearch 以来，我已经收到了一些关于生成电影条形码的电子邮件，有一段时间我通常不涉及*可视化方法*，我最终决定就此写一篇博文。毕竟这是一个非常巧妙的技术！

在本教程的剩余部分，我将演示如何编写自己的 Python + OpenCV 应用程序来生成自己的电影条形码。

## 用 OpenCV 和 Python 生成电影条形码

为了构建电影条形码，我们需要完成三项任务:

*   任务#1:确定视频文件中的帧数。计算电影中的总帧数可以让我们知道在电影条形码可视化中我们应该有多少帧*包括*。太多的框架和我们的条形码将*巨大*；太少的帧和电影条形码将是不美观的。
*   任务#2:生成电影条形码数据。一旦我们知道了我们想要包含在电影条形码中的视频帧的总数，我们就可以循环每第 *N* 帧并计算 RGB 平均值，同时维护一个平均值列表。这是我们实际的电影条形码数据。
*   任务#3:显示电影条形码。给定一组帧的 RGB 平均值列表，我们可以获取这些数据并创建显示在屏幕上的实际的*电影条形码可视化*。

这篇文章的其余部分将演示如何完成这些任务。

### 电影条形码项目结构

在我们深入本教程之前，让我们先来讨论一下我们的项目/目录结构，具体如下:

```py
|--- output/
|--- videos/
|--- count_frames.py
|--- generate_barcode.py
|--- visualize_barcode.py

```

`output`目录将存储我们实际的电影条形码(生成的电影条形码图像和序列化的 RGB 平均值)。

然后我们有了`videos`文件夹，我们的输入视频文件就存放在这个文件夹中。

最后，我们需要三个助手脚本:`count_frames.py`、`generate_barcode.py`和`visualize_barcode.py`。我们将在接下来的章节中讨论每一个 Python 文件。

### 安装先决条件

我假设您的系统上已经安装了 OpenCV(如果没有，请参考本页中的[，我在这里提供了在各种不同平台上安装 OpenCV 的教程)。](https://pyimagesearch.com/opencv-tutorials-resources-guides/)

除了 OpenCV，你还需要 [scikit-image](http://scikit-image.org/) 和 [imutils](https://github.com/jrosebr1/imutils) 。您可以使用`pip`安装两者:

```py
$ pip install --upgrade scikit-image imutils

```

现在花点时间安装/升级这些包，因为我们在本教程的后面会用到它们。

### 计算视频中的帧数

在上周的博文中，我讨论了如何[(有效地)确定视频文件](https://pyimagesearch.com/2017/01/09/count-the-total-number-of-frames-in-a-video-with-opencv-and-python/)中的帧数。因为我已经深入讨论了这个主题，所以今天我不打算提供完整的代码概述。

也就是说，你可以在下面找到`count_frames.py`的源代码:

```py
# import the necessary packages
from imutils.video import count_frames
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-o", "--override", type=int, default=-1,
	help="whether to force manual frame count")
args = vars(ap.parse_args())

# count the total number of frames in the video file
override = False if args["override"] < 0 else True
total = count_frames(args["video"], override=override)

# display the frame count to the terminal
print("[INFO] {:,} total frames read from {}".format(total,
	args["video"][args["video"].rfind(os.path.sep) + 1:]))

```

顾名思义，这个脚本只计算视频文件中的帧数。

举个例子，让我们看看我最喜欢的电影《侏罗纪公园》的预告片:

<https://www.youtube.com/embed/lc0UehYemQA?feature=oembed>