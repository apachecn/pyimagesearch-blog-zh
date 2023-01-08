# 使用 OpenCV 写入视频

> 原文：<https://pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/>

[![writing_to_video_example_02](img/b4414a512042f05bdb205755b141f753.png)](https://pyimagesearch.com/wp-content/uploads/2016/02/writing_to_video_example_02.jpg)

让我先说用 OpenCV 写视频是一件非常痛苦的事情。

本教程的目的是**帮助你开始用 OpenCV 3** 编写视频文件，提供(并解释)一些样板代码，并详细说明我是如何在自己的系统上编写视频的。

但是，如果您试图在自己的应用程序中使用 OpenCV 编写视频文件，请做好准备:

1.  研究安装在您系统上的视频编解码器。
2.  玩各种编解码器+文件扩展名，直到视频成功写入磁盘。
3.  确保你在一个远离孩子的僻静地方——那里会有很多咒骂和诅咒。

你看，虽然用于用 OpenCV 创建视频文件的函数如`cv2.VideoWriter`、`cv2.VideoWriter_fourcc`和`cv2.cv.FOURCC`在中有很好的记录，但是*没有*有很好的记录的是成功编写视频文件所需的编解码器+文件扩展名的组合*、**、**、*。

我需要创建一个用 OpenCV 编写视频文件的应用程序已经有很长时间了，所以当我坐下来为这篇博文编写代码时，我对我花了这么长时间编写这个例子感到非常惊讶(也非常沮丧)。

事实上，我是*唯一*能够使用 ***OpenCV 3 的代码！*** 这篇文章中详细描述的代码*与 OpenCV 2.4.X 不兼容*(尽管我已经强调了在 OpenCV 2.4 上运行所需的代码更改，如果你想试一试的话)。

***注意:**如果您需要在您的系统上安装 OpenCV 的帮助，[请查阅此页](https://pyimagesearch.com/opencv-tutorials-resources-guides/)以获得各种平台的安装说明列表。另外，一定要看看* [实用 Python 和 OpenCV](https://pyimagesearch.com/practical-python-opencv/) *的 Quickstart 包和硬拷贝包，其中包括一个可下载的 Ubuntu VirtualBox 虚拟机，预配置和预安装了 Open 3。*

无论如何，在这篇文章的剩余部分，我将演示如何使用 OpenCV 将视频写入文件。希望您能够在自己的应用程序中使用这些代码，而不会掉太多头发(我已经秃顶了，所以我没有这个问题)。

<https://www.youtube.com/embed/HRXwqx1ep3M?feature=oembed>