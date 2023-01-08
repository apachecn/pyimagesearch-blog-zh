# 使用 OpenCV 和 ImageZMQ 通过网络进行实时视频流传输

> 原文：<https://pyimagesearch.com/2019/04/15/live-video-streaming-over-network-with-opencv-and-imagezmq/>

![](img/f06064f44bf01eafe534197047585248.png)

在今天的教程中，您将学习如何使用 OpenCV 通过网络传输实时视频。**具体来说，您将学习如何实现 Python + OpenCV 脚本来*从摄像机捕获*视频帧并将其传输到服务器。**

大约每周，我都会收到一篇博客帖子的评论或电子邮件中的一个问题，大概是这样的:

> 你好，Adrian，我正在做一个项目，我需要使用 OpenCV 将帧从客户端摄像头传输到服务器进行处理。我应该使用 IP 摄像头吗？树莓派有用吗？RTSP 流媒体怎么样？你试过用 FFMPEG 或者 GStreamer 吗？你建议我如何处理这个问题？

这是一个很好的问题——如果你曾经尝试过用 OpenCV 进行视频直播，你就会知道有很多不同的选择。

你可以走 IP 摄像头路线。但 IP 摄像头可能是一个痛苦的工作。一些 IP 摄像机甚至不允许你访问 RTSP(实时流协议)流。其他 IP 摄像头根本无法与 OpenCV 的`cv2.VideoCapture`功能配合使用。IP 相机可能太贵了，超出了你的预算。

在这些情况下，你只能使用标准的网络摄像头— **问题就变成了，** ***你如何使用 OpenCV 从网络摄像头传输帧？***

使用 FFMPEG 或 GStreamer 绝对是一个选择。但是这两者都是工作中的痛苦。

今天我将向大家展示我的首选解决方案，使用**消息传递库**，具体来说就是 ZMQ 和 ImageZMQ，后者是由 PyImageConf 2018 演讲者 Jeff Bass 开发的。Jeff 为 ImageZMQ 投入了大量*的工作，他的努力确实得到了体现。*

正如你将看到的，OpenCV 视频流的这种方法不仅*可靠*而且*非常容易使用*，只需要几行代码。

**要了解如何使用 OpenCV 执行实时网络视频流，*请继续阅读！***

## 使用 OpenCV 和 ImageZMQ 通过网络进行实时视频流传输

<https://www.youtube.com/embed/tvzWNh094V0?feature=oembed>