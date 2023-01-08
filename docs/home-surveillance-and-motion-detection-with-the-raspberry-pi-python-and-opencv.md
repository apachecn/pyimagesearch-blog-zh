# 利用 Raspberry Pi、Python、OpenCV 和 Dropbox 实现家庭监控和运动检测

> 原文：<https://pyimagesearch.com/2015/06/01/home-surveillance-and-motion-detection-with-the-raspberry-pi-python-and-opencv/>

[![pi_home_surveillance_animated](img/73adc3f5e59de462eef98592a7564734.png)](https://pyimagesearch.com/wp-content/uploads/2015/05/pi_home_surveillance_animated.gif)

哇，上周关于[构建基本运动检测系统的博文](https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/)居然是 ***牛逼*** *。写这本书很有趣，我从像你一样的读者那里得到的反馈让我觉得很值得努力去收集。*

对于那些刚刚开始调整的人来说，上周关于使用计算机视觉构建运动检测系统的帖子是由我的朋友詹姆斯**溜进我的冰箱**和**偷了我最后一瓶令人垂涎的啤酒**激发的。虽然我不能*证明*是他，但我想看看是否有可能使用计算机视觉和树莓酱来*当场抓住他*如果他再次试图偷我的啤酒。

正如你在这篇文章的结尾所看到的，我们将要构建的家庭监控和运动检测系统不仅是****简单的*** ，而且是 ***非常强大的*** 来实现这个特定的目标。*

 *今天，我们将扩展我们的基本运动检测方法，并:

1.  让我们的运动检测系统**更加鲁棒**，这样它就可以**全天连续运行****而不受光照条件变化的影响。**
***   更新我们的代码，这样我们的家庭监控系统就可以在树莓派上运行了。*   **与 Dropbox API** 集成，这样我们的 Python 脚本就可以*自动*将安全照片上传到我们的个人 Dropbox 账户。**

 **在这篇文章中，我们将会看到很多代码，所以请做好准备。但是我们会学到很多东西。更重要的是，在这篇文章结束时，你将拥有一个自己的可以工作的 Raspberry Pi 家庭监控系统。

你可以直接在下面找到完整的演示视频，以及这篇文章底部的一堆其他例子。

**更新:2017 年 8 月 24 日—** 这篇博文中的所有代码都已更新，可以与 Dropbox V2 API 配合使用，因此您不再需要复制和粘贴视频中使用的验证密钥。更多细节请见这篇博文的剩余部分。

<https://www.youtube.com/embed/BhD1aDEV-kg?feature=oembed>***