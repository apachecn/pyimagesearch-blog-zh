# 利用 OpenCV 进行实例分割

> 原文：<https://pyimagesearch.com/2018/11/26/instance-segmentation-with-opencv/>

![](img/44a5a9e3cc40f46e6747f00bcc5bfca4.png)

在本教程中，您将学习如何使用 OpenCV、Python 和深度学习来执行实例分割。

早在 9 月份，我看到微软在他们的 Office 365 平台上发布了一个非常棒的功能——**进行视频电话会议的能力，*模糊背景*，让你的同事只能看到你(而看不到你身后的任何东西)。**

这篇文章顶部的 GIF 展示了一个类似的特性，这是我为了今天的教程而实现的。

无论你是在酒店房间接电话，还是在丑陋不堪的办公楼里工作，或者只是不想清理家庭办公室，电话会议模糊功能都可以让与会者专注于你(而不是背景中的混乱)。

对于在家工作并希望保护家庭成员隐私的人来说，这样的功能会特别有帮助(T2)。

想象一下，您的工作站可以清楚地看到您的厨房，您不会希望您的同事看到您的孩子吃晚饭或做作业吧！相反，只需打开模糊功能，一切就搞定了。

为了构建这样一个功能，微软利用了计算机视觉、深度学习，最值得注意的是， ***实例分割。***

我们在上周的博客文章中[介绍了 Mask R-CNN 的实例分割，今天我们将采用我们的 Mask R-CNN 实现，并使用它来构建一个类似 Microsoft Office 365 的视频模糊功能。](https://pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/)

**要了解如何使用 OpenCV 执行实例分割，*继续阅读！***

## 利用 OpenCV 进行实例分割

<https://www.youtube.com/embed/puSN8Dg-bdI?feature=oembed>