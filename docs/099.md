# 用 OpenCV 进行活性检测

> 原文：<https://pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/>

![](img/db312187f0ce959f0e8e86ae1d90ed6e.png)

在本教程中，您将学习如何使用 OpenCV 执行活性检测。您将创建一个能够在人脸识别系统中识别假脸并执行反人脸欺骗的活体检测器。

在过去的一年里，我创作了许多人脸识别教程，包括:

*   [*OpenCV 人脸识别*](https://pyimagesearch.com/2018/09/24/opencv-face-recognition/)
*   [*人脸识别用 dlib、Python 和深度学习*](https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
*   *[树莓派人脸识别](https://pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/)*

然而，我在电子邮件和面部识别帖子的评论部分被问到的一个常见问题是:

> **我如何辨别真实的和*假的*面孔？**

想想如果一个邪恶的用户试图*故意绕过你的面部识别系统*会发生什么。

这样的用户可以尝试拿起另一个人的照片。也许他们甚至**在他们的智能手机上有一张照片或视频，他们可以举到负责执行面部识别的相机**(比如在这篇文章顶部的图片中)。

在这种情况下，面对摄像头的人脸完全有可能被正确识别…但最终会导致未经授权的用户绕过您的人脸识别系统！

你将如何辨别这些“假”和“真/合法”的面孔？如何将反面部欺骗算法应用到面部识别应用程序中？

答案是将 ***活性检测*** 应用于 OpenCV，这正是我今天要讨论的内容。

**要了解如何将 OpenCV 的活性检测融入您自己的人脸识别系统，*请继续阅读！***

## 用 OpenCV 进行活性检测

***2020-06-11 更新:**此博文现已兼容 TensorFlow 2+!*

<https://www.youtube.com/embed/MPedzm6uOMA?feature=oembed>