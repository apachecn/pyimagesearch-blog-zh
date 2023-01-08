# 用 OpenCV 进行神经类型转换

> 原文：<https://pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/>

![](img/2aff3332f3271a738c0d86784baa733d.png)

在本教程中，您将学习如何使用 OpenCV、Python 和深度学习将神经风格转换应用于图像和实时视频。在本指南结束时，你将能够通过神经风格转移生成美丽的艺术作品。

最初的神经风格转移算法是 Gatys 等人在他们 2015 年的论文 [*中介绍的一种艺术风格*](https://arxiv.org/abs/1508.06576) 的神经算法(其实这正是我在 [*里面教你如何用 Python*](https://pyimagesearch.com/deep-learning-computer-vision-python-book/) 实现和训练用于计算机视觉的深度学习的确切算法)。

2016 年，Johnson 等人发表了用于实时风格转移的 [*感知损失和超分辨率*](https://cs.stanford.edu/people/jcjohns/eccv16/) ，利用感知损失将神经风格转移框架化为类超分辨率问题。最终的结果是一个神经风格转换算法，它比 Gatys 等人的方法快了三个数量级*(虽然有一些缺点，我将在指南的后面讨论它们)。*

在这篇文章的其余部分，你将学习如何将神经风格转换算法应用到你自己的图像和视频流中。

**要学习如何使用 OpenCV 和 Python 应用神经风格转移，*继续阅读！***

## 用 OpenCV 进行神经类型转换

<https://www.youtube.com/embed/DRpydtvjGdE?feature=oembed>