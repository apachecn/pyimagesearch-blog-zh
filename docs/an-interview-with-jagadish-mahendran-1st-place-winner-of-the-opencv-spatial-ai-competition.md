# OpenCV 空间人工智能竞赛冠军 Jagadish Mahendran 访谈

> 原文：<https://pyimagesearch.com/2021/03/31/an-interview-with-jagadish-mahendran-1st-place-winner-of-the-opencv-spatial-ai-competition/>

在这篇文章中，我采访了高级计算机视觉/人工智能(AI)工程师 Jagadish Mahendran，他最近使用新的 OpenCV AI 套件(OAK)在 OpenCV 空间 AI 比赛中获得了第一名。

Jagadish 的获奖项目是一个为视障人士设计的计算机视觉系统，让用户能够成功地在 T2 和 T4 导航。他的项目包括:

*   人行横道自动检测
*   人行横道和停车标志检测
*   悬垂障碍物检测
*   *…还有更多！*

最重要的是，整个项目是围绕新的 OpenCV 人工智能套件(OAK)构建的，这是一种专为计算机视觉设计的嵌入式设备。

和我一起了解 Jagadish 的项目，以及他如何使用计算机视觉来帮助视障人士。

## **采访 OpenCV 空间人工智能竞赛冠军 Jagadish Mahendran**

阿德里安:欢迎你，贾加迪什！非常感谢你能来。很高兴您能来到 PyImageSearch 博客。

阿德里安，很高兴接受你的采访。谢谢你邀请我。

* * *

阿德里安:在我们开始之前，你能简单介绍一下你自己吗？你在哪里工作，你在那里的角色是什么？

**Jagadish:** 我是一名高级计算机视觉/人工智能(AI)工程师。我曾为多家创业公司工作，在那里我为库存管理机器人和烹饪机器人构建了 AI 和感知解决方案。

* * *

阿德里安:你最初是如何对计算机视觉和机器人感兴趣的？

**Jagadish:** 我从本科开始就对 AI 感兴趣，在那里我有机会和朋友一起制作了一个微型鼠标机器人。我在硕士期间被计算机视觉和机器学习吸引住了。从那以后，与这些令人惊奇的技术一起工作变得非常有趣。

* * *

**Adrian:** 您最近在 OpenCV 空间人工智能竞赛中获得了第一名，祝贺您！你能给我们更多关于比赛的细节吗？有多少支队伍参加了比赛，比赛的最终目标是什么？

**Jagadish:** 谢谢。由英特尔赞助的 OpenCV Spatial AI 2020 竞赛分为两个阶段。包括大学实验室和公司在内的约 235 个具有各种背景的团队参与了第一阶段，其中涉及提出一个使用带有深度(OAK-D)传感器的 OpenCV AI 套件解决现实世界问题的想法。31 个团队被选入第二阶段，我们必须用 3 个月的时间来实现我们的想法。最终目标是开发一个使用 OAK-D 传感器的全功能人工智能系统。

* * *

**Adrian:** 你的获奖解决方案是一个为视障人士设计的视觉系统。你能告诉我们更多关于你的项目吗？

**Jagadish:** 文献上甚至市场上都有各种视觉辅助系统。由于硬件限制、成本和其他挑战，他们中的大多数人不使用深度学习方法。但最近，在 edge AI 和传感器空间方面有了显著的改善，我认为这可以为硬件有限的视觉辅助系统提供深度学习支持。

我开发了一个可穿戴视觉辅助系统，使用 OAK-D 传感器进行感知，使用外部神经计算棒(NCS2)和我 5 岁的笔记本电脑进行计算。该系统可以执行各种计算机视觉任务，帮助视障人士理解场景。

这些任务包括:探测障碍物；海拔变化；了解道路、人行道和交通状况。

该系统可以检测交通标志以及许多其他类别，如人、汽车、自行车等。该系统还可以使用点云检测障碍物，并使用语音界面更新个人关于其存在的信息。个人也可以使用语音识别系统与系统交互。

以下是一些输出示例:

* * *

阿德里安:请告诉我们用于开发项目提交材料的硬件。个人是否需要佩戴大量笨重的硬件和设备？

**Jagadish:** 我采访了一些视障人士，了解到走在街上受到太多关注是视障人士面临的主要问题之一。因此，物理系统作为辅助设备不引人注目是一个主要目标。开发的系统很简单——物理设置包括我 5 岁的笔记本电脑、2 个神经计算棒、藏在棉背心内的摄像头、GPS，如果需要，还可以在腰包/腰包内放置一个额外的摄像头。大多数这些设备都很好地包装在背包里。总的来说，它看起来像一个穿着背心到处走的大学生。我在我的商业区走来走去，完全没有引起特别的注意。

* * *

**Adrian:** 你为什么选择 OpenCV AI Kit (OAK)，更确切的说是可以计算深度信息的 OAK-D 模块？

**Jagadish:** 主办方提供 OAK-D 作为比赛的一部分，它有很多好处。它很小。除了 RGB 图像，它还可以提供深度图像。这些深度图像对于探测障碍物非常有用，即使不知道障碍物是什么。此外，它还有一个片上人工智能处理器，这意味着计算机视觉任务在帧到达主机之前就已经执行了。这使得系统超快。

* * *

阿德里安:你有针对视觉障碍的视觉系统的演示吗？

**Jagadish:** 演示可以在这里找到:

<https://www.youtube.com/embed/ui_p5x8n2tA?feature=oembed>