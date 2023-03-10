# 生成对抗网络简介

> 原文：<https://pyimagesearch.com/2021/09/13/intro-to-generative-adversarial-networks-gans/>

这篇文章涵盖了高层次的生成对抗网络(GAN)的直觉，各种 GAN 变体，以及解决现实世界问题的应用。

这是 GAN 教程系列的第一篇文章:

1.  *生成对抗网络(GANs)简介*(本帖)
2.  [*入门:DCGAN for Fashion-MNIST*](https://pyimagesearch.com/2021/11/11/get-started-dcgan-for-fashion-mnist/)
3.  [*GAN 训练挑战:针对彩色图像的 DCGAN*](https://pyimagesearch.com/2021/12/13/gan-training-challenges-dcgan-for-color-images/)

## 【GANs 如何工作

GANs 是一种**生成**模型，它观察许多样本分布，生成更多相同分布的样本。其他生成模型包括变分自动编码器()和[自回归](https://arxiv.org/abs/1601.06759)模型。

### **氮化镓架构**

在基本的 GAN 架构中有两个网络:发生器模型和鉴别器模型。GANs 得名于单词**“对抗性的”**，因为这两个网络同时接受训练并相互竞争，就像在国际象棋这样的零和游戏中一样。

**生成器**模型生成新图像。生成器的目标是生成看起来如此真实的图像，以至于骗过鉴别器。在用于图像合成的最简单 GAN 架构中，输入通常是随机噪声，输出是生成的图像。

**鉴别器**只是一个你应该已经熟悉的二值图像分类器。它的工作是分类一幅图像是真是假。

***注:*** *在更复杂的 GANs 中，我们可以用图像或文本作为鉴别器的条件，用于图像到图像的翻译或文本到图像的生成。*

将所有这些放在一起，下面是一个基本的 GAN 架构的样子:生成器生成假图像；我们将真实图像(训练数据集)和伪图像分批次输入鉴别器。鉴别器然后判断一幅图像是真的还是假的。

### **训练甘斯**

#### 极大极小游戏:G 对 D

大多数深度学习模型(例如，图像分类)都基于优化:找到成本函数的低值。gan 是不同的，因为两个网络:生成器和鉴别器，每个都有自己的成本和相反的目标:

*   生成器试图欺骗鉴别者，让他们认为假图像是真的
*   鉴别器试图正确地分类真实和伪造的图像

下面的最小最大游戏数学函数说明了训练中的这种对抗性动态。如果你不理解数学，不要太担心，我会在未来的 DCGAN 帖子中对 G 损失和 D 损失进行编码时进行更详细的解释。

发生器和鉴别器在训练期间都随着时间而改进。生成器在产生类似训练数据的图像方面变得越来越好，而鉴别器在区分真假图像方面变得越来越好。

训练 GANs 是为了在游戏中找到一个**平衡**当:

*   生成器生成的数据看起来几乎与训练数据相同。
*   鉴别器不再能够区分伪图像和真实图像。

#### **艺术家与批评家**

模仿杰作是学习艺术的一个好方法——“艺术家如何在世界知名的博物馆里临摹杰作”作为一个模仿杰作的人类艺术家，我会找到我喜欢的艺术品作为灵感，并尽可能多地复制它:轮廓、颜色、构图和笔触，等等。然后一位评论家看了看复制品，告诉我它是否像真正的杰作。

GANs 培训与此过程类似。我们可以把生成者想象成艺术家，把鉴别者想象成批评家。请注意人类艺术家和机器(GANs)艺术家之间的类比差异:生成器无法访问或看到它试图复制的杰作。相反，它只依靠鉴别器的反馈来改善它生成的图像。

### **评估指标**

好的 GAN 模型应该具有好的图像质量**——例如，不模糊并且类似于训练图像；以及**多样性**:生成了各种各样的图像，这些图像近似于训练数据集的分布。**

 **要评估 GAN 模型，您可以在训练期间直观地检查生成的图像，或者通过生成器模型进行推断。如果您想对您的 GANs 进行定量评估，这里有两个流行的评估指标:

1.  **Inception Score，**捕获生成图像的 ***质量*** 和 ***多样性***
2.  **弗雷歇初始距离**比较真实图像和虚假图像，而不仅仅是孤立地评估生成的图像

## **GAN 变体**

自 [Ian Goodfellow 等人在 2014 年](https://arxiv.org/abs/1406.2661)发表最初的 GANs 论文以来，出现了许多 GAN 变体。它们倾向于相互建立，要么解决特定的培训问题，要么创建新的 GANs 架构，以便更好地控制 GANs 或获得更好的图像。

以下是一些具有突破性的变体，为未来 GAN 的发展奠定了基础。这绝不是所有 GAN 变体的完整列表。

[**DCGAN**](https://arxiv.org/abs/1511.06434) (深度卷积生成对抗网络的无监督表示学习)是第一个在其网络架构中使用卷积神经网络(CNN)的 GAN 提案。今天的大多数 GAN 变体多少都是基于 DCGAN 的。因此，DCGAN 很可能是你的第一个 GAN 教程，学习 GAN 的“Hello-World”。

[**WGAN**](https://arxiv.org/abs/1701.07875)(wasser stein GAN)和 [**WGAN-GP**](https://arxiv.org/abs/1704.00028v3) (被创建来解决 GAN 训练挑战，例如模式崩溃——当生成器重复产生相同的图像或(训练图像的)一个小的子集。WGAN-GP 通过使用梯度惩罚而不是训练稳定性的权重削减来改进 WGAN。

[**cGAN**](https://arxiv.org/abs/1411.1784) (条件生成对抗网)首先引入了基于条件生成图像的概念，条件可以是图像类标签、图像或文本，如在更复杂的 GANs 中。 **Pix2Pix** 和 **CycleGAN** 都是条件 GAN，使用图像作为图像到图像转换的条件。

[**pix 2 pixhd**](https://github.com/NVIDIA/pix2pixHD)****利用条件 GANs 进行高分辨率图像合成和语义操纵】理清多种输入条件的影响，如论文示例中所述:为服装设计控制生成的服装图像的颜色、纹理和形状。此外，它还可以生成逼真的 2k 高分辨率图像。****

 ****[**萨根**](https://arxiv.org/abs/1805.08318) (自我注意生成对抗网络)提高图像合成质量:通过将自我注意模块(来自 NLP 模型的概念)应用于 CNN，使用来自所有特征位置的线索生成细节。谷歌 DeepMind 扩大了 SAGAN 的规模，以制造 BigGAN。

[**BigGAN**](https://arxiv.org/abs/1809.11096) (高保真自然图像合成的大规模 GAN 训练)可以创建高分辨率和高保真的图像。

ProGAN、StyleGAN 和 StyleGAN2 都能创建高分辨率图像。

[**ProGAN**](https://arxiv.org/abs/1710.10196) (为提高质量、稳定性和变化性而进行的 GANs 渐进增长)使网络渐进增长。

NVIDIA Research 推出的 [**StyleGAN**](https://arxiv.org/abs/1812.04948) (一种基于风格的生成式对抗网络生成器架构)，使用带有自适应实例规范化(AdaIN)的 ProGAN plus 图像风格转移，能够控制生成图像的风格。

[**【StyleGAN 2】**](https://arxiv.org/abs/1912.04958)(StyleGAN 的图像质量分析与改进)在原始 StyleGAN 的基础上，在归一化、渐进生长和正则化技术等方面进行了多项改进。

## **氮化镓应用**

gan 用途广泛，可用于多种应用。

### **图像合成**

图像合成可能很有趣，并提供实际用途，如机器学习(ML)培训中的图像增强或帮助创建艺术品和设计资产。

GANs 可以用来创造在之前从未存在过的图像**，这也许是 GANs 最出名的地方。他们可以创造看不见的新面孔，猫的形象和艺术品，等等。我在下面附上了几张高保真图片，它们是我从 StyleGAN2 支持的网站上生成的。去这些链接，自己实验，看看你从实验中得到了什么图像。**

Zalando Research 使用 GANs 生成基于颜色、形状和纹理的时装设计([解开 GANs](https://arxiv.org/abs/1806.07819) 中的多个条件输入)。

脸书研究公司的 Fashion++超越了创造时尚的范畴，提出了改变时尚的建议:“什么是时尚？”

GANs 还可以帮助训练强化剂。比如英伟达的 [GameGAN](https://arxiv.org/abs/2005.12126) 模拟游戏环境。

### **图像到图像的翻译**

图像到图像转换是一项计算机视觉任务，它将输入图像转换到另一个领域(例如，颜色或风格)，同时保留原始图像内容。这也许是在艺术和设计中使用 GANs 的最重要的任务之一。

Pix2Pix (使用条件对抗网络的图像到图像翻译)是一个条件 GAN，它可能是最著名的图像到图像翻译 GAN。然而，Pix2Pix 的一个主要缺点是它需要成对的训练图像数据集。

CycleGAN 基于 Pix2Pix 构建，只需要不成对的图像，在现实世界中更容易获得。它可以把苹果的图像转换成橘子，把白天转换成黑夜，把马转换成斑马……好的。这些可能不是真实世界的用例；从那时起，艺术和设计领域出现了许多其他的图像到图像的 GANs。

现在你可以把自己的自拍翻译成漫画、绘画、漫画，或者任何你能想象到的其他风格。例如，我可以使用[白盒卡通](https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/paper/06791.pdf)将我的自拍变成卡通版本:

**彩色化**不仅可以应用于黑白照片，还可以应用于艺术作品或设计资产。在艺术品制作或 UI/UX 设计过程中，我们从轮廓开始，然后着色。自动上色有助于为艺术家和设计师提供灵感。

### **文本到图像**

我们已经看到了 GANs 的很多图像到图像的翻译例子。我们还可以使用单词作为条件来生成图像，这比使用类别标签作为条件要灵活和直观得多。

近年来，自然语言处理和计算机视觉的结合已经成为一个热门的研究领域。这里有几个例子: [StyleCLIP](https://github.com/orpatashnik/StyleCLIP) 和[驯服高分辨率图像合成的变形金刚](https://arxiv.org/abs/2012.09841v3)。

### **超越影像**

GANs 不仅可以用于图像，还可以用于音乐和视频。比如 Magenta 项目的 [GANSynth](https://magenta.tensorflow.org/gansynth) 会做音乐。这里有一个有趣的视频动作转移的例子，叫做“现在每个人都跳舞”( [YouTube](https://youtu.be/PCBTZh41Ris) | [Paper](https://arxiv.org/abs/1808.07371) )。我一直喜欢看这个迷人的视频，专业舞者的舞步被转移到业余舞者身上。

<https://www.youtube.com/embed/PCBTZh41Ris?feature=oembed>******