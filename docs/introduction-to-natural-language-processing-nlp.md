# 自然语言处理(NLP)简介

> 原文：<https://pyimagesearch.com/2022/06/27/introduction-to-natural-language-processing-nlp/>

* * *

## **目录**

*   [自然语言处理(NLP)简介](#h2NLP)

*   [自然语言处理](#h3NLP)
*   [前言](#h3Preface)
*   自然语言处理的开端
*   [自然语言处理找到立足点](#h3Footing)
*   [NLP 的崛起](#h3Rise)

*   [总结](#h2Summary)

*   [引用信息](#h3Citation)

* * *

## [**自然语言处理入门**](#TOC)

 **很多时候，人脑的聪明是理所当然的。每天，我们都要处理无数的信息，这些信息都属于不同的领域。

我们用眼睛看事物。我们把看到的物体分成不同的组。我们在工作中应用数学公式。甚至我们的交流模式也需要大脑处理信息。

所有这些任务都在不到一秒钟的时间内完成。人工智能的最终目标一直是再造大脑。但是现在，我们受到了一些限制，比如计算能力和数据。

制造能同时完成多项任务的机器极其困难。因此，我们将问题分类，并将其主要分为计算机视觉和自然语言处理。

我们已经变得擅长用图像数据制作模型。图像有肉眼可见的潜在模式，在它们的核心，图像是矩阵。数字的模式现在是可以识别的，特别是通过卷积神经网络的进步。

但是当我们进入自然语言处理(NLP)领域时会发生什么呢？你如何让计算机理解语言、语义、语法等背后的逻辑。？

本教程标志着我们的 NLP 系列的开始。在本教程中，您将了解 NLP 的历史演变。

本课是关于 NLP 101 的 4 部分系列的第 1 部分:

1.  ***[【自然语言处理入门】](https://pyimg.co/60xld)* (今日教程)**
2.  *单词袋(BoW)模型简介*
3.  *Word2Vec:自然语言处理中的嵌入研究*
4.  【BagofWords 和 Word2Vec 的比较

**要了解自然语言处理的历史，** ***只要保持阅读。***

* * *

## [**自然语言处理入门**](#TOC)

 *** * *

### [**自然语言处理**](#TOC)

由于图像的核心是矩阵，卷积滤波器可以很容易地帮助我们检测图像的特征。语言就不一样了。CV 技术最多只能教会一个模型从图像中识别字母。

即使这样也导致了用 26 个标签进行训练，总的来说，这是一个非常糟糕的方法，因为我们根本没有抓住语言的本质。那我们在这里做什么？我们如何解开语言之谜(**图 1** )？

先来个大剧透；我们目前正处于语言模型的时代，如 GPT-3(生成式预训练转换器 3)和 BERT(来自转换器的双向编码器表示)。这些模型遵循完美的语法和语义，能够与我们进行对话。

但这一切是从哪里开始的？

让我们通过历史来简单了解一下 NLP。

* * *

### [**前言**](#TOC)

人类创造了语言作为交流的媒介，以便更有效地分享信息。我们足够聪明去创造复杂的范式作为语言的基础。纵观历史，语言经历了广泛的变化，但通过语言分享信息的本质却保持不变。

当我们听到苹果这个词时，一个新鲜的红色椭圆形水果的形象就会浮现在我们的脑海中。我们可以立即将这个词与我们脑海中的形象联系起来。我们所看到的，我们所触摸到的，以及我们所感觉到的。我们复杂的神经系统对这些刺激做出反应，我们的大脑帮助我们将这些感觉归类为固定的词语。

但在这里，我们面对的是一台计算机，它只能理解 a `0`或`1`是什么。我们的规则和范例不适用于计算机。那么，我们如何向计算机解释像语言这样复杂的东西呢？

在此之前，重要的是要明白我们对语言的理解并不像今天这样敏锐。作为一门科学，语言是语言学的主题。因此，自然语言处理成为语言学本身的一个子集。

所以让我们稍微绕一下语言学本身是如何发展成今天这个样子的。

* * *

### [**自然语言处理的雏形**](#TOC)

语言学本身就是对人类语言的科学研究。这意味着它采取了一种彻底的、有条理的、客观的、准确的对语言各个方面进行检查的方法。不用说，NLP 中的很多基础都和语言学有直接的联系。

现在你可能会问这和我们今天的旅程有什么关系。答案在于一个人的故事，他现在被认为是 20 世纪语言学之父，弗迪南·德·索绪尔。

在 20 世纪的第一个十年，德·索绪尔在日内瓦大学教授一门课程，这门课程采用了一种将语言描述为系统的方法。

一位著名的俄罗斯语言学家 Vladimir Plungyan 后来指出，

> 语言学中的“索绪尔革命”的本质是，语言被规定不被视为事实的混乱总和，而是被视为一座大厦，其中所有元素都相互联系( [source](https://en.wikipedia.org/wiki/Ferdinand_de_Saussure) )。

对德·索绪尔来说，语言中的声学声音代表了一种随语境变化的概念。

他的遗作《语言学概论》将他的结构主义语言观推向了语言学的中心。

将语言视为一个系统的结构主义方法是我们可以在现代 NLP 技术中普遍看到的。根据德索绪尔和他的学生的观点，答案在于将语言视为一个系统，在这个系统中，你可以将各种元素相互关联，从而通过因果关系识别语境。

我们的下一站是 20 世纪 50 年代，当时艾伦·图灵发表了他著名的“[计算机械和智能](https://en.wikipedia.org/wiki/Computing_Machinery_and_Intelligence)”文章，现在被称为图灵测试。该测试确定计算机程序在有独立人类法官在场的实时对话中模仿人类的能力(**图 3** )。

虽然这个测试有几个限制，但是仍然使用这个测试启发的几个检查。最值得注意的是，当浏览互联网时，CAPTCHA(区分计算机和人类的完全自动化的公共图灵测试)不时弹出。

图灵测试通常被称为“模仿游戏”,因为该测试旨在观察机器是否能模仿人类。原文章《计算机械与智能》，问“机器能思考吗？”这里出现的一个大问题是模仿是否等同于独立思考的能力。

1957 年，诺姆·乔姆斯基的“[句法结构](https://www.amazon.com/Syntactic-Structures-Noam-Chomsky/dp/1614278040/ref=sr_1_1?crid=3PEU4L4DAPYKV&keywords=1957%2C+Noam+Chomsky+Syntactic+Structures&qid=1654395189&sprefix=1957%2C+noam+chomsky+%2Caps%2C114&sr=8-1)”采用了基于规则的方法，但仍然成功地彻底改变了 NLP 世界。然而，这个时代也有自己的问题，尤其是计算的复杂性。

在此之后，出现了一些发明，但计算复杂性带来的令人震惊的问题似乎阻碍了任何重大进展。

那么，在研究人员慢慢获得足够的计算能力之后，会发生什么呢？

* * *

### [**自然语言处理找到了自己的立足点**](#TOC)

一旦对复杂的硬编码规则的依赖减轻，使用决策树等早期机器学习算法就可以获得出色的结果。然而，这一突破是由完全不同的原因造成的。

20 世纪 80 年代统计计算的兴起也进入了 NLP 领域。这些模型的基础仅仅在于为输入要素分配加权值的能力。因此，这意味着输入将总是决定模型所采取的决策，而不是基于复杂的范例。

基于统计的 NLP 的一个最简单的例子是 n-grams，其中使用了马尔可夫模型的概念(当前状态仅依赖于前一状态)。这里，想法是在上下文中识别单词对其相邻单词的解释。

推动 NLP 世界向前发展的最成功的概念之一是递归神经网络(RNNs) ( **图 4** )。

RNNs 背后的想法既巧妙又简单。我们有一个递归单元，输入`x1`通过它传递。递归单元给我们一个输出`y1`和一个隐藏状态`h1`，它携带来自`x1`的信息。

RNN 的输入是代表单词序列的符号序列。这对于所有输入重复进行，因此，来自先前状态的信息总是被保留。当然，rnn 并不完美，被更强的算法(例如 LSTMs 和 GRUs)所取代。

这些概念使用了 RNNs 背后相同的总体思想，但是引入了一些额外的效率机制。LSTM(长短期记忆)细胞有三个路径或门:输入、输出和遗忘门。LSTMs 试图解决长期依赖性问题，它可以将输入与其之前的长序列相关联。

然而，LSTMs 带来了复杂性的问题。门控递归单元(GRUs)通过减少门的数量和降低 LSTMs 的复杂性来解决这个问题。

让我们花一点时间来欣赏这些算法出现在 20 世纪 90 年代末和 21 世纪初的事实，当时计算能力仍然是一个问题。因此，让我们看看我们在强大的计算能力下取得的成就。

* * *

### [**NLP 的崛起**](#TOC)

在我们进一步讨论之前，让我们先看一下计算机如何理解语言。计算机可以创建一个矩阵，其中的列表示上下文，在该矩阵中对行中的单词进行评估(**表 1** )。

| 
 | **活着** | **财富** | **性别** |
| **男人** | one | -1 | -1 |
| **女王** | one | one | one |
| **框** | -1 | Zero | Zero |
| **表 1:** 嵌入。 |

既然我们不能把单词的意思填鸭式地输入计算机，为什么不创造一个我们可以表达单词的有限范围呢？

这里，单词`Man`在`Alive`列下具有值`1`，在`Wealth`列下具有值`-1`，在`Gender`列下具有值`-1`。同样，我们在`Alive`、`Wealth`和`Gender`下有值为`1`、`1`和`1`的单词`Queen`。

注意`Gender`和`Wealth`列是如何为这两个单词取极坐标值的。如果这就是我们如何向计算机解释`Man`是穷的，而`Queen`是富的，或者`Man`是男的，而`Queen`是女的呢？

所以我们尝试在有限的 N 维空间中“表示”每个单词。该模型基于每个单词在每个 N 维中的权重来理解每个单词。这种表示学习方法在 2003 年首次出现，自 2010 年以来在 NLP 领域得到了广泛应用。

2013 年发表了`word2vec`系列论文。它使用了表征学习(嵌入)的概念，通过在 N 维空间中表达单词，根据定义，作为存在于该空间中的向量(**图 5** )。

根据输入语料库的好坏，适当的训练将显示，当在可见空间中表达时，具有相似上下文的单词最终会在一起，如**图 5** 所示。根据数据的好坏和一个词在相似上下文中的使用频率，它的含义取决于它的相邻词。

这个概念又一次打开了 NLP 的世界，直到今天，嵌入在所有后续的研究中都扮演着重要的角色。Word2Vec 值得注意的精神追随者是 FastText 系列论文，该系列论文引入了子词的概念来进一步增强模型。

2017 年，注意力的概念出现了，它使一个模型专注于每个输入单词与每个输出单词的相关性。令人困惑的变形金刚概念是基于一种被称为自我关注的注意力变体。

变形金刚已经产生了足够强大的模型，甚至可以轻松击败图灵测试。这本身就证明了我们在教会计算机如何理解语言的过程中已经走了多远。

最近，当经过任务训练的 GPT-3 模型出现在网络上时，GPT-3 模型引起了巨大的轰动。这些模型可以完美地与任何人进行对话，这也成为了娱乐的主题，因为针对不同的任务对它们进行微调会产生非常有趣的结果。

看看变形金刚对语言的把握有多好(**图 6** )。

GPT-尼奥 1.3B 是 EleutherAI 的 GPT-3 复制模型，它提供了几个起始标记，为我们提供了一小段输出，最大限度地尊重了句法和语义规则。

NLP 一度被认为过于昂贵，其研究被严重叫停。我们缺乏计算能力和获取数据的能力。现在我们有了可以和我们保持对话的模型，甚至不会怀疑我们在和非人类对话。

然而，如果你想知道 GPT-尼奥名字中的 1.3B 代表什么，那就是模型中参数的数量。这*充分说明了*当今最先进的(SOTA)语言模型所拥有的计算复杂性。

* * *

* * *

## [**汇总**](#TOC)

对 NLP 历史的简要回顾表明，研究工作在很久以前就开始了。研究人员利用我们在语言学上对人类语言的理解所奠定的基础，并对如何推进 NLP 有了正确的想法。

然而，技术的局限性成了最大的障碍，在某一点上，这一领域的研究几乎停滞不前。但是技术只朝一个方向发展，那就是前进。技术的发展为 NLP 研究人员提供了足够的计算能力，拓宽了他们的视野。

我们正处于语言模型帮助创造虚拟助手的阶段，这些助手可以与我们交谈，帮助我们完成任务，等等。想象一下，世界已经发展到一个地步，盲人可以要求虚拟助手描述一幅图像，而且它可以完美地做到这一点。

但是这种进步是以苛刻的计算能力要求为代价的，最重要的是，要访问成吨成吨的数据。语言是这样一个话题，像我们在图像中所做的那样应用增强技术根本不能帮助我们。因此，接下来的研究集中在如何减少这些庞大的需求。

即便如此，NLP 多年来的发展还是值得称赞的。这些概念既巧妙又直观。本系列的下一篇博客将更详细地关注现代 NLP 概念。

* * *

### [**引用信息**](#TOC)

**Chakraborty，D.** “自然语言处理(NLP)导论”， *PyImageSearch* ，P. Chugh，A. R. Gosthipaty，S. Huot，K. Kidriavsteva，R. Raha，A. Thanki 编辑。，2022 年，【https://pyimg.co/60xld 

```py
@incollection{Chakraborty_2022_NLP,
  author = {Devjyoti Chakraborty},
  title = {Introduction to Natural Language Processing {(NLP)}},
  booktitle = {PyImageSearch},
  editor = {Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/60xld},
}
```

* * *****