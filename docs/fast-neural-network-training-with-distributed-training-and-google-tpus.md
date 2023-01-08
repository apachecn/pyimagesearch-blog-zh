# 使用分布式训练和 Google TPUs 的快速神经网络训练

> 原文：<https://pyimagesearch.com/2021/12/06/fast-neural-network-training-with-distributed-training-and-google-tpus/>

我的日常生活包括训练很多深度学习模型。有时候我很幸运拥有一个小小的架构，却能够提供非凡的结果。其他时候，我不得不踏上训练大型架构的艰难道路，以获得好的结果。

随着数据饥渴的深度学习模型的规模不断增加，我们很少谈论训练一个少于**1000 万个参数**的模型。因此，硬件访问受限的人没有机会训练这些模型，即使他们有机会，训练时间也是如此之长，他们不能以他们想要的速度迭代该过程。

## **利用分布式训练和 Google TPUs 进行快速神经网络训练**

在这篇文章中，我将提供一些我发现对加快我的培训过程特别有用的商业秘密。我们将讨论用于深度学习的不同硬件，以及不会使正在使用的硬件挨饿的高效数据管道。这篇文章将会让你和你的培训渠道更有效率。

在文章中，我们将讨论:

*   用于深度学习的不同硬件
*   高效的数据管道
*   分发培训流程

**要了解如何使用 Google TPUs、** ***执行分布式训练，请继续阅读。***

### **配置您的开发环境**

为了遵循这个指南，你需要在你的系统上安装 **TensorFlow** 和 **TensorFlow 数据集**库。

幸运的是，这些包是 pip 可安装的:

```py
$ pip install tensorflow
$ pip install tensorflow-datasets
```

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

在我们继续之前，让我们首先回顾一下我们的项目目录结构。首先访问本指南的 ***“下载”*** 部分，检索源代码和 Python 脚本。

然后，您将看到以下目录结构:

```py
$ tree . --dirsfirst
.
├── outputs
│   ├── cpu.png
│   ├── gpu.png
│   └── tpu.png
├── pyimagesearch
│   ├── autoencoder.py
│   ├── config.py
│   ├── data.py
│   └── loss.py
├── train_cpu.py
├── train_gpu.py
└── train_tpu.py

2 directories, 10 files
```

在`pyimagesearch`模块中，我们有以下文件:

*   `autoencoder.py`:定义需要训练的自动编码器模型
*   `config.py`:定义培训所需的配置文件
*   `data.py`:定义模型训练步骤的数据管道
*   `loss.py`:定义将用于培训的损失

最后，我们有三个 Python 脚本:

*   `train_cpu.py`:在 CPU 上训练模型
*   `train_gpu.py`:在 GPU 上训练模型
*   在 TPU 上训练模型

`outputs`目录包含在不同硬件上训练的自动编码器的推理图像。

### **硬件**

在深度学习中，最基本的运算是*矩阵乘法*的运算。我们乘得越快，我们在训练中达到的速度就越快。密歇根大学有一个关于深度学习中的[硬件](https://www.youtube.com/watch?v=oXPX8GIOiU4&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&index=9)的精彩讲座，我建议观看，以了解硬件多年来如何演变以适应深度学习。在这一节中，我们将反复讨论硬件的类型，并试图找出哪一种更适合我们的目的。

#### **CPU**

中央处理器(CPU)是一个基于[冯·诺伊曼](https://en.wikipedia.org/wiki/Von_Neumann_architecture)T2【nn】架构的处理器。该架构提出了一种具有以下组件的电子计算机:

*   **处理单元:**处理输入的数据
*   一个控制单元:它保存指令和一个程序计数器来控制整个工作流程
*   **内存:**用于存储

在冯诺依曼体系结构中，指令和数据存在于存储器中。处理器访问指令并相应地处理数据。它还使用内存来存储中间计算，并在以后访问它来完成任何计算。

这种架构非常灵活。我们基本上可以提供任何指令和数据，处理器将完成剩下的工作。然而，这种灵活性是有代价的——速度。

该架构依赖于存储器访问，也依赖于下一步的控制指令。内存访问变成了所谓的[冯诺依曼瓶颈](https://www.techopedia.com/definition/14630/von-neumann-bottleneck)。即使我们整天都在做矩阵乘法，CPU 也没有办法去猜测未来的运算；因此，它需要不断访问数据和指令。

谷歌关于 TPUs 的指南中的一个片段揭示了上述问题。

> 每个 CPU 的算术逻辑单元(alu)是保存和控制乘法器和加法器的组件，一次只能执行一个计算。每次，CPU 都必须访问内存，这限制了总吞吐量并消耗大量能量。

**图 2** 显示了 CPU 中矩阵乘法的简化版本。操作按顺序进行，每一步都有存储器访问。

让我们用 TensorFlow 测试一下我们的 CPU 执行矩阵乘法的速度。打开 Google Colab 笔记本，粘贴以下代码，亲自查看结果。

```py
# import the necessary packages
import tensorflow as tf
import time

# initialize the operands
w = tf.random.normal((1024, 512, 16))
x = tf.random.normal((1024, 16, 512))
b = tf.random.normal((1024, 512, 512))

# start timer
start = time.time()

# perform matrix multiplication
output = tf.matmul(w, x) + b

# end timer
end = time.time()

# print the time taken to perform the operation
print(f"time taken: {(end-start):.2f} sec")

>>> ​​time taken: 0.79 sec
```

让我们用上面的代码对我们的 CPU 做一点时间测试。在这里，我们模拟了乘法和加法运算，这是深度学习中最常见的运算。我们看到操作需要`0.79 sec`来完成。

#### **图形处理器**

图形处理单元(GPU)试图通过在单个处理器上集成数千个算术逻辑单元(alu)来提高 CPU 的吞吐量。通过这种方式，GPU 实现了操作的并行性。

矩阵乘法是让深度学习计算适合 GPU 的并行运算。GPU 不是专门为矩阵乘法而构建的，这意味着它们仍然需要从下一步开始访问内存和控制指令——这就是冯诺依曼瓶颈。即使遇到瓶颈，由于并行操作，GPU 在训练过程中也提供了一个重要的进步。

**图 3** 展示了 GPU 上矩阵乘法的简化版本。请注意 alu 的增加如何帮助实现并行和更快的计算。

下面的代码和用 CPU 做的一样。这里的变化与所使用的硬件有关。我们在这里使用 GPU 来运行代码。结果，代码比 CPU 少花了大约`~99%`时间。这表明了我们的 GPU 有多么强大，以及并行性如何产生巨大的差异。我强烈推荐在带有 GPU 运行时的 Google Colab 笔记本中运行以下代码。

```py
# import the necessary packages
import tensorflow as tf
import time

# initialize the operands
w = tf.random.normal((1024, 512, 16))
x = tf.random.normal((1024, 16, 512))
b = tf.random.normal((1024, 512, 512))

# start timer
start = time.time()

# perform matrix multiplication
output = tf.matmul(w, x) + b

# end timer
end = time.time()

# print the time taken to perform the operation
print(f"time taken: {(end-start):.6f} sec")

>>> time taken: 0.000436 sec
```

#### **TPUs**

我们已经可以破译张量处理单元(TPU)在深度学习中的优势。

以下是该指南的一个片段:

> 云 TPU 是定制设计的机器学习 ASIC(专用集成芯片)，为翻译、照片、搜索、助手和 Gmail 等谷歌产品提供支持。TPU 优于其他设备的一个好处是大大减少了冯诺依曼瓶颈。因为这种处理器的主要任务是矩阵处理，TPU 的硬件设计人员知道执行该操作的每个计算步骤。因此，他们能够放置数千个乘法器和加法器，并将它们直接连接起来，形成一个由这些运算符组成的大型物理矩阵。这被称为脉动阵列架构。

在脉动阵列架构的帮助下，TPU 首先加载参数，然后动态处理数据。该架构使得数据可以被系统地相乘和相加，而不需要存储器访问来获取指令或存储中间结果。

**图 4** 显示了 TPU 处理步骤的可视化资源:

<https://www.youtube.com/embed/JC84GCU7zqA?feature=oembed>*