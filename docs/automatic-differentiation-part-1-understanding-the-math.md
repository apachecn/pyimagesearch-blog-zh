# 自动微分第 1 部分:理解数学

> 原文：<https://pyimagesearch.com/2022/12/05/automatic-differentiation-part-1-understanding-the-math/>

* * *

## **目录**

* * *

## [**自动微分第一部分:理解数学**](#TOC)

在本教程中，你将学习**反向传播**所需的自动微分背后的数学。

本课是关于**Autodiff 101-从头开始了解自动区分的两部分系列的第一部分**:

1.  [***自动微分第一部分:理解数学***](https://pyimg.co/pyxml) **(今日教程)**
2.  *自动微分第 2 部分:使用微克实现*

**要了解自动微分，** ***只需继续阅读。***

* * *

## [**自动微分第一部分:理解数学**](#TOC)

想象你正徒步下山。天很黑，有很多颠簸和转弯。你无法知道如何到达中心。现在想象一下，你每前进一次，都要暂停一下，拿出山丘的拓扑图，计算下一组你的方向和速度。听起来很无趣，对吧？

如果你读过我们的教程，你会知道这个类比指的是什么。小山是你的损失景观，拓扑图是多元微积分的规则集，你是神经网络的参数。目标是达到全局最小值。

这就引出了一个问题:

> *为什么我们今天使用深度学习框架？*

脑海里首先闪现的是**自动微分**。我们写向前传球，就是这样；不用担心后传。每个操作符都是自动微分的，并等待在优化算法中使用(如随机梯度下降)。

今天在本教程中，我们将走过自动微分的山谷。

* * *

### [**简介**](#TOC)

在本节中，我们将为理解`autodiff`奠定必要的基础。

* * *

#### [**雅可比**](#TOC)

让我们考虑一个函数![F \colon \mathbb{R}^{n} \to \mathbb{R}](img/5dad79efc2d96ca89a7044e43fc82e25.png "F \colon \mathbb{R}^{n} \to \mathbb{R}")

. ![F](img/3795a8e26f6613a5d67a394c2baafdb4.png "F")is a multivariate function that simultaneously depends on multiple variables. Here the multiple variables can be ![x = \{x_{1}, x_{2}, \ldots, x_{n}\}](img/5d71fdc0205175955e3e267c51efa4b0.png "x = \{x_{1}, x_{2}, \ldots, x_{n}\}"). The output of the function is a **scalar value**. This can be considered as a neural network that takes an image and outputs the probability of a dog’s presence in the image.

***注意*** **:** 让我们回忆一下，在神经网络中，我们计算的是关于参数(权重和偏差)而不是输入(图像)的梯度。因此，函数的域是参数而不是输入，这有助于保持梯度计算的可访问性。我们现在需要从使**简单**和**有效**的角度考虑我们在本教程中所做的一切，以获得关于权重和偏差(*参数*)的梯度。图 1 中的**对此进行了说明。**

神经网络由许多子层组成。所以让我们考虑一下我们的函数![F(x)](img/08d1b56e1e49db8d9464fb1e6f6c73bf.png "F(x)")

as a composition of multiple functions (primitive operations).

![F(x) \ = \ D \circ C \circ B \circ A](img/28ace9da7e32994018593379d71c3406.png "F(x) \ = \ D \circ C \circ B \circ A")

该功能![F(x)](img/08d1b56e1e49db8d9464fb1e6f6c73bf.png "F(x)")

is composed of four primitive functions, namely ![D, C, B, \text{ and } A](img/9a5529f8ac62fb77df4671da24c487d3.png "D, C, B, \text{ and } A"). For anyone new to composition, we can call ![F(x)](img/08d1b56e1e49db8d9464fb1e6f6c73bf.png "F(x)")to be a function where ![D(C(B(A(x))))](img/12d5092cc4fdfc16af2541cac8a7382d.png "D(C(B(A(x))))")is equal to ![F(x)](img/08d1b56e1e49db8d9464fb1e6f6c73bf.png "F(x)").

下一步是找到![F(x)](img/08d1b56e1e49db8d9464fb1e6f6c73bf.png "F(x)")的梯度

. However, before diving into the gradients of the function, let us revisit Jacobian matrices. It turns out that the derivatives of a multivariate function are a Jacobian matrix consisting of partial derivatives of the function w.r.t. all the variables upon which it depends.

考虑两个多元函数，![u](img/5a599d2ae07a4744c86840c1e5e5afa7.png "u")

and ![v](img/5cf5c733f94b1c39da6382d8e46f92a3.png "v"), which depend on the variables ![x](img/475aa77643ede4a370076ea4f6abd268.png "x")and ![y](img/1308649485b87d470e3ba53ff87b8970.png "y"). The Jacobian would look like this:

![\displaystyle\frac{\partial{(u, v)}}{\partial{x, y}} \ = \ \begin{bmatrix} \displaystyle\frac{\partial u}{\partial x} & \displaystyle\frac{\partial u}{\partial y}\\  \\ \displaystyle\frac{\partial v}{\partial x} & \displaystyle\frac{\partial v}{\partial y} \end{bmatrix}](img/4e138674aa9ad0f7c9f55e34069dbdd3.png "\displaystyle\frac{\partial{(u, v)}}{\partial{x, y}} \ = \ \begin{bmatrix} \displaystyle\frac{\partial u}{\partial x} & \displaystyle\frac{\partial u}{\partial y}\\  \\ \displaystyle\frac{\partial v}{\partial x} & \displaystyle\frac{\partial v}{\partial y} \end{bmatrix}")

现在让我们计算函数![F(x)](img/08d1b56e1e49db8d9464fb1e6f6c73bf.png "F(x)")的雅可比矩阵

. We need to note here that the function depends of ![n](img/f4495f912b12e5a248ce066ec8a2b2f6.png "n")variables ![x = \{x_{1}, x_{2}, \ldots, x_{n}\}](img/5d71fdc0205175955e3e267c51efa4b0.png "x = \{x_{1}, x_{2}, \ldots, x_{n}\}"), and outputs a scalar value. This means that the Jacobian will be a row vector.

![F^\prime(x) \ = \ \displaystyle\frac{\partial{y}}{\partial{x}} \ = \ \begin{bmatrix} \displaystyle\frac{\partial y}{\partial x_{1}} & \ldots  & \displaystyle\frac{\partial y}{\partial x_{n}} \end{bmatrix}](img/7406be3f531267f400e18af295368fdf.png "F^\prime(x) \ = \ \displaystyle\frac{\partial{y}}{\partial{x}} \ = \ \begin{bmatrix} \displaystyle\frac{\partial y}{\partial x_{1}} & \ldots  & \displaystyle\frac{\partial y}{\partial x_{n}} \end{bmatrix}")

* * *

#### [**链式法则**](#TOC)

还记得我们的函数![F(x)](img/08d1b56e1e49db8d9464fb1e6f6c73bf.png "F(x)")

is composed of many primitive functions? The derivative of such a composed function is done with the help of the chain rule. To help our way into the chain rule, let us first write down the composition and then define the intermediate values.

![F(x) = D(C(B(A(x))))](img/8777f22449df5824a9553dab32c6b50d.png "F(x) = D(C(B(A(x))))")

is composed of:

*   ![y = D(c)](img/ef50e5746f306bad0ac073c3899a2940.png "y = D(c)")
*   ![c = C(b)](img/c4adf6e6d8c90775539636be0c209ba6.png "c = C(b)")
*   ![b = B(a)](img/27d533ebe313a8470f6ca645578839f3.png "b = B(a)")
*   ![a = A(x)](img/b2447eb5400ea0d705d1a41a4a3e6ca7.png "a = A(x)")

现在作文已经拼出来了，我们先来求中间值的导数。

*   ![D^\prime(c) = \displaystyle\frac{\partial{y}}{\partial{c}}](img/5e71b94c58169f68842c7b732be1bfac.png "D^\prime(c) = \displaystyle\frac{\partial{y}}{\partial{c}}")
*   ![C^\prime(b) = \displaystyle\frac{\partial{c}}{\partial{b}}](img/f3b91cc574d05368a6608f1d6066deb0.png "C^\prime(b) = \displaystyle\frac{\partial{c}}{\partial{b}}")
*   ![B^\prime(a) = \displaystyle\frac{\partial{b}}{\partial{a}}](img/caeec877a1f44890685eff6a2c293c58.png "B^\prime(a) = \displaystyle\frac{\partial{b}}{\partial{a}}")
*   ![A^\prime(x) = \displaystyle\frac{\partial{a}}{\partial{x}}](img/06d0ff3778b40f460014901564aa06f7.png "A^\prime(x) = \displaystyle\frac{\partial{a}}{\partial{x}}")

现在借助链式法则，我们推导出函数![F(x)](img/08d1b56e1e49db8d9464fb1e6f6c73bf.png "F(x)")的导数

.

![F^\prime(x) \ = \ \displaystyle\frac{\partial{y}}{\partial{c}} \displaystyle\frac{\partial{c}}{\partial{b}} \displaystyle\frac{\partial{b}}{\partial{a}} \displaystyle\frac{\partial{a}}{\partial{x}}](img/0917403d6b272386407850c17f66efd8.png "F^\prime(x) \ = \ \displaystyle\frac{\partial{y}}{\partial{c}} \displaystyle\frac{\partial{c}}{\partial{b}} \displaystyle\frac{\partial{b}}{\partial{a}} \displaystyle\frac{\partial{a}}{\partial{x}}")

* * *

#### [**混合了雅可比和链式法则**](#TOC)

在了解了雅可比矩阵和链式法则之后，让我们一起将两者形象化。如**图二**所示。

![F^\prime(x) \ = \ \displaystyle\frac{\partial{y}}{\partial{x}} \ = \ \begin{bmatrix} \displaystyle\frac{\partial y}{\partial x_{1}} & \ldots  & \displaystyle\frac{\partial y}{\partial x_{n}} \end{bmatrix}](img/7406be3f531267f400e18af295368fdf.png "F^\prime(x) \ = \ \displaystyle\frac{\partial{y}}{\partial{x}} \ = \ \begin{bmatrix} \displaystyle\frac{\partial y}{\partial x_{1}} & \ldots  & \displaystyle\frac{\partial y}{\partial x_{n}} \end{bmatrix}")

![F^\prime(x) \ = \ \displaystyle\frac{\partial{y}}{\partial{c}} \displaystyle\frac{\partial{c}}{\partial{b}} \displaystyle\frac{\partial{b}}{\partial{a}} \displaystyle\frac{\partial{a}}{\partial{x}}](img/0917403d6b272386407850c17f66efd8.png "F^\prime(x) \ = \ \displaystyle\frac{\partial{y}}{\partial{c}} \displaystyle\frac{\partial{c}}{\partial{b}} \displaystyle\frac{\partial{b}}{\partial{a}} \displaystyle\frac{\partial{a}}{\partial{x}}")

我们函数的导数![F(x)](img/08d1b56e1e49db8d9464fb1e6f6c73bf.png "F(x)")

is just the matrix multiplication of the Jacobian matrices of the intermediate terms.

现在，这就是我们要问的问题:

> 我们做矩阵乘法的顺序有关系吗？

* * *

#### [**正向和反向累加**](#TOC)

在本节中，我们试图理解雅可比矩阵乘法排序问题的答案。

在两种极端情况下，我们可以对乘法进行排序:正向累加和反向累加。

* * *

#### [**正向累加**](#TOC)

如果我们按照与函数![F(x)](img/08d1b56e1e49db8d9464fb1e6f6c73bf.png "F(x)")相同的顺序从右到左排列乘法

was evaluated, the process is called forward accumulation. The best way to think about the ordering is to place brackets in the equation, as shown in **Figure 3**.

![F^\prime(x) \ = \ \displaystyle\frac{\partial{y}}{\partial{c}} \left(\frac{\partial{c}}{\partial{b}} \left(\frac{\partial{b}}{\partial{a}} \displaystyle\frac{\partial{a}}{\partial{x}}\right)\right)](img/e52dd5d55959dbfcc3bae1f8637478e8.png "F^\prime(x) \ = \ \displaystyle\frac{\partial{y}}{\partial{c}} \left(\frac{\partial{c}}{\partial{b}} \left(\frac{\partial{b}}{\partial{a}} \displaystyle\frac{\partial{a}}{\partial{x}}\right)\right)")

使用功能![F : \mathbb{R}^{n} \to \mathbb{R}](img/24cb24ffa7b6280665db6f9f7cab3ca1.png "F : \mathbb{R}^{n} \to \mathbb{R}")

, the forward accumulation process is matrix multiplication in all the steps. This is more [FLOPs](https://stackoverflow.com/a/60275432/19623137).

***注意:*** 向前累加在我们想要得到一个函数的导数的时候是很有好处的![F: \mathbb{R} \to \mathbb{R}^{n}](img/ceebc0f8f5f4450a6df2815dcf1e0952.png "F: \mathbb{R} \to \mathbb{R}^{n}")

.

理解转发累加的另一种方式是考虑雅可比矢量积(JVP)。考虑一个雅各比派![F^\prime(x)](img/afb70686ec8aacbd739feea64397aaac.png "F^\prime(x)")

and a vector ![v](img/5cf5c733f94b1c39da6382d8e46f92a3.png "v"). The Jacobian-Vector Product would look to be ![F^\prime(x)v](img/0152e3550aaf75c6cf76f8da35637e05.png "F^\prime(x)v")

![F^\prime(x)v \ = \ \displaystyle\frac{\partial{y}}{\partial{c}} \left(\displaystyle\frac{\partial{c}}{\partial{b}} \left(\displaystyle\frac{\partial{b}}{\partial{a}} \left(\displaystyle\frac{\partial{a}}{\partial{x}} v\right)\right)\right)](img/a2b96dbc8d299f394b5ce42d1e9fc73e.png "F^\prime(x)v \ = \ \displaystyle\frac{\partial{y}}{\partial{c}} \left(\displaystyle\frac{\partial{c}}{\partial{b}} \left(\displaystyle\frac{\partial{b}}{\partial{a}} \left(\displaystyle\frac{\partial{a}}{\partial{x}} v\right)\right)\right)")

这样做是为了让我们在所有阶段都有矩阵向量乘法(这使得过程更有效)。

➤ **问题:**如果我们有一个雅可比矢量积，我们如何从中获得雅可比？

➤ **回答:**我们传递一个热点向量，一次得到雅可比矩阵的每一列。

因此，我们可以将前向累加视为一个过程，在此过程中，我们构建每列的雅可比矩阵。

* * *

#### [**反向积累**](#TOC)

假设我们从左到右对乘法进行排序，方向与函数求值的方向相反。在这种情况下，这个过程叫做反向积累。该过程的示意图如**图 4** 所示。

![F^\prime(x) \ = \ \left(\left(\displaystyle\frac{\partial{y}}{\partial{c}} \displaystyle\frac{\partial{c}}{\partial{b}}\right) \displaystyle\frac{\partial{b}}{\partial{a}} \right)\displaystyle\frac{\partial{a}}{\partial{x}}](img/e987958081400bc34847500a086deb59.png "F^\prime(x) \ = \ \left(\left(\displaystyle\frac{\partial{y}}{\partial{c}} \displaystyle\frac{\partial{c}}{\partial{b}}\right) \displaystyle\frac{\partial{b}}{\partial{a}} \right)\displaystyle\frac{\partial{a}}{\partial{x}}")

事实证明，用反向累加来推导一个函数的导数![F : \mathbb{R}^{n} \to \mathbb{R}](img/24cb24ffa7b6280665db6f9f7cab3ca1.png "F : \mathbb{R}^{n} \to \mathbb{R}")

is a vector to matrix multiplication at all steps. This means that for the particular function, reverse accumulation has lesser FLOPs than forwarding accumulation.

理解前向累加的另一种方法是考虑一个矢量雅可比乘积(VJP)。考虑一个雅各比派![F^\prime(x)](img/afb70686ec8aacbd739feea64397aaac.png "F^\prime(x)")

and a vector ![v](img/5cf5c733f94b1c39da6382d8e46f92a3.png "v"). The Vector-Jacobian Product would look to be ![v^{T}F^\prime(x)](img/55c7a586fe5ce78e8823802e46773ed6.png "v^{T}F^\prime(x)")

![v^{T}F^\prime(x) \ = \ \left(\left(\left(v^{T} \displaystyle\frac{\partial{y}}{\partial{c}}\right) \displaystyle\frac{\partial{c}}{\partial{b}}\right) \displaystyle\frac{\partial{b}}{\partial{a}}\right)\displaystyle\frac{\partial{a}}{\partial{x}}](img/2770acaf3556ef64c754af0f05436ac3.png "v^{T}F^\prime(x) \ = \ \left(\left(\left(v^{T} \displaystyle\frac{\partial{y}}{\partial{c}}\right) \displaystyle\frac{\partial{c}}{\partial{b}}\right) \displaystyle\frac{\partial{b}}{\partial{a}}\right)\displaystyle\frac{\partial{a}}{\partial{x}}")

这允许我们在所有阶段进行向量矩阵乘法(这使得过程更有效)。

➤ **问题:**如果我们有一个向量雅可比乘积，我们如何从中获得雅可比？

➤ **回答:**我们传递一个独热向量，一次得到雅可比矩阵的每一行。

所以我们可以把逆向累加看作是一个建立每行雅可比矩阵的过程。

现在，如果我们考虑我们之前提到的函数![F(x)](img/08d1b56e1e49db8d9464fb1e6f6c73bf.png "F(x)")

, we know that the Jacobian ![F^\prime(x)](img/afb70686ec8aacbd739feea64397aaac.png "F^\prime(x)")is a row vector. Therefore, if we apply the reverse accumulation process, which means the Vector-Jacobian Product, we can obtain the row vector in one shot. On the other hand, if we apply the forward accumulation process, the Jacobian-Vector Product, we will obtain a single element as a column, and we would need to iterate to build the entire row.

这就是为什么反向累加在神经网络文献中更常用的原因。

* * *

* * *

## [**汇总**](#TOC)

在本教程中，我们学习了自动微分的数学，以及如何将其应用于神经网络的参数。下一篇教程将对此进行扩展，看看我们如何使用 python 包实现自动微分。该实现将涉及创建 python 包并使用它来训练神经网络的逐步演练。

你喜欢关于自动微分基础的数学教程吗？让我们知道。

**推特:** [@PyImageSearch](https://twitter.com/pyimagesearch)

* * *

### [**参考文献**](#TOC)

*   [自动微分—VideoLectures.NET](http://videolectures.net/deeplearning2017_johnson_automatic_differentiation/)

* * *

### [**引用信息**](#TOC)

A. R. Gosthipaty 和 R. Raha。“自动微分第一部分:理解数学”， *PyImageSearch* ，P. Chugh、S. Huot、K. Kidriavsteva 和 A. Thanki 编辑。，2022 年，【https://pyimg.co/pyxml 

```py
@incollection{ARG-RR_2022_autodiff1,
  author = {Aritra Roy Gosthipaty and Ritwik Raha},
  title = {Automatic Differentiation Part 1: Understanding the Math},
  booktitle = {PyImageSearch},
  editor = {Puneet Chugh and Susan Huot and Kseniia Kidriavsteva and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/pyxml},
}
```

* * *

* * *