# 使用 TensorFlow 和 Keras 的 NeRF 的计算机图形和深度学习:第 1 部分

> 原文：<https://pyimagesearch.com/2021/11/10/computer-graphics-and-deep-learning-with-nerf-using-tensorflow-and-keras-part-1/>

几天前，在浏览我的图片库时，我看到了三张图片，如图 1 所示。

我们可以看到，这些照片是一个楼梯，但从不同的角度。我拍了三张照片，因为我不确定仅凭一张照片就能捕捉到这美丽的一幕。我担心我会错过正确的视角。

这让我想到，“如果有一种方法可以从这些照片中捕捉整个 3D 场景，会怎么样？”

这样，你(我的观众)就能确切地看到我那天所看到的。

在 [NeRF:将场景表示为用于视图合成的神经辐射场中，Mildenhall 等人(2020)](https://arxiv.org/abs/2003.08934) 提出了一种方法，事实证明这正是我所需要的。

让我们看看通过复制论文的方法我们取得了什么，如图**图 2** 所示。例如，给算法一些不同视角的一盘热狗的图片(*上*)就可以精确地生成整个 3D 场景(*下*)。

神经辐射场(NeRF)汇集了**深度学习**和**计算机图形学**。虽然我们 PyImageSearch 已经写了很多关于深度学习的文章，但这将是我们第一次谈论计算机图形学。这个系列的结构将最适合初学者。 ***我们期望没有计算机图形学知识的人*** 。

***注:*** *一个更简单的 NeRF 实现为我们赢得了* [*TensorFlow 社区聚焦奖*](https://twitter.com/TensorFlow/status/1466150113814929413) *。*

**要了解计算机图形学和图像渲染，*继续阅读。***

* * *

## **[使用 TensorFlow 和 Keras 的 NeRF 的计算机图形学和深度学习:第 1 部分](#TOC)**

计算机图形学是现代技术的奇迹之一。渲染真实 3D 场景的应用范围从电影、太空导航到医学。

本课是关于使用 TensorFlow 和 Keras 的 NeRF 的计算机图形和深度学习的 3 部分系列的第 1 部分:

*   使用 TensorFlow 和 Keras 的 NeRF 的计算机图形学和深度学习:第 1 部分(本教程)
*   使用 TensorFlow 和 Keras 的 NeRF 的计算机图形学和深度学习:第二部分(下周教程)
*   *使用 TensorFlow 和 Keras 的 NeRF 的计算机图形学和深度学习:第 3 部分*

在本教程中，我们将涵盖计算机图形世界中的相机工作。我们还将向您介绍我们将要工作的数据集。

我们将本教程分为以下几个小节:

*   **世界坐标框架:**表示物理的 3D 世界
*   **摄像机坐标框架:**表示虚拟 3D 摄像机世界
*   **坐标变换:**从一个坐标系映射到另一个坐标系
*   **投影变换:**在 2D 平面(相机传感器)上形成图像
*   **数据集:**了解 NeRF 的数据集

想象一下。你带着相机出去，发现了一朵美丽的花。你想想你想捕捉它的方式。现在是时候调整相机方向，校准设置，并点击图片。

将世界场景转换为图像的整个过程被封装在一个通常称为**前向成像模型的数学模型中。**我们可以将模型可视化在**图 3** 中。

前向成像模型从**世界坐标框架**中的一点开始。然后我们使用**坐标转换**将它转换到**摄像机坐标框架**。之后，我们使用**投影变换**将摄像机坐标变换到**图像平面**上。

* * *

### **[世界坐标框架](#TOC)**

我们在现实世界中看到的所有形状和物体都存在于 3D 参照系中。我们称这个参考系为世界坐标系统。使用这个框架，我们可以很容易地定位三维空间中的任何点或物体。

让我们来点![P](img/eeeebd93c479d01033cf1c82fcc5c60a.png "P")

in the 3D space as shown in **Figure 4**.

在这里，![\hat{x}_{w}](img/58580841e2f8244bdc013b95e707dbc4.png "\hat{x}_{w}")

, ![\hat{y}_{w}](img/9d35fb76a47dcbc9050fa2eccc2f74d7.png "\hat{y}_{w}"), and ![\hat{z}_w](img/f3bc51c9cb53fe07e7ca38202856ab03.png "\hat{z}_w")represent the three axes in the world coordinate frame. The location of the point ![P](img/eeeebd93c479d01033cf1c82fcc5c60a.png "P")is expressed through the vector ![X_{w} ](img/2906d20559ceba9b4de865347ff33861.png "X_{w} ").

![X_{w} = \begin{bmatrix} x_w \\ y_w \\ z_w \\ \end{bmatrix} ](img/5b9ae422c31d28b72251d9860795eb1d.png "X_{w} = \begin{bmatrix} x_w \\ y_w \\ z_w \\ \end{bmatrix} ")

* * *

### **[摄像机坐标框](#TOC)**

和世界坐标系一样，我们有另一个参照系，叫做相机坐标系，如图**图 5** 所示。

***这一帧位于摄像机*** 的中心。与世界坐标框架不同，这不是一个静态的参考框架。我们可以像移动相机拍照一样移动这个框架。

同一点![P](img/eeeebd93c479d01033cf1c82fcc5c60a.png "P")

from **Figure 4** can now be located with both frames of reference, as shown in **Figure 6**.

而在世界坐标框架中，该点由![X_{w} ](img/2906d20559ceba9b4de865347ff33861.png "X_{w} ")定位

vector, in the camera coordinate frame, it is located by the ![X_{c} ](img/7d8c2329c986eaf521dd6b744e9f9e4c.png "X_{c} ")vector as shown in **Figure 6**.

![X_{c} = \begin{bmatrix} x_c \\ y_c \\ z_c \\ \end{bmatrix}](img/9a068e4848b8e320cd17fdf4e763cbd9.png "X_{c} = \begin{bmatrix} x_c \\ y_c \\ z_c \\ \end{bmatrix}")

***注:*** *点的位置![P](img/eeeebd93c479d01033cf1c82fcc5c60a.png "P")不变。只是看待这个点的方式随着参照系的变化而变化。*

* * *

### **[坐标变换](#TOC)**

我们建立了两个坐标框架:世界和相机。现在让我们定义两者之间的映射。

让我们来点![P ](img/dfde8956c1ee189ef875441420bb69ae.png "P ")

from **Figure 6**. Our goal is to build a bridge between the camera coordinates ![X_{c}](img/a5affa35364f76000a546a427c627d23.png "X_{c}")and world coordinates ![X_{w}](img/506f27171f1c219c973cff459562b718.png "X_{w}").

From **Figure 5**, we can say that

![X_{c} = R \times (X_{w}-C_{w})](img/c5ad19dacd47bac0a91631bbe2a9fc8a.png "X_{c} = R \times (X_{w}-C_{w})")

在哪里

*   ![R](img/ecef2e349c0d42065bb14ef75a500855.png "R")表示摄像机坐标框架相对于世界坐标框架的方向。方向由矩阵表示。

    ![R = \begin{bmatrix} r_{11} & r_{12} & r_{13} \\ r_{21} & r_{22} & r_{23} \\ r_{31} &r_{32} & r_{33} \\ \end{bmatrix} ](img/cd8ece5b54672fe23ed8bb0800487507.png "R = \begin{bmatrix} r_{11} & r_{12} & r_{13} \\ r_{21} & r_{22} & r_{23} \\ r_{31} &r_{32} & r_{33} \\ \end{bmatrix} ")

    ![\begin{bmatrix} r_{11} & r_{12} & r_{13}\\ \end{bmatrix} ](img/15ac50158b79d51ab96bc67b0982a943.png "\begin{bmatrix} r_{11} & r_{12} & r_{13}\\ \end{bmatrix} ")→世界坐标系中的![x_c ](img/a08af2a01ffcc81970ae6cee6469fc62.png "x_c ")方向。

    ![\begin{bmatrix} r_{21} & r_{22} & r_{23}\\ \end{bmatrix} ](img/a98558a276db02e0a62b36e14d5ec4b0.png "\begin{bmatrix} r_{21} & r_{22} & r_{23}\\ \end{bmatrix} ")→世界坐标系中的![y_c ](img/d6b012cc1d8868d6e68a40210ad784db.png "y_c ")方向。

    ![\begin{bmatrix} r_{31} & r_{32} & r_{33}\\ \end{bmatrix} ](img/ce3b8c38a00aaf346420454dd66c6700.png "\begin{bmatrix} r_{31} & r_{32} & r_{33}\\ \end{bmatrix} ")→世界坐标系中![z_c ](img/3a33219655f2ef069673a3b1b1842043.png "z_c ")的方向。

*   ![C_w ](img/01dae9dd3a193c578070c47cbf2abe51.png "C_w ")代表摄像机坐标框架相对于世界坐标框架的位置。位置由向量表示。

我们可以将上面的等式展开如下

![X_{c} = R\times (X_{w}-C_{w}) \\ \Rightarrow X_{c} = R \times X_w - R \times C_w \\ \Rightarrow X_{c} = R \times X_w + t ](img/8e7734713309403279ad20561c2da34e.png "X_{c} = R\times (X_{w}-C_{w}) \\ \Rightarrow X_{c} = R \times X_w - R \times C_w \\ \Rightarrow X_{c} = R \times X_w + t ")

其中![t](img/35f482cb64a2e4778fb2718b111d8b99.png "t")

represents the translation matrix ![-(R \times C_w) ](img/e88ba8af95a1303791a08e325f30a496.png "-(R \times C_w) "). The mapping between the two coordinate systems has been devised but is not yet complete. In the above equation, we have a matrix multiplication along with a matrix addition. It is always preferable to compress things to a single matrix multiplication if we can. To do so we will use a concept called [homogeneous coordinates](https://en.wikipedia.org/wiki/Homogeneous_coordinates).

The homogeneous coordinate system allows us to represent an ![N](img/afff76eada32d0b1ca79954b0cb72a4d.png "N")dimensional point ![x = [x_0, x_1, \dots, x_n]](img/37c282989d37059c7e32664a10e715bd.png "x = [x_0, x_1, \dots, x_n]")in an ![N+1](img/b63b06c8b2dfcfa5207d2816b1de1fd2.png "N+1")dimensional space ![\tilde{x} = [\tilde{x}_0, \tilde{x}_1, \dots, \tilde{x}_n, w]](img/7034faeccb017d1c8887949d74b61369.png "\tilde{x} = [\tilde{x}_0, \tilde{x}_1, \dots, \tilde{x}_n, w]")with a fictitious variable ![w \ne 0](img/2c4a68dab8a027e9f0601c11e9ce55c2.png "w \ne 0")such that

![x_0 = \displaystyle\frac{\tilde{x}_0}{w}, \space x_1 = \displaystyle\frac{\tilde{x}_1}{w} ,\dots ,\space x_n = \displaystyle\frac{\tilde{x}_n}{w} ](img/1142ac4f72680348450cd85eed359010.png "x_0 = \displaystyle\frac{\tilde{x}_0}{w}, \space x_1 = \displaystyle\frac{\tilde{x}_1}{w} ,\dots ,\space x_n = \displaystyle\frac{\tilde{x}_n}{w} ")

使用齐次坐标系我们可以转换![X_w](img/68034168e04b332ec11a32385fbe645c.png "X_w")

(3D) to ![\tilde{X}_w](img/ebf592957ef823a32ba98db0411597af.png "\tilde{X}_w")(4D).

![X_w \equiv \begin{bmatrix} x\\ y\\ z\\ 1\\ \end{bmatrix} \equiv \begin{bmatrix} wx_w\\ wy_w\\ wz_w\\ w\\ \end{bmatrix} \equiv \begin{bmatrix} \tilde{x}\\ \tilde{y}\\ \tilde{z}\\ w\\ \end{bmatrix} \equiv \tilde{X}_w](img/9c5052e2d6faef4a99ede3dacfe3ca8d.png "X_w \equiv \begin{bmatrix} x\\ y\\ z\\ 1\\ \end{bmatrix} \equiv \begin{bmatrix} wx_w\\ wy_w\\ wz_w\\ w\\ \end{bmatrix} \equiv \begin{bmatrix} \tilde{x}\\ \tilde{y}\\ \tilde{z}\\ w\\ \end{bmatrix} \equiv \tilde{X}_w")

有了齐次坐标，我们可以把方程压缩成矩阵乘法。

![\tilde{X}_c = \begin{bmatrix} x_c\\ y_c\\ z_c\\ 1\\ \end{bmatrix} = \begin{bmatrix} R & t \\ 0 & 1 \\ \end{bmatrix} \times \tilde{X}_w = \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x\\ r_{21} & r_{22} & r_{23} & t_y\\ r_{31} & r_{32} & r_{33} & t_z\\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_w\\ y_w\\ z_w\\ 1\\ \end{bmatrix}  \Rightarrow \boxed{ \tilde{X}_c = C_{ex} \times \tilde{X}_w}](img/bf15e602243542ca9789007692828a32.png "\tilde{X}_c = \begin{bmatrix} x_c\\ y_c\\ z_c\\ 1\\ \end{bmatrix} = \begin{bmatrix} R & t \\ 0 & 1 \\ \end{bmatrix} \times \tilde{X}_w = \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x\\ r_{21} & r_{22} & r_{23} & t_y\\ r_{31} & r_{32} & r_{33} & t_z\\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_w\\ y_w\\ z_w\\ 1\\ \end{bmatrix}  \Rightarrow \boxed{ \tilde{X}_c = C_{ex} \times \tilde{X}_w}")

其中![C_{ex}](img/695fa2aac6d8341a7b0ef751e4a9be16.png "C_{ex}")

is the matrix that holds the orientation and position of the camera coordinate frame. We can call this matrix the **Camera Extrinsic** since it represents values like rotation and translation, both of which are external properties of the camera.

![C_{ex} = \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x\\ r_{21} & r_{22} & r_{23} & t_y\\ r_{31} & r_{32} & r_{33} & t_z\\ 0 & 0 & 0 & 1 \end{bmatrix} ](img/cf1b370e85abedb7ea55e6de1eabc78d.png "C_{ex} = \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x\\ r_{21} & r_{22} & r_{23} & t_y\\ r_{31} & r_{32} & r_{33} & t_z\\ 0 & 0 & 0 & 1 \end{bmatrix} ")

* * *

### **[射影变换](#TOC)**

我们从一个点开始![P](img/eeeebd93c479d01033cf1c82fcc5c60a.png "P")

and its (homogeneous) world coordinates ![\tilde{X}_w](img/ebf592957ef823a32ba98db0411597af.png "\tilde{X}_w"). With the help of the camera extrinsic matrix ![C_{ex}](img/695fa2aac6d8341a7b0ef751e4a9be16.png "C_{ex}"), ![\tilde{X}_w](img/ebf592957ef823a32ba98db0411597af.png "\tilde{X}_w")was transformed into its (homogeneous) camera coordinates ![\tilde{X}_c](img/bff9fc6e8062ccb516850583cb63e82f.png "\tilde{X}_c").

Now we come to the final stage of actually materializing an image from the 3D camera coordinates ![\tilde{X}_c](img/bff9fc6e8062ccb516850583cb63e82f.png "\tilde{X}_c")as shown in **Figure 7**.

要理解射影变换，我们唯一需要的就是相似三角形。我们来做一个类似三角形的入门。

我们已经在图 8 和图 9 中看到了类似的三角形。具有相似的三角形

![\displaystyle\frac{AB}{A' B'} = \displaystyle\frac{CB}{C' B'}](img/25f1afe140a5def1b8397910de0e6bce.png "\displaystyle\frac{AB}{A' B'} = \displaystyle\frac{CB}{C' B'}")

是的，你猜对了，![AB'C'](img/b440f053ff942d15100ab1cef2f60233.png "AB'C'")

and ![ABP](img/91f5559c6b13bbc0521a4aa656e6b12d.png "ABP")are similar triangles in **Figure 10**.

从相似三角形的性质，我们可以推导出

![\displaystyle\frac{x_i}{f} = \displaystyle\frac{x_c}{z_c}; \space \displaystyle\frac{y_i}{f} = \displaystyle\frac{y_c}{z_c}](img/0887e274e65aaf1cdc2a876984448e26.png "\displaystyle\frac{x_i}{f} = \displaystyle\frac{x_c}{z_c}; \space \displaystyle\frac{y_i}{f} = \displaystyle\frac{y_c}{z_c}")

因此，它遵循:

![x_i = f\displaystyle\frac{x_c}{z_c}\\ \space \\ y_i = f\displaystyle\frac{y_c}{z_c}\\ ](img/80b56c2c61a859a6a6a52cd20c8c38fb.png "x_i = f\displaystyle\frac{x_c}{z_c}\\ \space \\ y_i = f\displaystyle\frac{y_c}{z_c}\\ ")

现在，重要的是要记住，实际的图像平面不是虚拟平面，而是图像传感器阵列。3D 场景落在该传感器上，这导致图像的形成。因此![x_i](img/9b499ed0ed29ef5ab0b7262cb1e66493.png "x_i")

and ![y_i](img/40e25c331d79edf50a2dbfe6d128a233.png "y_i")in the image plane can be substituted with pixel values ![u,v](img/2bd13bd23c6fd9e8402f9713fd0ddefc.png "u,v").

![u = f\displaystyle\frac{x_c}{z_c} \\ \space \\ v = f\displaystyle\frac{y_c}{z_c} \\ ](img/dea77942f046f0a4323db8993320b147.png "u = f\displaystyle\frac{x_c}{z_c} \\ \space \\ v = f\displaystyle\frac{y_c}{z_c} \\ ")

图像平面中的像素从左上角`(0, 0)`开始，因此也需要相对于图像平面的中心移动像素。

![u = f\displaystyle\frac{x_c}{z_c} + o_x \\ \space \\ v = f\displaystyle\frac{y_c}{z_c} + o_y](img/e39a7561b81916d7b2ec4b395a6b42ff.png "u = f\displaystyle\frac{x_c}{z_c} + o_x \\ \space \\ v = f\displaystyle\frac{y_c}{z_c} + o_y")

在这里，![o_x](img/8c06bb43a7c3669544cf911ad6f4d99e.png "o_x")

and ![o_y](img/2a60eae51092f4a8af4420fb6eae4517.png "o_y")are the center points of the image plane.

现在我们有一个来自 3D 摄像机空间的点，用![u,v](img/2bd13bd23c6fd9e8402f9713fd0ddefc.png "u,v")来表示

in the image plane. Again to make matrices agree, we have to express the pixel values using homogeneous representation.

![u,v](img/2bd13bd23c6fd9e8402f9713fd0ddefc.png "u,v")的齐次表示

, where ![u = \displaystyle\frac{\tilde{u}}{\tilde{w}}](img/0777579d12945497cdd30e77126a418b.png "u = \displaystyle\frac{\tilde{u}}{\tilde{w}}")and ![v = \displaystyle\frac{\tilde{v}}{\tilde{w}}](img/7c8a392e2f0164065ec0b047c54716bc.png "v = \displaystyle\frac{\tilde{v}}{\tilde{w}}")

![\begin{bmatrix} u\\ v\\ 1\\ \end{bmatrix} \equiv \begin{bmatrix} \tilde{u}\\ \tilde{v}\\ \tilde{w}\\ \end{bmatrix} \equiv \begin{bmatrix} z_cu\\ z_cv\\ z_c\\ \end{bmatrix} ](img/c9d7418b12f30cf0fff348144003b247.png "\begin{bmatrix} u\\ v\\ 1\\ \end{bmatrix} \equiv \begin{bmatrix} \tilde{u}\\ \tilde{v}\\ \tilde{w}\\ \end{bmatrix} \equiv \begin{bmatrix} z_cu\\ z_cv\\ z_c\\ \end{bmatrix} ")

这可以进一步表示为:

![\begin{bmatrix} z_cu\\ z_cv\\ z_c\\ \end{bmatrix}= \begin{bmatrix} fx_c +z_co_x\\ fy_c +y_co_y\\ z_c\\ \end{bmatrix} = \begin{bmatrix} f & 0 & o_x & 0\\ 0 & f & o_y & 0\\ 0 & 0 & 1 & 0\\ \end{bmatrix} \begin{bmatrix} x_c\\ y_c\\ z_c\\ 1\\ \end{bmatrix}](img/b930def3a6113cc1778341e030906429.png "\begin{bmatrix} z_cu\\ z_cv\\ z_c\\ \end{bmatrix}= \begin{bmatrix} fx_c +z_co_x\\ fy_c +y_co_y\\ z_c\\ \end{bmatrix} = \begin{bmatrix} f & 0 & o_x & 0\\ 0 & f & o_y & 0\\ 0 & 0 & 1 & 0\\ \end{bmatrix} \begin{bmatrix} x_c\\ y_c\\ z_c\\ 1\\ \end{bmatrix}")

最后，我们有:

![\begin{bmatrix} \tilde{u}\\ \tilde{v}\\ \tilde{w}\\ \end{bmatrix} = \begin{bmatrix} f & 0 & o_x & 0\\ 0 & f & o_y & 0\\ 0 & 0 & 1 & 0\\ \end{bmatrix} \begin{bmatrix} x_c\\ y_c\\ z_c\\ 1\\ \end{bmatrix} ](img/fbe746c998164db2e894c5e4340dfdb6.png "\begin{bmatrix} \tilde{u}\\ \tilde{v}\\ \tilde{w}\\ \end{bmatrix} = \begin{bmatrix} f & 0 & o_x & 0\\ 0 & f & o_y & 0\\ 0 & 0 & 1 & 0\\ \end{bmatrix} \begin{bmatrix} x_c\\ y_c\\ z_c\\ 1\\ \end{bmatrix} ")

这可以简单地表达为

![\tilde{u} = C_{in} \times \tilde{x}_c](img/9b36fd5d4dc0f76ca68a9f7e99bca997.png "\tilde{u} = C_{in} \times \tilde{x}_c")

其中![\tilde{x}_c](img/7c5bbaf5b1967d1dc2430b8c80f2166c.png "\tilde{x}_c")

is the set of vectors containing the location of the point in camera coordinate space and ![\hat{u}](img/b9f88fd0dc534c8bf0fa571b772a1f13.png "\hat{u}")is the set of values containing the location of the point on the image plane. Respectively, ![C_{in}](img/71f76088e0a3c29c8e0a34558bcc0c78.png "C_{in}")represents the set of values needed to map a point from the 3D camera space to the 2D space.

![C_{in} = \begin{bmatrix} f & 0 & o_x & 0\\ 0 & f & o_y & 0\\ 0 & 0 & 1 & 0\\ \end{bmatrix} ](img/90373b4e27a14bffb32c86834fb20945.png "C_{in} = \begin{bmatrix} f & 0 & o_x & 0\\ 0 & f & o_y & 0\\ 0 & 0 & 1 & 0\\ \end{bmatrix} ")

我们可以叫![C_{in}](img/71f76088e0a3c29c8e0a34558bcc0c78.png "C_{in}")

the **camera intrinsic** since it represents values like focal length and center of the image plane along ![x](img/475aa77643ede4a370076ea4f6abd268.png "x")and ![y](img/1308649485b87d470e3ba53ff87b8970.png "y")axes, both of which are internal properties of the camera.

* * *

### **[数据集](#TOC)**

理论够了！给我看一些代码。

在本节中，我们将讨论我们将要处理的数据。作者开源了他们的数据集，你可以在这里找到它。数据集的链接发表在 [NeRF](https://github.com/bmild/nerf) 的官方知识库中。数据集的结构如图**图 11** 所示。

有两个文件夹，`nerf_synthetic`和`nerf_llff_data`。接下来，我们将使用这个系列的合成数据集。

让我们看看`nerf_synthetic`文件夹里有什么。`nerf_synthetic`文件夹中的数据如图**图 12** 所示。

这里有很多人造物体。让我们下载其中一个，看看里面有什么。我们选择了“ship”数据集，但是可以随意下载其中的任何一个。

解压缩数据集后，您会发现包含图像的三个文件夹:

*   `train`
*   `val`
*   `test`

以及包含照相机的方向和位置的三个文件。

*   `transforms_train.json`
*   `transforms_val.json`
*   `transforms_test.json`

为了更好地理解 json 文件，我们可以打开一个[空白的 Colab 笔记本](https://research.google.com/colaboratory/)，上传`transforms_train.json`。我们现在可以对它进行探索性的数据分析。

```py
# import the necessary packages
import json
import numpy as np

# define the json training file
jsonTrainFile = "transforms_train.json"

# open the file and read the contents of the file
with open(jsonTrainFile, "r") as fp:
    jsonTrainData = json.load(fp)

# print the content of the json file
print(f"[INFO] Focal length train: {jsonTrainData['camera_angle_x']}")
print(f"[INFO] Number of frames train: {len(jsonTrainData['frames'])}")

# OUTPUT
# [INFO] Focal length train: 0.6911112070083618
# [INFO] Number of frames train: 100
```

我们从在**线 2 和 3** 上导入必要的包`json`和`numpy`开始。

然后我们加载 json 并在**第 6-10 行**读取它的值。

JSON 文件有两个父键，分别叫做`camera_angle_x`和`frames`。我们看到`camera_angle_x`对应于相机的视野，`frames`是每个图像(帧)的元数据集合。

在**第 13 行和第 14 行，**我们打印 json 键的值。**第 17 行和第 18 行**显示输出。

让我们再深入调查一下`frames`。

```py
# grab the first frame
firstFrame = jsonTrainData["frames"][0]

# grab the transform matrix and file name
tMat = np.array(firstFrame["transform_matrix"])
fName = firstFrame["file_path"]

# print the data
print(tMat)
print(fName)

# OUTPUT
# array([[-0.92501402,  0.27488998, -0.26226836, -1.05723763],
#       [-0.37993318, -0.66926789,  0.63853836,  2.5740304 ],
#       [ 0\.        ,  0.6903013 ,  0.72352195,  2.91661024],
#       [ 0\.        ,  0\.        ,  0\.        ,  1\.        ]])
# ./train/r_0
```

我们在第 20 行抓取第一帧**。每一帧都是一个字典，包含两个键`transform_matrix`和`file_path`，如**第 23 行和第 24 行**所示。`file_path`是考虑中的图像(帧)的路径，`transform_matrix`是该图像的摄像机到世界矩阵。

在**第 27 行和第 28 行，**我们打印了`transform_matrix`和`file_path`。**第 31-35 行**显示输出。**

* * *

## **[汇总](#TOC)**

在本教程中，我们研究了计算机图形学中的一些基本主题。这对于理解 NeRF 至关重要。虽然这是基本的，但仍然是向前迈进的重要一步。

我们可以回忆一下我们在**三个简单步骤**中学到的内容:

1.  正向成像模式(拍照)
2.  世界到相机(3D 到 3D)转换
3.  相机到图像(3D 到 2D)转换

此时，我们也熟悉了所需的数据集。这涵盖了所有的先决条件。

下周我们将看看这篇论文的各种基本概念: [NeRF:将场景表示为用于视图合成的神经辐射场](https://arxiv.org/abs/2003.08934)。我们还将学习如何使用 TensorFlow 和 Python 实现这些概念。

我们希望你喜欢这个教程，一定要下载数据集并尝试一下。

### **[引用信息](#TOC)**

**gothipaty，A. R .，和 Raha， **R.**** “使用 TensorFlow 和 Keras 的 NeRF 的计算机图形学和深度学习:第 1 部分”， *PyImageSearch* ，2021，[https://PyImageSearch . com/2021/11/10/Computer-Graphics-and-Deep-Learning-with-NeRF-using-tensor flow-and-Keras-Part-1/](https://pyimagesearch.com/2021/11/10/computer-graphics-and-deep-learning-with-nerf-using-tensorflow-and-keras-part-1/)

```py
@article{Gosthipaty_Raha_2021_pt1,
    author = {Aritra Roy Gosthipaty and Ritwik Raha},
    title = {Computer Graphics and Deep Learning with {NeRF} using {TensorFlow} and {Keras}: Part 1},
    journal = {PyImageSearch},
    year = {2021},
    note = {https://pyimagesearch.com/2021/11/10/computer-graphics-and-deep-learning-with-nerf-using-tensorflow-and-keras-part-1/},   
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***