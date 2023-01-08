# GAN 培训挑战:用于彩色图像的 DCGAN

> 原文：<https://pyimagesearch.com/2021/12/13/gan-training-challenges-dcgan-for-color-images/>

在本教程中，您将学习如何训练 DCGAN 生成彩色时尚图像。在培训过程中，您将了解常见挑战、应对这些挑战的技术以及 GAN 评估指标。

本课是 GAN 系列教程的第三篇:

1.  [*生成对抗网络简介*](https://pyimagesearch.com/2021/09/13/intro-to-generative-adversarial-networks-gans/)
2.  [*入门:DCGAN for Fashion-MNIST*](https://pyimagesearch.com/2021/11/11/get-started-dcgan-for-fashion-mnist/)
3.  *GAN 训练* *挑战:彩色图像的 DCGAN】(本帖)*

**要了解如何训练 DCGAN 生成彩色时尚图像，以及常见的 GAN 训练挑战和最佳实践，** ***继续阅读。***

在我之前的文章中，[开始:时尚 MNIST 的 DCGAN](https://pyimagesearch.com/2021/11/11/get-started-dcgan-for-fashion-mnist/)，你学习了如何训练一个 DCGAN 来生成灰度时尚 MNIST 图像。在这篇文章中，让我们用彩色图像训练一个 DCGAN，以展示 GAN 训练的常见挑战。我们还将简要讨论一些改进技术和 GAN 评估指标。请跟随 Colab 笔记本[的教程，这里](https://github.com/margaretmz/GANs-in-Art-and-Design/blob/main/3_dcgan_color_images.ipynb)是完整的代码示例。

### 用于彩色图像的 DCGAN

我们将从我之前的[帖子](https://pyimagesearch.com/2021/11/11/get-started-dcgan-for-fashion-mnist/)中获取 DCGAN 代码作为基线，然后对训练彩色图像进行调整。由于我们已经在我的上一篇文章中详细介绍了 DCGAN 的端到端培训，现在我们将只关注为彩色图像培训 DCGAN 所需的关键变化:

1.  **数据:**从 Kaggle 下载彩色图像，预处理到`[-1, 1]`范围。
2.  **生成器:**调整如何对模型架构进行上采样，以生成彩色图像。
3.  **鉴别器:**将输入图像形状从`28×28×1`调整到`64×64×3`。

有了这些改变，你就可以开始在彩色图像上训练 DCGAN 了；然而，当处理彩色图像或除 MNIST 或时尚-MNIST 之外的任何数据时，您会意识到 GAN 培训是多么具有挑战性。甚至用时尚 MNIST 的灰度图像进行训练也可能很棘手。

#### 1.准备数据

我们将使用 Kaggle 的一个名为[服装&模特](https://www.kaggle.com/dqmonn/zalando-store-crawl)的数据集来训练 DCGAN，这个数据集是从[Zalando.com](http://zalando.com)那里收集的服装碎片。有六个类别，超过 16k 的彩色图像，大小为`606×875`，将调整到`64×64`进行训练。

要从 Kaggle 下载数据，您需要提供您的 Kaggle 凭据。您可以将 Kaggle json 文件上传到 Colab，或者将您的 Kaggle 用户名和密钥放在笔记本中。我们选择了后者。

```py
os.environ['KAGGLE_USERNAME']="enter-your-own-user-name" 
os.environ['KAGGLE_KEY']="enter-your-own-user-name" 
```

下载并解压到一个名为`dataset`的目录。

```py
!kaggle datasets download -d dqmonn/zalando-store-crawl -p datasets
!unzip datasets/zalando-store-crawl.zip -d datasets/
```

下载并解压缩数据后，我们设置数据所在的目录。

```py
zalando_data_dir = "/content/datasets/zalando/zalando/zalando"
```

然后，我们使用 Keras' `image_dataset_from_directory`从目录中的图像创建一个`tf.data.Dataset`，它将在稍后用于训练模型。最后，我们指定了`64×64`的图像大小和`32`的批量大小。

```py
train_images = tf.keras.utils.image_dataset_from_directory(
   zalando_data_dir, label_mode=None, image_size=(64, 64), batch_size=32)
```

让我们在**图 1** 中想象一个训练图像作为例子:

和以前一样，我们将图像归一化到`[-1, 1]`的范围，因为生成器的最终层激活使用了`tanh`。最后，我们通过使用`tf.dataset`的`map`函数和`lambda`函数来应用规范化。

```py
train_images = train_images.map(lambda x: (x - 127.5) / 127.5)
```

#### 2.发电机

我们在`build_generator`函数中使用 keras `Sequential` API 创建生成器架构。在我之前的【DCGAN 帖子中，我们已经讨论了如何创建生成器架构的细节。下面我们来看看如何**调整上采样**来生成期望的`64×64×3`彩色图像尺寸:

*   我们为彩色图像更新`CHANNELS = 3`，而不是为灰度图像更新 1。
*   2 的步幅是宽度和高度的一半，所以你可以向后算出最初的图像尺寸:对于时尚 MNIST，我们向上采样为`7 -> 14 -> 28`。现在，我们正在使用大小为`64×64`的训练图像，因此我们会像`8 -> 16 -> 32 -> 64`一样向上采样几次。这意味着我们多加了一套`Conv2DTranspose -> BatchNormalization -> ReLU`。

生成器的另一个变化是将内核大小从 5 更新为 4，以避免减少生成图像中的棋盘状伪像(参见**图 2** )。

这是因为根据后[解卷积和棋盘伪影](https://distill.pub/2016/deconv-checkerboard/)，内核大小 5 不能被步幅 2 整除。所以解决方案是使用内核大小 4 而不是 5。

我们可以在**图 3** 中看到 DCGAN 发生器的架构:

通过调用**图 4** 中的`generator.summary()`，用代码可视化生成器架构:

#### 3.鉴别器

鉴别器架构的主要变化是图像输入形状:我们使用的是形状`[64, 64, 3]`而不是`[28, 28, 1]`。我们还增加了一组`Conv2D -> BatchNormalization -> LeakyReLU`来平衡上面提到的生成器中增加的架构复杂性。其他一切都保持不变。

我们可以在**图 5** 中看到 DCGAN 鉴频器架构:

通过调用**图 6** 中的`discriminator.summary()`，用代码可视化鉴别器架构:

#### DCGAN 模型

我们再次通过子类`keras.Model`定义 DCGAN 模型架构，并覆盖`train_step`来定义定制的训练循环。代码中唯一的细微变化是对真正的标签应用单侧标签平滑。

```py
     real_labels = tf.ones((batch_size, 1))
     real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))
```

这种技术降低了鉴频器的过度自信，因此有助于稳定 GAN 训练。参考 Adrian Rosebrock 的帖子[使用 Keras、TensorFlow 和深度学习进行标签平滑](https://pyimagesearch.com/2019/12/30/label-smoothing-with-keras-tensorflow-and-deep-learning/)了解标签平滑的一般详细信息。在[训练 GANs 的改进技术](https://arxiv.org/abs/1606.03498)一文中提出了用于正则化 GAN 训练的“单侧标签平滑”技术，在这里你也可以找到其他的改进技术。

#### 为培训监控定义 Kera `Callback`

没有变化的相同代码—覆盖 Keras 回调以在训练期间监视和可视化生成的图像。

```py
class GANMonitor(keras.callbacks.Callback):
    def __init__():
    ...
    def on_epoch_end():
    ...
    def on_train_end():
    ...
```

#### 训练 DCGAN 模型

这里我们将 dcgan 模型和 DCGAN 类放在一起:

```py
dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
```

编译 dcgan 模型，主要变化是学习率。在这里，我将鉴别器学习率设置为`0.0001`，将发电机学习率设置为`0.0003`。这是为了确保鉴别器不会超过发生器的功率。

```py
D_LR = 0.0001 # discriminator learning rate
G_LR = 0.0003 # generator learning rate

dcgan.compile(
   d_optimizer=keras.optimizers.Adam(learning_rate=D_LR, beta_1 = 0.5),
   g_optimizer=keras.optimizers.Adam(learning_rate=G_LR, beta_1 = 0.5), 
   loss_fn=keras.losses.BinaryCrossentropy(),
)
```

现在我们干脆调用`model.fit()`来训练`dcgan`模型！

```py
NUM_EPOCHS = 50 # number of epochs
dcgan.fit(train_images, epochs=NUM_EPOCHS, 
callbacks=[GANMonitor(num_img=16, latent_dim=LATENT_DIM)])
```

以下是生成器在整个 DCGAN 训练过程中创建的图像截图(**图 7** ):

### GAN 培训挑战

现在我们已经完成了用彩色图像训练 DCGAN。让我们来讨论一下 GAN 训练的一些常见挑战。

gan 很难训练，以下是一些众所周知的挑战:

1.  不收敛:不稳定，消失梯度，或缓慢训练
2.  模式崩溃
3.  难以评估

#### 不收敛

与训练其他模型(如图像分类器)不同，训练期间 D 和 G 的损失或准确性仅单独测量 D 和 G，而不测量 GAN 的整体性能以及生成器创建图像的能力。当发生器和鉴频器之间达到平衡时，GAN 模型是“好”的，通常鉴频器的损耗约为 0.5。

**甘练不稳:**很难让 D 和 G 保持平衡达到一个均衡。看看训练中的损失，你会发现它们可能会剧烈波动。D 和 G 都可能停滞不前，永远无法改善。长时间的训练并不总是让发电机变得更好。发生器的图像质量可能会随着时间而恶化。

**消失梯度:**在定制训练循环中，我们讨论了如何计算鉴频器和发电机损耗，计算梯度，然后使用梯度进行更新。发生器依靠鉴别器的反馈进行改进。如果鉴别器如此强大，以至于它压倒了生成器:它可以分辨出每次都有假图像，那么生成器就停止了训练。

您可能会注意到，有时生成的图像即使经过一段时间的训练，质量仍然很差。这意味着模型无法在鉴别器和发生器之间找到平衡。

**实验:**让 D 架构强很多(模型架构中的参数更多)或者比 G 训练得更快(比如把 D 的学习率提高到比 G 的学习率高很多)。

#### 模式崩溃

当生成器重复生成相同的图像或训练图像的小子集时，会发生模式崩溃。一个好的生成器应该生成各种各样的图像，这些图像在所有类别上都类似于训练图像。当鉴别器不能辨别出生成的图像是假的时，就会发生模式崩溃，因此生成器会不断生成相同的图像来欺骗鉴别器。

实验:为了模拟代码中的模式崩溃问题，尝试将噪声向量维数从 100 减少到 10；或者将噪声向量维数从 100 增加到 128 以增加图像多样性。

#### 难以评估

评估 GAN 模型具有挑战性，因为没有简单的方法来确定生成的图像是否“好”。与图像分类器不同，根据基本事实标签，预测是正确的还是不正确的。这就引出了下面关于我们如何评估 GAN 模型的讨论。

### GAN 评估指标

成功的生成器有两个标准——它应该生成具有以下特征的图像:

1.  良好的**品质:**高保真逼真，
2.  **多样性**(或多样性):训练图像不同类型(或类别)的良好表示。

我们可以用一些度量标准定性地(视觉检查图像)或定量地评估模型。

**通过目视检查进行定性**评估。正如我们在 DCGAN 训练中所做的那样，我们查看在同一种子上生成的一组图像，并在视觉上检查随着训练的进行图像是否看起来更好。这对于一个玩具例子来说是可行的，但是对于大规模训练来说太耗费人力了。

初始得分(IS)和弗雷歇初始距离(FID)是定量比较 GAN 模型**的两个流行指标。**

本文介绍了**初始评分**:[训练 GANs 的改进技术](https://arxiv.org/abs/1606.03498)。它测量生成图像的质量和多样性。这个想法是使用 inception 模型对生成的图像进行分类，并使用预测来评估生成器。分数越高，表示模型越好。

**弗雷歇初始距离** (FID)也使用初始网络进行特征提取并计算数据分布。FID 通过查看生成的图像和训练图像而不是孤立地只查看生成的图像来改进 IS。较低的 FID 意味着生成的图像更接近真实图像，因此是更好的 GAN 模型。

## **总结**

在这篇文章中，你已经学会了如何训练 DCGAN 生成彩色时尚图片。您还了解了 GAN 培训的常见挑战、一些改进技巧以及 GAN 评估指标。在我的下一篇文章中，我们将学习如何进一步提高瓦瑟斯坦甘(WGAN)和瓦瑟斯坦甘梯度惩罚(WGAN-GP)的训练稳定性。

### **引用信息**

**Maynard-Reid，m .**“GAN 训练挑战:用于彩色图像的 DCGAN”， *PyImageSearch* ，2021，[https://PyImageSearch . com/2021/12/13/GAN-Training-Challenges-DCGAN-for-Color-Images/](https://pyimagesearch.com/2021/12/13/gan-training-challenges-dcgan-for-color-images/)

```py
@article{Maynard-Reid_2021_GAN_Training,
  author = {Margaret Maynard-Reid},
  title = {{GAN} Training Challenges: {DCGAN} for Color Images},
  journal = {PyImageSearch},
  year = {2021},
  note = {https://pyimagesearch.com/2021/12/13/gan-training-challenges-dcgan-for-color-images/},
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***