# 有 WGAN 和 WGAN-GP 的动漫脸

> 原文：<https://pyimagesearch.com/2022/02/07/anime-faces-with-wgan-and-wgan-gp/>

在这篇文章中，我们实现了两种 GAN 变体:Wasserstein GAN (WGAN)和 wasser stein GAN with Gradient Penalty(WGAN-GP)，以解决我在上一篇文章 [GAN 训练挑战:彩色图像的 DCGAN】中讨论的训练不稳定性。我们将训练 WGAN 和 WGAN-GP 模型，生成丰富多彩的`64×64`动漫脸。](https://pyimagesearch.com/2021/12/13/gan-training-challenges-dcgan-for-color-images/)

这是我们 GAN 教程系列的第四篇文章:

1.  [*生成对抗网络简介*](https://pyimagesearch.com/2021/09/13/intro-to-generative-adversarial-networks-gans/)
2.  [*入门:DCGAN for Fashion-MNIST*](https://pyimagesearch.com/2021/11/11/get-started-dcgan-for-fashion-mnist/)
3.  [*GAN 训练挑战:DCGAN 换彩色图像*](https://pyimagesearch.com/2021/12/13/gan-training-challenges-dcgan-for-color-images/)
4.  *有 WGAN 和 WGAN-GP 的动漫脸*(本教程)

我们将首先一步一步地浏览 WGAN 教程，重点介绍 WGAN 论文中介绍的新概念。然后我们讨论如何通过一些改变来改进 WGAN，使之成为 WGAN-GP。

### **瓦瑟斯坦甘**

论文[中介绍了 Wasserstein GAN](https://arxiv.org/abs/1701.07875) 。其主要贡献是利用 Wasserstein 损失解决 GAN 训练不稳定性问题，这是 GAN 训练的一个重大突破。

回想一下在 DCGAN 中，当鉴别器太弱或太强时，它不会给生成器提供有用的反馈来进行改进。训练时间更长并不一定让 DCGAN 模型更好。

有了 WGAN，这些训练问题可以用新的 Wasserstein loss 来解决:我们不再需要在鉴别器和生成器的训练中进行仔细的平衡，也不再需要仔细设计网络架构。WGAN 具有几乎在任何地方都连续且可微的线性梯度(**图 1** )。这解决了常规 GAN 训练的消失梯度问题，

以下是 WGAN 白皮书中介绍的一些新概念或主要变化:

*   Wasserstein distance (或推土机的距离):测量将一种分布转换成另一种分布所需的努力。
*   **瓦瑟斯坦损失:**一个新的损失函数，衡量瓦瑟斯坦距离。
*   在 WGAN 中，这个鉴别器现在被称为**评论家**。我们训练**一个输出数字的评论家**，而不是训练一个鉴别器(二进制分类器)来辨别一个图像是真是假(生成的)。
*   批评家必须满足 [**李普希兹约束**](https://en.wikipedia.org/wiki/Lipschitz_continuity) 才能对瓦瑟斯坦的损失起作用。
*   WGAN 使用**权重剪辑**来执行 1-Lipschitz 约束。

在我们实施每个新的 GAN 架构时，我将重点介绍与以前的 GAN 版本相比的变化，以帮助您了解新概念。以下是 WGAN 与 DCGAN 相比的主要变化:

**表 1** 总结了将 DCGAN 更新为 WGAN 所需的更改:

现在让我们浏览代码，用 TensorFlow 2 / Keras 在 WGAN 中实现这些更改。在遵循下面的教程时，请参考 WGAN Colab 笔记本[这里](https://colab.research.google.com/github/margaretmz/GANs-in-Art-and-Design/blob/main/4_wgan_anime_faces.ipynb)的完整代码。

#### **设置**

首先，我们确保将 Colab 硬件加速器的运行时设置为 GPU。然后我们导入所有需要的库(例如 TensorFlow 2、Keras 和 Matplotlib 等。).

#### **准备数据**

我们将使用来自 Kaggle 的一个名为[动漫人脸数据集](https://www.kaggle.com/splcher/animefacedataset)的数据集来训练 DCGAN，该数据集是从[www.getchu.com](http://www.getchu.com/)刮来的动漫人脸集合。有 63，565 个小彩色图像要调整大小到`64×64`进行训练。

要从 Kaggle 下载数据，您需要提供您的 Kaggle 凭据。你可以上传卡格尔。或者把你的 Kaggle 用户名和密钥放在笔记本里。我们选择了后者。

```py
os.environ['KAGGLE_USERNAME']="enter-your-own-user-name" 
os.environ['KAGGLE_KEY']="enter-your-own-user-name" 
```

将数据下载并解压缩到名为`dataset`的目录中。

```py
!kaggle datasets download -d splcher/animefacedataset -p dataset
!unzip datasets/animefacedataset.zip -d datasets/
```

下载并解压缩数据后，我们设置图像所在的目录。

```py
anime_data_dir = "/content/datasets/images"
```

然后，我们使用`image_dataset_from_directory`的 Keras utils 函数从目录中的图像创建一个`tf.data.Dataset`，它将用于稍后训练模型。我们指定了`64×64`的图像大小和`256`的批量大小。

```py
train_images = tf.keras.utils.image_dataset_from_directory(
   anime_data_dir, label_mode=None, image_size=(64, 64), batch_size=256)
```

让我们想象一个随机的训练图像。

```py
image_batch = next(iter(train_images))
random_index = np.random.choice(image_batch.shape[0])
random_image = image_batch[random_index].numpy().astype("int32")
plt.axis("off")
plt.imshow(random_image)
plt.show()
```

下面是这个随机训练图像在**图 2** 中的样子:

和以前一样，我们将图像归一化到`[-1, 1]`的范围，因为生成器的最终层激活使用了`tanh`。最后，我们通过使用带有`lambda`函数的`tf.dataset`的`map`函数来应用规范化。

```py
train_images = train_images.map(lambda x: (x - 127.5) / 127.5)
```

#### **发电机**

WGAN 发生器架构没有变化，与 DCGAN 相同。我们在`build_generator`函数中使用 Keras `Sequential` API 创建生成器架构。参考我之前两篇 DCGAN 帖子中关于如何创建生成器架构的细节: [DCGAN 用于时尚-MNIST](https://pyimagesearch.com/2021/11/11/get-started-dcgan-for-fashion-mnist/) 和 [DCGAN 用于彩色图像](https://pyimagesearch.com/2021/12/13/gan-training-challenges-dcgan-for-color-images/)。

在`build_generator()`函数中定义了生成器架构之后，我们用`generator = build_generator()`构建生成器模型，并调用`generator.summary()`来可视化模型架构。

#### **评论家**

在 WGAN 中，我们有一个批评家指定一个衡量 Wasserstein 距离的分数，而不是真假图像二进制分类的鉴别器。请注意，评论家的输出现在是一个分数，而不是一个概率。批评家被 1-Lipschitz 连续性条件所约束。

这里有相当多的变化:

*   将`discriminator`重命名为`critic`
*   使用权重剪裁在批评家上实施 1-Lipschitz 连续性
*   将评论家的激活功能从`sigmoid`更改为`linear`

**将`discriminator`更名为`critic`**

如果您从 DCGAN 代码开始，您需要将`discriminator`重命名为`critic`。您可以使用 Colab 中的“查找和替换”功能进行所有更新。

所以现在我们有一个函数叫做`build_critic`而不是`build_discriminator`。

**重量削减**

WGAN 通过使用权重裁剪来实施 1-Lipschitz 约束，我们通过子类化`keras.constraints.Constraint`来实现权重裁剪。详细文件参考 Keras [层重量约束](https://keras.io/api/layers/constraints/.)。下面是我们如何创建`WeightClipping`类:

```py
class WeightClipping(tf.keras.constraints.Constraint):
   def __init__(self, clip_value):
       self.clip_value = clip_value

   def __call__(self, weights):
       return tf.clip_by_value(weights, -self.clip_value, self.clip_value)

   def get_config(self):
       return {'clip_value': self.clip_value}
```

然后在`build_critic`函数中，我们用`WeightClipping`类创建了一个`[-0.01, 0.01]`的`constraint`。

```py
constraint = WeightClipping(0.01)
```

现在我们将`kernel_constraint = constraint`添加到评论家的所有`CONV2D`层中。例如:

```py
model.add(layers.Conv2D(64, (4, 4), 
          padding="same", strides=(2, 2),
          kernel_constraint = constraint, 
          input_shape=input_shape))
```

**线性激活**

在 critic 的最后一层，我们将激活从`sigmoid`更新为`linear`。

```py
model.add(layers.Dense(1, activation="linear"))
```

请注意，在 Keras 中，`Dense`层默认有`linear`激活，所以我们可以省略`activation="linear"`部分，编写如下代码:

```py
model.add(layers.Dense(1))
```

我把`activation = "linear"`留在这里是为了说明，在将 DCGAN 更新为 WGAN 时，我们将从`sigmoid`变为`linear`激活。

现在我们已经在`build_critic`函数中定义了模型架构，让我们用`critic = build_critic(64, 64, 3)`构建 critic 模型，并调用`critic.summary()`来可视化 critic 模型架构。

#### **WGAN 模型**

我们通过子类`keras.Model`定义 WGAN 模型架构，并覆盖`train_step`来定义定制的训练循环。

WGAN 的这一部分有一些变化:

*   更新批评家比更新生成器更频繁
*   评论家不再有形象标签
*   使用 Wasserstein 损失代替二元交叉熵(BCE)损失

**更新评论家比更新生成器更频繁**

根据论文建议，我们更新评论家的频率是生成器的 5 倍。为了实现这一点，我们向`WGAN`类的`critic_extra_steps`到`__init__`传递了一个额外的参数。

```py
def __init__(self, critic, generator, latent_dim, critic_extra_steps):
    ...
    self.c_extra_steps = critic_extra_steps
    ...
```

然后在`train_step()`中，我们使用一个`for`循环来应用额外的训练步骤。

```py
for i in range(self.c_extra_steps):
         # Step 1\. Train the critic
         ...

# Step 2\. Train the generator
```

**图像标签**

根据我们如何编写 Wasserstein 损失函数，我们可以 1)将 1 作为真实图像的标签，将负的作为伪图像的标签，或者 2)不分配任何标签。

下面是对这两个选项的简要说明。使用标签时，Wasserstein 损失计算为`tf.reduce mean(y_true * y_pred)`。如果我们有真实图像上的损耗+虚假图像上的损耗和仅虚假图像上的发生器损耗的评论家损耗，那么它导致评论家损耗的`tf.reduce_mean (1 * pred_real - 1 * pred_fake)`和发生器损耗的`-tf.reduce_mean(pred_fake)`。

请注意，评论家的目标并不是试图给自己贴上`1`或`-1`的标签；相反，它试图最大化其对真实图像的预测和对虚假图像的预测之间的差异。因此，在沃瑟斯坦损失的案例中，标签并不重要。

所以我们选择后一种不分配标签的选择，你会看到所有真假标签的代码都被去掉了。

**瓦瑟斯坦损失**

批评家和创造者的损失通过`model.compile`传递:

```py
def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
	super(WGAN, self).compile()
	...
	self.d_loss_fn = d_loss_fn 
	self.g_loss_fn = g_loss_fn
```

然后在`train_step`中，我们使用这些函数分别计算训练期间的临界损耗和发电机损耗。

```py
def train_step(self, real_images):

	for i in range(self.c_extra_steps):
	         # Step 1\. Train the critic
	         ...

	   d_loss = self.d_loss_fn(pred_real, pred_fake) # critic loss

	# Step 2\. Train the generator
	...
	g_loss = self.g_loss_fn(pred_fake) # generator loss
```

#### 用于训练监控的**Keras`Callback`**

与 DCGAN 相同的代码，没有变化—覆盖 Keras `Callback`以在训练期间监控和可视化生成的图像。

```py
class GANMonitor(keras.callbacks.Callback):
    def __init__():
    ...
    def on_epoch_end():
    ...
    def on_train_end():
    ...
```

#### **编译并训练 WGAN**

**组装 WGAN 模型**

我们将`wgan`模型与上面定义的 WGAN 类放在一起。请注意，根据 WGAN 文件，我们需要将评论家的额外培训步骤设置为`5`。

```py
wgan = WGAN(critic=critic,
             generator=generator,
             latent_dim=LATENT_DIM,
             critic_extra_steps=5) # UPDATE for WGAN
```

**瓦瑟斯坦损失函数**

如前所述，WGAN 的主要变化是 Wasserstein loss 的用法。以下是如何计算评论家和发电机的 Wasserstein 损失——通过在 Keras 中定义自定义损失函数。

```py
# Wasserstein loss for the critic
def d_wasserstein_loss(pred_real, pred_fake):
   real_loss = tf.reduce_mean(pred_real)
   fake_loss = tf.reduce_mean(pred_fake)
   return fake_loss - real_loss

# Wasserstein loss for the generator
def g_wasserstein_loss(pred_fake):
   return -tf.reduce_mean(pred_fake)
```

**编译 WGAN**

现在我们用 RMSProp 优化器编译`wgan`模型，按照 WGAN 论文，学习率为 0.00005。

```py
LR = 0.00005 # UPDATE for WGAN: learning rate per WGAN paper
wgan.compile(
   d_optimizer = keras.optimizers.RMSprop(learning_rate=LR, clipvalue=1.0, decay=1e-8), # UPDATE for WGAN: use RMSProp instead of Adam
   g_optimizer = keras.optimizers.RMSprop(learning_rate=LR, clipvalue=1.0, decay=1e-8), # UPDATE for WGAN: use RMSProp instead of Adam
   d_loss_fn = d_wasserstein_loss,
   g_loss_fn = g_wasserstein_loss
)
```

注意在 DCGAN 中，我们使用`keras.losses.BinaryCrossentropy()`而对于 WGAN，我们使用上面定义的自定义`wasserstein_loss`函数。这两个`wasserstein_loss`函数通过`model.compile()`传入。它们将用于自定义训练循环，如上面的覆盖`_step`部分所述。

**训练 WGAN 模型**

现在我们干脆调用`model.fit()`来训练`wgan`模型！

```py
NUM_EPOCHS = 50 # number of epochs
wgan.fit(train_images, epochs=NUM_EPOCHS, callbacks=[GANMonitor(num_img=16, latent_dim=LATENT_DIM)])
```

### **Wasserstein GAN 用梯度罚**

虽然 WGAN 通过 Wasserstein 的损失提高了训练的稳定性，但即使是论文本身也承认“*重量削减显然是一种实施 Lipschitz 约束*的可怕方式。”大的削波参数会导致训练缓慢，并阻止评论家达到最佳状态。与此同时，过小的剪裁很容易导致渐变消失，这正是 WGAN 提出要解决的问题。

本文介绍了 Wasserstein 梯度罚函数(WGAN-GP)，改进了 Wasserstein GANs 的训练。它进一步改进了 WGAN，通过使用梯度惩罚而不是权重剪裁来加强评论家的 1-Lipschitz 约束。

我们只需要做一些更改就可以将 WGAN 更新为 WGAN-WP:

*   从评论家的建筑中删除批量规范。
*   使用**梯度惩罚**而不是权重剪裁来加强 Lipschitz 约束。
*   用亚当**优化器** (α = 0.0002，β [1] = 0.5，β [2] = 0.9)代替 RMSProp。

请参考 WGAN-GP Colab 笔记本[此处](https://colab.research.google.com/github/margaretmz/GANs-in-Art-and-Design/blob/main/4_wgan_gp_anime_faces.ipynb)获取完整的代码示例。在本教程中，我们只讨论将 WGAN 更新为 WGAN-WP 的增量更改。

#### **添加梯度惩罚**

梯度惩罚意味着惩罚具有大范数值的梯度，下面是我们在 Keras 中如何计算它:

```py
def gradient_penalty(self, batch_size, real_images, fake_images):
    """ Calculates the gradient penalty.

    Gradient penalty is calculated on an interpolated image
    and added to the discriminator loss.
    """

    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    # 1\. Create the interpolated image
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 2\. Get the Critic's output for the interpolated image
        pred = self.critic(interpolated, training=True)

    # 3\. Calculate the gradients w.r.t to the interpolated image
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 4\. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    # 5\. Calculate gradient penalty
    gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
    return gradient_penalty
```

然后在`train_step`中，我们计算梯度惩罚，并将其添加到原始评论家损失中。注意惩罚权重(或系数λƛ)控制惩罚的大小，它被设置为每张 WGAN 纸 10。

```py
gp = self.gradient_penalty(batch_size, real_images, fake_images)
d_loss = self.d_loss_fn(pred_real, pred_fake) + gp * self.gp_weight
```

#### **移除批次号**

虽然批归一化有助于稳定 GAN 训练中的训练，但它对梯度惩罚不起作用，因为使用梯度惩罚，我们会单独惩罚每个输入的 critic 梯度的范数，而不是整个批。所以我们需要从批评家的模型架构中移除批量规范代码。

#### **Adam Optimizer 而不是 RMSProp**

DCGAN 使用 Adam 优化器，对于 WGAN，我们切换到 RMSProp 优化器。现在对于 WGAN-GP，我们切换回 Adam 优化器，按照 WGAN-GP 论文建议，学习率为 0.0002。

```py
LR = 0.0002 # WGAN-GP paper recommends lr of 0.0002
d_optimizer = keras.optimizers.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.9) g_optimizer = keras.optimizers.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.9)
```

我们编译和训练 WGAN-GP 模型 50 个时期，并且我们观察到由该模型生成的更稳定的训练和更好的图像质量。

**图 3** 分别比较了真实(训练)图像和 WGAN 和 WGAN-GP 生成的图像。

WGAN 和 WGAN-GP 都提高了训练稳定性。权衡就是他们的训练收敛速度比 DCGAN 慢，图像质量可能会稍差；然而，随着训练稳定性的提高，我们可以使用更复杂的生成器网络架构，从而提高图像质量。许多后来的 GAN 变体采用 Wasserstein 损失和梯度惩罚作为缺省值，例如 ProGAN 和 StyleGAN。甚至 TF-GAN 库默认使用 Wasserstein 损失。

## **总结**

在这篇文章中，你学习了如何使用 WGAN 和 WGAN-GP 来提高 GAN 训练的稳定性。您了解了使用 TensorFlow 2 / Keras 从 DCGAN 迁移到 WGAN，然后从 WGAN 迁移到 WGAN-GP 的增量变化。你学会了如何用 WGAN 和 WGAN-GP 生成动漫脸。

### **引用信息**

**梅纳德-里德，M.** 《有 WGAN 和 WGAN-GP 的动漫脸》， *PyImageSearch* ，2022，【https://pyimg.co/9avys】T4

```py
@article{Maynard-Reid_2022_Anime_Faces,
  author = {Margaret Maynard-Reid},
  title = {Anime Faces with {WGAN} and {WGAN-GP}},
  journal = {PyImageSearch},
  year = {2022},
  note = {https://pyimg.co/9avys},
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***