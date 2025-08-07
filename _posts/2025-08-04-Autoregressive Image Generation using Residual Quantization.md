# CVPR 2022 Autoregressive Image Generation using Residual Quantization

![main](images/2025-08-04-Autoregressive%20Image%20Generation%20using%20Residual%20Quantization/main.png)

### Preface

VAE（Variational Autoencoder）模型是Diffusion模型的发展的基础，而VAE模型本身也有相当的应用场景和空间，如Latent Diffusion Model里就是用VAE模型将高分辨率图像映射到Latent Space上再进行Diffusion操作的。VAE通过引入高斯先验与 KL 正则化，将输入编码为连续潜向量并重构输出，实现了生成式建模的基础结构；VQ‑VAE则将连续潜空间量化为离散代码，避免 post‑erior collapse 并方便使用离散 token 建模 ；而这篇2022年发表在CVPR上的 RQ‑VAE，则是对VQ-VAE的改进，将 residual vector quantization 引入潜空间，即对编码残差进行逐层量化，用较小码本与固定 token 数生成 richer 离散表达，同时支持更高效的序列压缩，显著提升高分辨率图像生成质量与速度。

### VAE

首先，还是简要介绍一下VAE模型的内容。VAE的架构图如下所示：

![vae](images/2025-08-04-Autoregressive%20Image%20Generation%20using%20Residual%20Quantization/vae.png)

VAE有两部分组成，即Encoder和Decoder，Encoder把图像编码为潜空间上的一个高斯分布，而Decoder则负责从高斯分布中采样，并还原为图像，通过Reconstruction Loss进行训练。

为什么是高斯分布呢，这是从AE进行改进得到的，AE模型的Encoder会把图像映射成一个单一的向量，而Decoder利用这个向量生成重构的图像。但是研究发现AE模型的生成能力太差，原因就出在向量上，因此为了让生成能力提升以及保证生成图像不要过于单一，VAE在AE的基础上进行了改进，就是把单一向量变成了高斯分布，这样的学习效果更好，同时因为采样的随机性，生成内容也会更加丰富。

从数学上来说，设当前的输入图像为$$\bold{x}$$，潜空间的向量为$$\bold{z}$$，基于上述的介绍，我们希望
$$
p(\bold{z}\mid\bold{x})=\mathcal{N}(\bold{z}\mid\boldsymbol{\mu},\operatorname{diag}(\boldsymbol{\sigma}^2))\tag{1},
$$
以及
$$
q(\bold{x}\mid\bold{z})=\mathcal{N}(\bold{x}\mid\boldsymbol{\mu'},\sigma_{dec}^2\bold{I})\tag{2},
$$
因为高斯分布有很多比较好的性质，通常认为给定$$\bold{z}$$，$$\bold{x}$$的分布也服从高斯分布（什么分布都可以，只是高斯分布好用，而且高维的高斯分布确实能建模出很多内容）。

有了数学层面的公式后，现在的问题是，这里的参数$$\boldsymbol{\mu},\operatorname{diag}(\boldsymbol{\sigma}^2),\boldsymbol{\mu'}$$都该怎么求呢？很简单，不能数学建模的参数就用神经网络进行学习：
$$
(\boldsymbol{\mu},\boldsymbol{\sigma}^2)=\operatorname{EncoderNetwork_{\phi}(\bold{x})}\\
\boldsymbol{\mu'}=\operatorname{DecoderNetwork}_{\theta}(\bold{z}),\tag{3}
$$
定义了神经网络的参数符号后，把上面两个概率分布重写为$$q_{\phi}(\bold{z}\mid\bold{x}),p_{\theta}(\bold{x}\mid\bold{z})$$（这里笔者也不知道为什么很多paper和tutorial喜欢把p，q反过来，可能是怕混淆？），最后的损失函数就是重构损失和KL散度求和，为
$$
\mathcal{L}(\bold{x};\phi,\theta)=||\bold{x}-\hat{\bold{x}}||^2+\operatorname{D}_{KL}(p(\bold{z}\mid\bold{x})||\mathcal{N}(p(\bold{z})))\\=||\bold{x}-\hat{\bold{x}}||^2+\operatorname{D}_{KL}(q_{\theta}(\bold{z}\mid\bold{x})||\mathcal{N}(p(\bold{z})))\\
=||\bold{x}-\hat{\bold{x}}||^2+\operatorname{D}_{KL}(\mathcal{N}(\boldsymbol{\mu}(\bold{x}),\boldsymbol{\sigma}^2(\bold{x}))||\mathcal{N}(0,\sigma^2\bold{I})),\tag{4}
$$

### VQ-VAE

简要讲过了VAE的核心内容，这一部分就进入到VQ-VAE，全称Vector Quantised Variational AutoEncoder，简单来说，是离散化版本的VAE，架构图如下：

![vqvae](images/2025-08-04-Autoregressive%20Image%20Generation%20using%20Residual%20Quantization/vqvae.png)

所谓离散化，就是Latent Space上的向量不再采用连续的高斯分布，而是用一堆离散的向量来表示，这一堆向量叫做codebook。这时VAE的生成pipeline为：通过Encoder将图像编码为离散向量，然后找出codebook里与该向量最接近的几个向量，随后导入Decoder中进行图像的生成。至于为什么要采用离散化，笔者认为，离散的 codebook 使得 Decoder 始终从训练中真实出现过的向量中生成，而不是借由连续分布进行插值。这样生成时重建误差比较小、图像更加清晰、不模糊，生成效果相较于VAE更加出色。

VQ-VAE的优势比较好理解，但是离散化带来了一系列问题：模型该怎么训练？毕竟梯度在codebook两端无法传递；codebook里的向量又该怎么训练？

为了解决训练的问题，VQ-VAE采用了一共三个loss，即reconstruction loss，alignment loss以及commitment loss。

reconstruction loss很好理解，让输入图像$$\bold{x}$$和输出图像$$\hat{\bold{x}}$$一致，这个loss能同时优化encoder和decoder的参数，以及这里就会设计到前面提的一个重要问题：梯度如何传递。既然梯度在codebook端传递不了，那么就直接把**梯度从decoder跳过codebook直接反传到encoder**。为什么能这么做呢？因为encoder的末端输出维度和decoder的首端输出维度相等，都为codebook embedding的维度，所以梯度传导到decoder的输入端时，可以直接把它转移到encoder的末端，然后对encoder进行梯度下降；

commitment loss则是为了让encoder生成的向量和codebook里的向量对齐，设encoder输出的embedding向量为$$z_e(\bold{x})$$（**注：encoder的真正输出应该是feature map，所谓的embedding向量是feature map的一个个patch，如上图中的结构**），通过查询得到的codebook里的最近的向量为$$\bold{e}$$，则alignment loss就是让$$\bold{e}$$和$$z_e(\bold{x})$$尽可能接近，在计算这个loss时，codebook的参数被冻结，优化的是encoder的参数；

alignment loss和commitment loss意思一样，只不过alignment loss冻结的是encoder的参数，通过优化codebook的参数让codebook的向量尽可能向encoder的输出靠近。

综合上述三个loss，用数学公式进行表示就是：
$$
L = \underbrace{\|\bold{x} - D(\bold{e}_k)\|_2^2}_{\text{reconstruction loss}}
  + \underbrace{\|\mathrm{sg}(z_e(\bold{x})) - \bold{e}_k\|_2^2}_{\text{alignment loss}}
  + \underbrace{\beta \cdot \|z_e(\bold{x}) - \mathrm{sg}(\bold{e}_k)\|_2^2}_{\text{commitment loss}},\tag{1}
$$
其中$$D$$表示decoder，$$sg$$表示stop-gradient，$$\bold{e}_k$$表示从codebook里选出的最近邻的向量。



### RQ-VAE

现在，知道了VAE和VQ-VAE，RQ-VAE的内容就比较好理解了。

![rqvae](images/2025-08-04-Autoregressive%20Image%20Generation%20using%20Residual%20Quantization/rqvae.png)

RQ-VAE的encoder将输入图像处理为feature map（$$H\times W\times C$$）后，把$$C$$张feature map的像素组合起来，将feature map转变为一系列长度为$$C$$的feature向量，每个feature向量都会通过最近邻在codebook里找到一个embedding，然后将feature map转化为embedding map，而在RQ-VAE中，则采用类似ResNet的思想，找出一系列embedding向量，具体流程如下：

- encoder输出的一个feature向量为$$\bold{Z}$$，令$$r_0=\bold{Z}$$；
- 令$$k_1 = \arg\min_{i}||r_0-e_i||$$，则$$e_{k_1}$$为$$r_0$$在codebook查询出的embedding；
- 令$$r_1=r_0-e_{k_1}$$，$$r_1$$就是残差Residual；
- 再用$$r_1$$进行上述操作，即$$k_i=\arg\min_j||r_{i-1}-e_j||,r_i=r_{i-1}-e_{k_i},\quad i=1,2,\cdots,D$$；
- 最终的输出为$$\hat{\bold{Z}}=\sum_{i=1}^{D}e_{k_i}$$（称为code stack，称呼很好理解）

就像ResNet解决了梯度消失的问题一样，RQ-VAE这种残差化的设计使得在codebook大小不变的情况下，表达能力会显著提升。

RQ-VAE的训练loss则跟VQ-VAE一样。



### RQ-Transformer

这篇论文除了介绍RQ-VAE之外，还提出了与之配套的RQ-Transformer，就是将RQ-VAE里学习到的一系列residual code embedding构成一个长度为$$H\times W\times D$$的序列，对这个序列的所有token都用masked self-attention，训练一个自回归的Transformer，也就是图中的Spatial Transformer；

而Depth Transformer则是利用Spatial Transformer的encoder输出，预测位置分布的概率。

用数学公式再解释一遍上述这些抽象的语言。

RQ-Transformer的输入是RQ-VAE里的residual code embedding：
$$
\bold{S} = \{\bold{S}_t^{(d)} \mid t = 1, \dots, T,\, d = 1, \dots, D\},
\quad T = H \times W,
$$
每一个$$\bold{S}_t^{(d)}$$的维度是RQ-VAE的feature map的维度$$C$$；那么$$\bold{S}$$的联合概率分布则可以写为：
$$
p(\bold{S}) = \prod_{t=1}^{T} \prod_{d=1}^{D}
  p\bigl(\bold{S}_t^{(d)} \mid \bold{S}_{<t},\, \bold{S}_{t,<d}\bigr)\tag{2},
$$
Spatial Transformer的输入token序列为每一批code stack，再加上positional encoding，记为$$\bold{u}={\bold{u}_1,\bold{u}_2,\cdots,\bold{u}_{H\times W}}$$，计算公式为
$$
\bold{u}_t=\operatorname{PE}_T(t)+\sum_{d=1}^{D}\operatorname{e}(\bold{S}_{t-1,d})\tag{3},
$$
其中$$\operatorname{PE}_T$$表示在$$t$$维度（$$H\times W$$）上的Position Embedding，$$\operatorname{e}$$表示Spatial Transformer的encoder；

然后利用masked self-attention生成每个位置的上下文$$\bold{h}_t$$，即
$$
\bold{h}_t=\operatorname{SpatialTransformer}(\bold{u}_1,\bold{u}_2,\cdots,\bold{u}_t)\tag{4},
$$


而Depth Transformer的输入则为
$$
\bold{v}_{td}=\operatorname{PE}_D(d)+\sum_{d'=1}^{d-1}\operatorname{e}(\bold{S}_{td'})\quad for\ d > 1,\tag{5}
$$
这里和Spatial Transformer不同的是，对$$d$$的长度没有限制为$$D$$，是一个全局的masked self-attention。

因此，Depth Transformer可被用来预测联合概率分布（为什么可以这样我没看懂⊙﹏⊙∥），即
$$
\bold{p}_{td}(k)=p(\bold{S}_{td}|\bold{S}_{<t,d},\bold{S}_{t,<d})=\operatorname{DepthTransformer}(\bold{v}_{t1},\cdots,\bold{v}_{td}),\tag{6}
$$
学习到这个分布后，整个的RQ-Transformer也可以用来生成图像了，因为可以根据当前的residual code预测后续的code，最后通过RQ-VAE的Decoder生成图像。



### Experiments

首先是生成图像的质量，VAE能做到这个程度还是很强的：

![exp1](images/2025-08-04-Autoregressive%20Image%20Generation%20using%20Residual%20Quantization/exp1.png)

然后是和其它方法的生成图像FID分数比较：

![exp2](images/2025-08-04-Autoregressive%20Image%20Generation%20using%20Residual%20Quantization/exp2.png)

![exp3](images/2025-08-04-Autoregressive%20Image%20Generation%20using%20Residual%20Quantization/exp3.png)

### Conclusion

笔者是在看推荐系统相关内容的时候看到RQ-VAE的身影的，个人认为这个设计思想以及后续和Transformer的结合是本篇论文的一大亮点。