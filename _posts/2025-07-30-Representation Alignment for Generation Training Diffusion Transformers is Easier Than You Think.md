## (ICLR 2025 oral) Representation Alignment for Generation: Training Diffusion Transformers is Easier Than You Think

![main](images/2025-07-30-Representation%20Alignment%20for%20Generation%20Training%20Diffusion%20Transformers%20is%20Easier%20Than%20You%20Think/main.png)

### Preface 

相信对于不少人来说，Generative Models和Discriminative Models似乎是两个不太搭边的理由，毕竟它们某种程度上代表两种完全相反的方向，Generative Models一般是从内容较少向内容较多的方向，比如说Diffusion从一个简单的标签变换到一个内容丰富的图像，而Discriminative Models则反过来，比如说从一张图像变成一个简单的标签。这两者之间的方法论也似乎没有多少交集。但这篇论文就提出，将Alignment这样一个在Discriminative Models里很常用的概念用在Generative Models中就能有很好的效果。这篇由Saining Xie等人推出的论文则在Diffusion Transformer上进行了对应的改进。

### Diffusion Transformer

讲到这里，笔者简单介绍一下Diffusion Transformer。

![dit](images/2025-07-30-Representation%20Alignment%20for%20Generation%20Training%20Diffusion%20Transformers%20is%20Easier%20Than%20You%20Think/dit.png)

一句话概括Diffusion Transformer：DDPM换了backbone，从U-Net换成Transformer了。

整个的架构其实就是Vision Transformer和LDM的结合体，先把图片通过VAE或Image Encoder编码到Latent Space（为什么这样做：Diffusion生成高分辨率图像太难，于是用VAE架构，先变换到低分辨率的Latent Space，Diffuion过后再用VAE重构成高分辨率图像），然后Patchify成一系列的tokens（16x16 -> 4个4x4）作为序列输入，加上时间、类别等embedding信息，然后经过多个Attention层，提取出features，随后通过线性层重构成与输入noised latent image形状相同的noise。



### Scalable Interpolant Transformer

这一架构简称SiT，是在DiT的基础上进行改进得到，模型架构都是一样的，只是DiT采用的是传统的DDPM式的扩散路径和训练方法，而SiT采取灵活插值和ODE/SDE视角的方式，这样时间步长就是灵活可变的了。

以REPA论文的符号为准，$$\bold{x}_*$$是一张图像，$$\bold{x}_t$$是加噪过后的图像，则有
$$
\bold{x}_t=\alpha_t\bold{x}_*+\sigma_t\epsilon,\quad \alpha_0=\sigma_T=1,\
\alpha_T=\sigma_0=0,
$$
和DDPM一模一样，只是这里的$$\alpha_t$$和$$\sigma_t$$成了连续函数，值取决于t的值，这两个函数应该是提前定义好的，一个单调递增一个单调递减。

为了和ODE/SDE等相匹配，SiT里采用的是velocity的训练方式，也就是对上式求导：
$$
\bold{\dot x}_t=\dot\alpha_t\bold{x}_*+\dot\sigma_t\epsilon=\bold{v}(\bold{x}_t,t),
$$
然后用DiT相同的架构来学习这个速度变化，即$$\bold{v}_{\theta}(\bold{x}_t,t)$$，损失函数为
$$
\mathcal{L}_{velocity}(\theta)=\mathbb{E}_{\bold{x}_*,\epsilon,t}[||\bold{v}_{\theta}(\bold{x}_t,t)-\dot\alpha_t\bold{x}_*-\dot\sigma_t\epsilon||^2],
$$
笔者认为还是比较好理解的，详细内容请参考原文。

### DINOv2

这是当前Vision Transformer的SOTA，在大规模数据集上通过Self-Supervised Distillation自监督训练得到。是这篇论文所采用的一个Pre-Trained Model，可以较好的提取出一张图片的embedding，这篇论文就用DINOv2生成的embedding作为后续改进DiT训练的基础。

### Method

作者首先做了一些实验，发现：

- Pre-Trained SiT尽管能学习到有用的语义表征，但是和DINOv2差别很大（比较方法是训练一个线性分类器，用这些representations进行分类比较准确率）；
- 这种差距随着层数深入和训练时间逐渐减小，但是速度很慢。

![dis1](images/2025-07-30-Representation%20Alignment%20for%20Generation%20Training%20Diffusion%20Transformers%20is%20Easier%20Than%20You%20Think/dis1.png)

因此，作者认为，认为加快这个对齐过程，应该能提高训练速度。

于是，这篇论文就简单的加了一个对齐loss，人为的加快了这个过程。

loss也很简单，$$f$$是一个pretrained encoder，比如DINOv2，$$\bold{y}_x=f(\bold{x}_*)$$是$$f$$的输出，$$h_{\phi}(f_{\theta}(\bold{z}_t))$$为Diffusion Transformer Encoder的输出经MLP线性映射后的最终输出（注意这个$$f_{\theta}$$只是表示Encoder部分，跟$$f$$不是一个内容，只是为了跟原文保持一致），拿这两部分算一个相似度，作为loss，加速这个对齐过程，就是最终的改进方法，即
$$
\mathcal{L}_{REPA}(\theta,\phi)=-\mathbb{E}_{\bold{x}_*,\epsilon,t}[\frac{1}{N}\sum_{n=1}^{N}\operatorname{sim}(\bold{y}_*^{[n]}, h_{\phi}(f_{\theta}(\bold{z}_t)))],
$$
sim表示相似度计算函数，n是patch的索引，这个loss是在各个patch上进行对齐，总的loss为
$$
\mathcal{L}=\mathcal{L}_{velocity}+\lambda\mathcal{L}_{REPA}.
$$
下面这张图非常清晰明了：

![structure](images/2025-07-30-Representation%20Alignment%20for%20Generation%20Training%20Diffusion%20Transformers%20is%20Easier%20Than%20You%20Think/structure.png)

### Experiments

首先从上一张图可以看到，这样改进后，SiT和DiT的训练收敛速度都快了很多，SiT甚至快了17.5倍。

收敛变快，那么同等训练轮次下，生成效果就会更好：

![exp1](images/2025-07-30-Representation%20Alignment%20for%20Generation%20Training%20Diffusion%20Transformers%20is%20Easier%20Than%20You%20Think/exp1.png)

同样的，FID分数也会有显著的降低，代表生成质量更加优秀：

![exp2](images/2025-07-30-Representation%20Alignment%20for%20Generation%20Training%20Diffusion%20Transformers%20is%20Easier%20Than%20You%20Think/exp2.png)

Ablation Study也能很明显的看到，REPA有效提升了representation的质量：

![exp3](images/2025-07-30-Representation%20Alignment%20for%20Generation%20Training%20Diffusion%20Transformers%20is%20Easier%20Than%20You%20Think/exp3.png)

### Conclusion

把分类、检测领域的一些方法运用到生成领域中，往往就会有奇效，谢赛宁参与的这篇REPA就是一个很好的例子，以及笔者注意到近期另一位大佬何恺明对这篇论文也有了改进，直接将Dispersive Loss加到了Diffusion里，效果也是非常好，论文在此[Diffuse and Disperse](https://arxiv.org/abs/2506.09027)。