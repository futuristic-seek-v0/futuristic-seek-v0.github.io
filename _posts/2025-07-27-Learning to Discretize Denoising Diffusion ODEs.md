## (ICLR 2025 oral) Learning to Discretize Denoising Diffusion ODEs

![main](images/2025-07-27-Learning%20to%20Discretize%20Denoising%20Diffusion%20ODEs/main.png)

### Preface

Diffusion模型作为近几年生成模型的热点，继GAN之后又掀起了一波Generative Models的浪潮。DDPM模型是比较经典的Diffusion模型，生成效果有质的飞跃，但是DDPM模型的大问题就是生成过程计算量太大了，毕竟只能一步一步的去噪，相较于GAN的一步生成，效率实在是过低。为了降低生成的计算量，提高速度，后续诸如DDIM等知名工作都是用一系列改进提高推理的速度。这篇文章从ODE的视角出发，运用Teacher-Student的蒸馏机制，优化采样步长，从而达到提高推理速度的目的。

### Diffusion Model

第一次写到Diffusion Models，有必要简单介绍一下Diffusion是什么，具体的介绍网上已经非常详细了，这里笔者就记录一下自己觉得比较关键的点。

既然叫Diffusion（扩散）模型，说明这个模型肯定跟物理中的热力学有关系。Diffusion模型模拟的是将数据分布变换到高斯噪声分布（类比于最终平衡态）的过程，这与非平衡热力学中，系统从非平衡态趋于平衡态非常类似。就像把一滴墨水滴到水槽里。

具体来说，Diffusion模型通过模拟一个正向的 **加噪声 Markov 过程**（forward diffusion），以及一个神经网络驱动的 **逆过程**（reverse denoising），实现从纯噪声逐步生成逼真样本。

**正向过程**

对于一个数据分布$$x_0 \sim q(x)$$，每一步逐步进行加噪：
$$
q(x_t \mid x_{t-1}) = \mathcal{N}\bigl(x_t;\,\sqrt{1 - \beta_t}\,x_{t-1},\;\beta_t I\bigr),
$$
这个过程为什么叫做“加噪”呢？因为
$$
x_{t} = \sqrt{1 - \beta_t} x_{t-1} + \beta_t\epsilon,\,\,\,\, \epsilon\sim\mathcal{N}(0,I),
$$
可以证明，这个公式得到的$$x_t$$符合$$\mathcal{N}\bigl(\sqrt{1 - \beta_t}\,x_{t-1},\;\beta_t I\bigr)$$，也就是这两个公式是等价的，第二个公式很显然就是“加噪”的形式，而为什么是$$\sqrt{1 - \beta_t}$$和$$\beta_t$$呢，这是为了让方差$$Var(x_t)$$不至于随着时间爆炸；

基于上述从$$t-1$$到$$t$$的一步的过程，还可以写出从0一步走到$$t$$的一步加噪公式，具体的证明可以参考[这篇 Tutorial](https://arxiv.org/abs/2403.18103)：
$$
q(x_t \mid x_0) = \mathcal{N}\left( x_t;\; \sqrt{\bar\alpha_t}\,x_0,\; \sigma_t^2 I \right),
$$
其中
$$
\alpha_t = 1 - \beta_t\\
\bar\alpha_t = \prod_{i=1}^t \alpha_i\\
\sigma_t = \sqrt{1 - \bar\alpha_t},
$$
**逆向过程**

这部分就是训练一个预测噪声的网络，最初的DDPM采用的是U-Net架构，即$$\epsilon_\theta(x_t, t)$$，作用是预测出这一时刻加上去的噪声。训练方法是将当前时刻加的噪声和预测的噪声做MSE loss；最后推理的时候，就是从一个纯高斯噪声，一步一步预测噪声，减去噪声，最后得到生成的图像。

已知前向过程$$q(x_t \mid x_0) = \mathcal{N}\bigl(x_t;\,\sqrt{\bar \alpha_t}\,x_0,\;(1 - \bar \alpha_t)\,I\bigr)$$ ，其中$$\bar \alpha_t = \prod_{i=1}^t (1 - \beta_i),\quad \sigma_t^2 = 1 - \bar \alpha_t$$；

loss函数可以写作
$$
L_t = \mathbb{E}_{x_0,\,\epsilon}\Bigl\| \epsilon_\theta(x_t, t) - \epsilon \Bigr\|^2,
$$
根据前面所写的公式，可构造$$x_t = \sqrt{\bar{\alpha_t}}x_0 + \sigma_t\epsilon$$，最优解可写为： 
$$
\epsilon_\theta(x_t, t) = \frac{x_t - \sqrt{\bar\alpha_t}\,\mathbb{E}[x_0 \mid x_t]}{\sigma_t},
$$
使用 Tweedie公式（$$\mu$$有先验分布，$$x$$为噪声观测）：
$$
\mathbb{E}[\mu\mid x] = x+\sum{\grad_x\log{\rho(x)}}\\
\quad x = \mu + \epsilon,\quad \epsilon \sim \mathcal{N}(0, \Sigma),
$$
建立条件期望与 score（梯度 log-density）的关系： 
$$
\nabla_{x_t} \log q_t(x_t) = \frac{1}{\sigma_t^2} \left(\sqrt{\bar \alpha_t}\, \mathbb{E}[x_0 \mid x_t] - x_t\right)
$$


整理可得： 
$$
\nabla_{x_t} \log q_t(x_t) = -\frac{1}{\sigma_t}\,\epsilon_\theta(x_t, t),
$$
因此最终得出：

$$
\nabla_{x_t} \log q_t(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sigma_t}
$$




### Diffusion ODE

从这里开始，笔者将逐渐介绍这篇论文的内容。首先，我们从SDE角度看待Diffusion的动态过程（上一节介绍的是离散形式，改成连续形式就很容易理解SDE角度），一个SDE可以表示为：
$$
\frac{d\bold{x}(t)}{dt}=\bold{f}(\bold{x},t)+g(t)\bold{\xi}(t)
$$
第一项是确定项，第二项就是带来随机性的项，对于Diffusion来说，$$\bold{\xi}(t)\sim\mathcal{N}(0,I)$$，这里已经跟论文里的公式(3)形式很接近了，由于是加噪过程，那么随机项的$$g(t)$$不妨认为是稳定大于0，改写一下上面的公式就可以得到：
$$
\begin{aligned}\frac{\partial\mathbf{x}_t}{\partial t}&=f(t)\mathbf{x}_t+\frac{g^2(t)}{2\sigma_t}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t,t),\mathrm{~where~}f(t)=\frac{\partial\log\alpha_t}{\partial t},g^2(t)=\frac{\partial\sigma_t^2}{\partial t}-2f(t)\sigma_t^2.\end{aligned}
$$
具体的公式推导网上参考非常多，是从Probability Flow ODE推导而来。

总之我们需要知道的是，Diffusion就可以看作是一个ODE问题，它的解是$$\Psi(\bold{x}_T,\{t_i\}_{i=0}^{N},\bold{\epsilon_{\theta}})$$，代表最终的$$\bold{x_0}$$，也就是生成的图像，$$\{t_i\}_{i=0}^{N}$$表示求解的中间过程的时间步。

### Method

讲了这么多终于到了正题，这篇文章通过什么来提升速度和计算效率呢？这篇文章考虑的问题是将一个预训练好的、时间步长比较小的Diffusion模型作为Teacher Model，蒸馏给一个Student Model，由于计算过程主要来自去噪阶段的逐时间步采样计算，也就是$$\{t_i\}_{i=0}^{N}$$，为此，本论文将时间步长作为参数进行优化，文中标记为$$\bold{\xi}$$，可以理解为一个学习时间步长的神经网络，蒸馏的方法是最小化Teacher Model和Student Model在去噪过程中最终解的距离，用公式表示就是
$$
\min_{\bold{\xi}}\mathcal{L}_{hard}(\bold{\xi}) = \min\mathbb{E}_{\bold{x_T}\sim\mathcal{N}(\bold{0},\sigma_T^2\bold{I})}[\operatorname{d}(\Psi_{\bold{\xi}}(\bold{x_T}),\Psi_{\bold{*}}(\bold{x_T}))],
$$
其中$$\Psi_{\bold{\xi}}(.)$$表示Student Model，$$\Psi_{\bold{\xi}}(\bold{x_T})$$表示解，$$\Psi_{*}(.)$$表示Teacher Model，$$\operatorname{d}(\cdot,\cdot)$$表示距离，可以是欧式距离也可以是其它的距离，本论文采用的是LPIPS距离，一句话概括就是把图像输入CNN，把中间变量也纳入了距离计算，详情可参考[The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924)。

其次，论文提到，严格的让Student Model遵循Teacher Model，也就是强迫Student Model必须像Teacher Model一样，从完全一样的$$\bold{x_T}$$去噪得到完全一样的$$\bold{x_0}$$，训练效果并不好，因为会带来欠拟合的问题，为此这篇论文放开了限制，只要让Student Model从一个和$$\bold{x_T}$$比较接近的噪声$$\bold{x_T}'$$，生成$$\bold{x_0}$$即可，上述的loss就可以改成
$$
\min_{\bold{\xi}}\mathcal{L}_{soft}(\bold{\xi}) = \min\mathbb{E}_{\bold{x_T}\sim\mathcal{N}(\bold{0},\sigma_T^2\bold{I})}[\min_{\bold{x_T}'\in B(\bold{x_T},r\sigma_T)}\operatorname{d}(\Psi_{\bold{\xi}}(\bold{x_T}'),\Psi_{\bold{*}}(\bold{x_T}))],
$$
其中$$B(\bold{x_T},r\sigma_T)$$表示以$$\bold{x_T}$$为球心、$$r\sigma_T$$为半径的球，$$r$$为超参数。

作者同时通过一张示意图，说明这样的设计是有效的。

![r](images/2025-07-27-Learning%20to%20Discretize%20Denoising%20Diffusion%20ODEs/r.png)

整个的LD3算法框架如图：

![algorithm](images/2025-07-27-Learning%20to%20Discretize%20Denoising%20Diffusion%20ODEs/algorithm.png)

### Experiments

这篇论文的作者在CIFAR10、AFHQv2、FFHQ等数据集上进行FID score的计算，得到下列的结果，很明显LD3方法的结果更加优异：

![exp1](images/2025-07-27-Learning%20to%20Discretize%20Denoising%20Diffusion%20ODEs/exp1.png)

此外，这篇论文还将LD3和其它相似方法进行生成上的比较，生成效果看起来确实是LD3更好：

![exp2](images/2025-07-27-Learning%20to%20Discretize%20Denoising%20Diffusion%20ODEs/exp2.png)

### Conclusion

其实这篇笔记写完了还是有点一知半解，主要的原因是对Diffusion的ODE视角了解确实不是很深，这里挖个坑，以后专门写一篇Diffuison ODE视角的笔记，不过这种soft的形式和Teacher-Student Model在生成领域的应用还是比较少见和有借鉴价值的。
