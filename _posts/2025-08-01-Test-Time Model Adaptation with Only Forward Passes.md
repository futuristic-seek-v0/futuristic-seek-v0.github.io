# (ICML 2024 oral) Test-Time Model Adaptation with Only Forward Passes

 ![main](images/2025-08-01-Test-Time Model%20Adaptation%20with%20Only%20Forward%20Passes/main.png)

### Preface

Test-Time Adaptation、Domain Adaptation、Domain Generalization等等一系列Transfer Learning的方法都需要根据变换的domain实时更新模型参数，或是整个模型参数，或是BatchNorm层，但是这篇ICML 2024的oral文章，巧妙地结合了CMA和Prompt Learning，在不改变模型参数的前提下，实现了Test-Time Adaptation，相比众多需要梯度下降更新参数的方法来说在计算量上有巨大提升，同时TTA的表现也有提升。

### Test-Time Adaptation

Test-Time Adaptation，简称TTA，是Transfer Learning的一个研究分支，在这个setting下，无法获取训练源数据集，无法获取测试数据的标签，只有在源数据集上训练好的预训练模型，同时还需要面对持续的Domain Shift问题。

最基本的方法是运用Entropy Loss，对于一个分类预测概率$$Y$$，Entropy的计算公式为：
$$
H(Y) = -\sum_{i=1}^{n} P(y_i) \log P(y_i)
$$
这个loss越大，说明当前的预测结果更加不确定，反之，loss越小（举个例子，$$Y=[1/n,1/n,\cdots,1/n]$$时值最大，$$Y=[1, 0, 0,\cdots,0]$$时值最小），预测结果越确定，这便是一个自监督的loss函数。

在每一次的Forward求出Prediction之后，计算当前Prediction的Entropy并作为损失函数，进行一次loss.backward()，每次test就会进行adaptation，所以叫做Test-Time Adaptation。详情可以参考下面的表，展示了TTA与其它setting的区别。

 ![tta](images/2025-08-01-Test-Time%20Model%20Adaptation%20with%20Only%20Forward%20Passes/tta.png)



### Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

TTA虽然只在测试时进行适应，但是适应依旧是采用Gradient Descent的方式更新模型的参数，这个计算量肯定也不容小觑，同时，TTA一般情况下都是持续进行的，模型参数会一直进行偏移，因此更新策略不当很容易造成Catastrophic Forgetting，用中文来说就是学习了新的Domain，反而把Source Domain的特性遗忘了。一直以来的TTA方法都是在更新策略上进行创新改进，但都没有跳出Gradient Descent的范畴，而这篇论文则创新性结合了Covariance Matrix Adaptation Evolution Strategy和Prompt Learning，在不更新模型的情况下完成了Adaptation，这就解决了Catastrophic Forgetting的问题。

接下来，笔者将介绍一下Covariance Matrix Adaptation Evolution Strategy以及其中涉及的一些重要算法和公式，当然，还是尽量用通俗易懂的语言记录，详情可参考[Tutorial]([1604.00772](https://arxiv.org/pdf/1604.00772))和互联网上的其它笔记。

CMA-ES想要解决的问题和神经网络模型非常相似，都是一种类似黑盒模型的情景。简单来说，有一个黑盒函数$$f$$，如何寻找$$f(x)$$的最大值（最小值）。最简单的思路是暴力搜索，把定义域内的每一个$$f(x)$$都算一遍，一定能穷举出来最后的答案，但是这种方法显然计算开销太大，不切实际。因此，CMA-ES的方法是，维护一个正态分布，通过反复迭代调整这个正态分布，让该分布逐渐往最大值对应的定义域靠拢，最终搜索出答案，大概的过程如下图所示。

![cma](images/2025-08-01-Test-Time%20Model%20Adaptation%20with%20Only%20Forward%20Passes/cma.png)

下面从数学公式的角度再记录一遍，设$$t$$时刻的分布均值为$$\bold{m}^t$$，协方差矩阵为$$\bold{C}^t$$，步长为$$\tau^t$$，然后通过正态分布进行采样：
$$
\bold{x}_k^{t+1}=\bold{m}^t+\tau^t\mathcal{N}(0,\bold{C}^t),\quad k=(1,\cdots,K),\tag{1}
$$
对这$$K$$个样本，分别计算函数值$$f(\bold{x}_k^{t+1})$$，取前$$\lambda$$个最大（最小）的值，用符号表示为$$\bold{x}_{i:\lambda}^{t+1}$$，表示经过排序后的第$$i$$个最佳样本，用它们求加权平均，权重值的计算则用位置偏好权重，即
$$
w_i^{\rm raw} = \ln\bigl(\lambda + \tfrac12\bigr) - \ln(i),\quad i=1,2,\dots,\lambda,\\

w_i = \frac{w_i^{\rm raw}}{\sum_{j=1}^\lambda w_j^{\rm raw}},\tag{2}
$$
这个权重计算方式天然使得排名靠前的样本权重值更大，同时样本间不会有过大的权重差异，凸显出排序优化，通过这个权重$$\bold{x}_{i:\lambda}^{t+1}$$加权计算，得到下一时刻的分布均值$$\bold{m}^{t+1}$$。



除了均值，这个算法最关键的自然是更新协方差矩阵$$\bold{C}^t$$，先取排序后的$$\lambda$$个样本计算标准化步向量，也就是排除掉步长的影响：
$$
\boldsymbol{y}_i^{t+1}
\;=\;
\frac{
  \boldsymbol{x}_{i:\lambda}^{\,t+1}
  - \boldsymbol{m}^t
}{\tau^t},
\quad
i = 1, \dots, \lambda; \tag{3}
$$
再引入一个参数，即有效选择质量
$$
\mu_{\text{eff}}=\frac{1}{\sum_{i=1}^\lambda w_i^2}\tag{4},
$$
引入一条累积路径$$\bold{p}_c^t$$，用来捕获历代均值变化方向的一致性信号：
$$
\boldsymbol{p}_c^{\,t+1}
=
(1 - c_c)\,\boldsymbol{p}_c^t
\;+\;
\sqrt{c_c (2 - c_c)\,\mu_\text{eff}}
\sum_{i=1}^\lambda w_i\,\boldsymbol{y}_i^{t+1},\tag{5}
$$
对$$\boldsymbol{y}_i^{t+1}$$的求和就代表了更新的方向，再附加一个移动平均，得到新一时刻的路径点，笔者个人认为，可以粗略地理解为均值变化的路径。

最终的矩阵更新为：
$$
\begin{aligned}
\bold{C}^{\,t+1}
&=
(1 - c_1 - c_\mu)\,\bold{C}^t
+ c_1\,\bold{p}_c^{\,t+1}\bold{p}_c^{\,t+1\,T}\\
&\quad
+\,c_\mu \sum_{i=1}^\lambda w_i\,
\bold{y}_i^{t+1}\bold{y}_i^{t+1\,T}
\end{aligned}\tag{6},
$$
第一项是对旧协方差的衰减，第二项是提升方向稳定性的协方差，第三项是利用前$$\lambda$$个样本的协方差信息进行计算（注意这里$$y$$是减过均值的，这么算就是在计算协方差），两个参数通常设置为
$$
c_1 \approx \frac{2}{(n + 1.3)^2 + \mu_\text{eff}},
\quad
c_\mu \approx \min\left(
 1 - c_1,\; \frac{2(\mu_\text{eff} - 2 + 1/\mu_\text{eff})}{(n + 2)^2 + \mu_\text{eff}}
\right),\tag{7}
$$
这就是完整的CMA-ES更新算法。

### Method

讲完了CMA-ES，接下来就回到这篇论文。这篇论文的思路很简单，我们需要明确，CMA-ES算法迭代搜索的是最优的prompt参数，黑盒函数则是模型的损失函数。

![structure](images/2025-08-01-Test-Time%20Model%20Adaptation%20with%20Only%20Forward%20Passes/structure.png)

用$$\bold{p}$$表示prompt的参数，$$f_{\Theta}$$表示模型，$$\mathcal{L}$$表示损失函数，则要优化的问题为：
$$
\bold{p}^*=\arg\min_{\bold{p}}\mathcal{L}(f_{\Theta}(\bold{p},\bold{x})),\tag{8}
$$
采样的策略为
$$
\bold{p}_k^{(t)}=\bold{m}^{(t)}+\tau^{(t)}\mathcal{N}(0,\bold{\Sigma}^{(t)}),\quad k=(1,\cdots,K),\tag{9}
$$


当然，针对ViT模型的架构，这里所采用的loss则有了些变化：
$$
\mathcal{L}(f_{\Theta}(\bold{p},\bold{\mathcal{X}_t}))=\sum_{\bold{x}\in\mathcal{X}_t}\sum_{c\in\mathcal{C}}-\hat{y}_c\log\hat{y}_c\\ + \lambda\sum_{i=1}^{N}||\bold{\mu}_i(\mathcal{X}_t)-\bold{\mu}_i^S||_2+||\bold{\sigma}_i(\mathcal{X}_t)-\bold{\sigma}_i^S||_2,\tag{10}
$$
其中$$\mathcal{X}$$表示 Batch Samples，$$\bold{\mu}_i(\mathcal{X}_t),\bold{\sigma}_i(\mathcal{X}_t)$$表示的是在当前Batch下，计算得出的第$$i$$层CLS feature的均值和方差，$$\bold{\mu}_i^S,\bold{\sigma}_i^S$$模型在Source Dataset上得到的第$$i$$层的CLS feature的均值和方差，这个设计也是进一步防止前文提到的Catastrophic Forgetting。



### Experiments

首先是TTA领域最经典的ImageNet-C的测试，本论文的方法FOA不但结果强过一众TTA方法，而且不需要梯度下降，可以说是非常的惊艳。

![exp1](images/2025-08-01-Test-Time%20Model%20Adaptation%20with%20Only%20Forward%20Passes/exp1.png)

然后是ImageNet-R/V2/Sketch，依旧是结果和计算量双杀。

![exp2](images/2025-08-01-Test-Time%20Model%20Adaptation%20with%20Only%20Forward%20Passes/exp2.png)

最后是一些分析实验和Ablation Study的内容。

![exp3](images/2025-08-01-Test-Time%20Model%20Adaptation%20with%20Only%20Forward%20Passes/exp3.png)



### Conclusion

又解锁了一个强力training-free方法CMA-ES，以及不用梯度下降的方法总是这么的惊艳，有一年前看到Tip-Adapter的不可思议的感觉，不禁感慨大道至简。