# (NeurIPS 2020) Geometric Dataset Distances via Optimal Transport
![结构图](images/2025-07-15-Geometric%20Dataset%20Distances%20via%20Optimal%20Transport/structure.png)
## Summary（总结）
写完笔记之后最后填，概述文章的内容，以后查阅笔记的时候先看这一段。注：写文章summary切记需要通过自己的思考，用自己的语言描述。忌讳直接Ctrl + c原文。

## Research Objectives（研究目标）
这篇文章属于Domain Adaptation的范畴，作者的想法是用一种距离来衡量不同数据集之间的差距，即使这些数据集的分布域完全不一样。
## Background / Problem Statement（背景/问题陈述）
现有的机器学习方法都基于以数据-标签形式数据集的监督学习样式，而现实中的数据和标签非常稀缺。因此，跨数据集和领域之间的知识迁移和学习技术显得非常重要。这其中，两个数据集的距离是一个通用且重要的概念。不同数据集之间的特征距离还较为容易衡量，而标签集之间的距离衡量则非常困难，现有的方法也很少直面这一问题。

为此，本论文提出一种基于Optimal Transport距离的方法，相较于现有的方法，能有效衡量不同数据集之间的特征和标签之间的差异，同时这种方法不依赖具体模型，操作起来非常便捷。

## Method（方法）
### Optimal Transport距离
Optimal Transport Distance，也就是OT距离，顾名思义，起源于最优传输问题。可以理解为将货物从仓库运往商场，在给定运输成本的情况下，寻找到最优的运输方案，并由此得出最小的运输成本。


将这一思想运用到概率分布上来，对于两个概率分布p，q，定义好恰当的传输成本计算方式，就可以用OT距离衡量两个概率分布之间的差异（p和q的积分和都是1，就相当于将总质量为1的货物从仓库p搬运到商场q），相较于KL散度、JS散度等指标，**OT距离对概率分布的几何结构信息更加敏感，无论是离散还是连续情景都可以计算。**

![ot](images/2025-07-15-Geometric%20Dataset%20Distances%20via%20Optimal%20Transport/ot2.png)

在数学层面简单介绍一下OT距离的计算。设$\mathcal{X}$是一个度量空间，$\alpha(x), \beta(x)$是$\mathcal{X}$上的两个概率分布，$\Pi({\alpha, \beta})$是$\alpha, \beta$全体联合分布的集合，$\pi(x, y)$是其中的任一一个联合分布。那么这个联合分布就可以看作是一种从$\alpha$到$\beta$的运输方式（对x的边缘分布等于$\alpha$，对y的边缘分布等于$\beta$，且边缘分布的积分和都是1，符合运输方案的一切特性），再定义好运输成本函数$c(x, y)$，就可以计算运输成本，再结合一下泛函中距离定义的思想，将最优运输成本定义为最终的OT距离，就得到了下述的计算式。（离散情况就把积分改成求和）

$$
\mathrm{OT}(\alpha,\beta)\triangleq\min_{\pi\in\Pi(\alpha,\beta)}\int_{\mathcal{X}\times\mathcal{X}}c(x,y)\mathrm{d}\pi(x,y).
$$

### Optimal Transport Dataset Distance
了解了OT距离后，论文的方法就比较好理解了。
对于在$\mathcal{X}\times\mathcal{Y}$两个数据集$\mathcal{D}_A=\{(x_A^{(i)},y_A^{(i)})\}_{i=1}^n\sim P_A(x,y), \mathcal{D}_B=\{(x_B^{(i)},y_B^{(i)})\}_{i=1}^n\sim P_B(x,y)$，要计算数据集之间的距离，同样是分为数据和标签两部分，数据部分的计算依旧较为简单，在特征空间里用欧几里得距离就可以，记为
$d_{\mathcal{X}}(x,x')$。而标签空间$\mathcal{Y}$上的距离计算就比较有难度了，这篇论文中就在这采用了OT距离。

具体来说，对于两个数据集里的不同标签$y, y'$，提取出他们的分布$\alpha_y(x)=P(x|Y=y)$，然后对两个分布$\alpha_y$和$\alpha_{y'}$，定义标签空间上的OT距离为
$$
d_{\mathcal{Y}}(y, y') = W(\alpha_y,\alpha_{y'})=OT(\alpha_y,\alpha_{y'})，
$$
最后，引入一个p幂次计算，得到最终的计算公式
$$
d_{OT}((x, y),(x',y'))=(d_{\mathcal{X}}(x,x')^p+W_p^p(\alpha_y,\alpha_{y'}))^{\frac{1}{p}},
$$
其中，$W_p(P, Q) = \left( \inf_{\gamma \in \Gamma(P, Q)} \int_{X \times Y} d(x, y)^p \, \mathrm{d}\gamma(x, y) \right)^{1/p}$，相较于普通的OT距离更加灵活。
## Evaluation（评估方法）
### MNIST相关数据集
作者在EMNIST、MNIST等数据集上进行了实验，用LeNet-5作为模型，得到数据集之间的距离如下图：
![mnist](images/2025-07-15-Geometric%20Dataset%20Distances%20via%20Optimal%20Transport/mnist.png)
随后是一些鲁棒性等指标的实验：
![robust](images/2025-07-15-Geometric%20Dataset%20Distances%20via%20Optimal%20Transport/mnist-robust.png)
![robust](images/2025-07-15-Geometric%20Dataset%20Distances%20via%20Optimal%20Transport/robust2.png)
## Conclusion（结论）
该方法相较于其它方法确实有所提升，因为笔者目前没有深挖这个领域，就不多评论了。

## Notes（笔记）
笔者看这篇论文主要是学习OT距离，看到OT距离还是很有使用价值的，这种从几何形状上衡量分布差异的距离在其它很多地方都有应用前景，是KL散度很好的替代。
<!-- ## References(optional) 
列出相关性高的文献，以便之后可以继续track下去。 -->
