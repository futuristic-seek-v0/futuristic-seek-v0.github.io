# (ICCV 2021) [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
![结构图](images/2025-07-17-DINOv2%20Learning%20Robust%20Visual%20Features%20without%20Supervision/main.png)
# Preface （前言）
DINOv2 是 Meta AI 在自监督学习领域继 DINO 之后推出的又一力作。它在 DINO 的基础上进行了多项改进，在图像和视频的自监督学习方面取得了显著的进展。本笔记将重点聚焦 DINOv2 与其前身 DINO 的主要不同之处。

# Key Idea（核心理念）
关于 DINO 的解读网上已经有很多了，笔者也写了一篇供参考。简言之，DINO 利用自蒸馏来训练一个 ViT 模型。它通过让一个“学生”网络去匹配一个“教师”网络的输出，从而学习到丰富的视觉特征。教师网络是学生网络过去权重的一个指数移动平均 (EMA)，这种机制避免了模型坍塌问题。详情可参见[这篇blog](https://zhuanlan.zhihu.com/p/1928953077152592104)。

# DINOv2 vs DINO
## Data Scaling（更大规模的数据集）
最能决定一个模型能力的往往是训练数据的数量和质量，DINOv2 最显著的改进之一是其训练数据规模的显著扩大。DINO 主要是在 ImageNet 上进行训练，而 DINOv2 则利用了一个包含数亿张图像的内部策划数据集，也就是LVD-142M，这个数据集是通过大规模的网络抓取和过滤得到的，可以提供更丰富、更多样化的视觉信息。该数据集的收集和处理过程也是一个比较庞大的工程，可参见原文的Data Processing部分，这里就不详细展开了。

## Improved Training Stability（改进训练稳定性）
数据集的规模得到了扩充，相应的训练过程也需要一些改进来提升稳定性。这里就按照论文的架构进行记录。
### Image-level objective（图像级目标）
这部分和DINO的核心算法基本相同，裁剪过的图像作为局部特征喂给student，全局图像喂给teacher，然后用交叉熵和EMA进行自蒸馏。

### Patch-level objective（Patch级目标）
ViT的机制是将图像进行分割，变成一系列的patch作为image token然后输入到ViT里进行后续的计算，因此这些patch也是能动动手脚，给模型上上强度的。具体来说，给student喂的数据会随机mask掉一些patch，给teacher喂的数据则没有mask。针对这些mask掉的patch，将它们的预测结果依次计算一个交叉熵loss（论文里叫做iBOT）。

### Untying head weights between both objectives（解耦目标间head权重）

Image-level objective和Patch-level objective的损失函数使用了同一套网络权重。但有实验证实，这样会使模型在Patch级别会欠拟合，在Image级别会过拟合。因此，这篇论文选择将这两套权重解绑，使得模型在两个级别都能够更好地学习特征表示。这个方法的优化目标是在两个级别都得到最佳的结果（也就是用两套参数，一套用patch训，另一套用image训）。

### Sinkhorn-Knopp centering
注意到DINO的架构中，teacher端是有center操作的，这部分就是对centering操作的改进。采用Sinkhorn-Knopp批量归一化方法进行代替，将非负的教师 softmax 输出固定为“批量双随机分布”（rows & cols sum to uniform），有效防止单一原型被过度利用；相比之下，DINO 原来的 centering 是一种对 logits 的 moving‑average 调整，虽然简便，但并不强制“行列均衡”。
这样做能在 batch 级别更有效地平衡teacher的输出分布、减少 collapse，并提升多样性和稳定性。

![distillation](images/2025-07-17-DINOv2%20Learning%20Robust%20Visual%20Features%20without%20Supervision/distillation.png)


### KoLeo regularizer（KoLeo 正则化）
KoLeo（Kozachenko–Leonenko）regularizer 是一种差分熵正则化，其核心目的是在batch级别将特征嵌入“均匀展开”，防止特征聚集或 collapse，提高表征多样性。其公式为
$$\
\mathcal{L}_{\mathrm{KoLeo}}
= -\frac{1}{B} \sum_{i=1}^{B} \log \Bigl( \min_{j \neq i} \|f_i - f_j\| \Bigr)
\
$$

### Teacher-Student Network
Teacher-Student架构是一种经典的蒸馏学习的架构，Student模型通过模仿Teacher模型的预测结果来进行学习。下图便是DINO中使用的Teacher-Student结构，也是DINO的核心所在。

一般情况下，Teacher是一个能力较强的训练好的模型，比如在ImageNet上训练好的ViT-L，而Student是一个能力较弱的模型，比如ViT-S，对于相同的输入，将Teacher和Student的输出（比如说是图像分类预测）进行交叉熵计算，然后将loss用于Student模型的参数更新，从而让Student的输出向Teacher靠拢。
![distillation](images/2025-07-16-Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers/distillation.png)

## Efficient Implementation（高效实施方案）
笔者关于 GPU 这些硬件的知识还是一片空白，挖个坑等以后学好了再来填吧。

## Experiments（实验）
当然，上述所说的情况指的就是普通的Distillation，DINO所使用的自蒸馏则稍有不同。

首先，在DINO中，Student和Teacher的架构完全相同，并不是一大一小、一强一弱（毕竟是自蒸馏，顾名思义就是一个模型自己蒸馏自己），两个模型均采用ViT架构，初始时参数也完全一致。

在训练过程中，喂给Teacher和Student的数据，以及二者参数更新机制均有所不同。对于一个输入x，DINO首先对x进行Augmentation，产生2个全局的输入数据和一些低分辨率的局部输入数据，全局输入数据喂给Teacher，而局部输入数据喂给Student。这样做能让Student用局部 view 去预测全局 view 的特征，训练引导Student提取语义一致、全局协调的表示，以此提升模型对局部信息的学习能力（通俗来说就是让模型更加细腻），从而提升模型的总体能力。

在参数更新上，Teacher和Student也有不同。对于Student，即采用前文提到的Teacher output和Student output的交叉熵损失进行参数更新，而Teacher的参数则通过EMA（指数滑动平均）进行更新，即
$$
\theta_{teacher} = m\theta_{teacher}+(1-m)\theta_{student},
$$
这样操作，能让Teacher变化更为缓慢，成为稳定版的Student。

关于EMA这部分，笔者的理解是，这样做让Teacher既能保存对全局信息的提取能力，同时也能享受到Student带来的局部信息提取能力提升，从而让Teacher始终比Student厉害，形成Student不停追赶Teacher的情景。

同时，注意到图中，teacher端有一个centering操作，这个操作是对Teacher输出的 logits（未经 softmax 的分数）减去一个动态维护的中心向量**c**，然后再进行softmax操作，即
$$
\bold{output}= \operatorname{softmax}(\bold{logits}-\bold{c}),
$$
这样做是为了防止某些 logit 维度长期偏高，简言之，centering操作让输出更“平滑”、更均匀，从而阻止Teacher身陷“每个输入都输出同一个均匀分布”的崩塌。

写到这里，DINO的核心算法就都讲完了，笔者主要是聚焦于Self-Distillation，毕竟其它部分都比较好理解，具体的技术细节可以参考原文。

## Experiments（实验）
![experiments](images/2025-07-17-DINOv2%20Learning%20Robust%20Visual%20Features%20without%20Supervision/experiments1.png)

![experiments2](images/2025-07-17-DINOv2%20Learning%20Robust%20Visual%20Features%20without%20Supervision/experiments2.png)
## Conclusion（结论）
其实在读过DINO之后，对DINOv2有过期待，好奇会有什么创新点，读完后发现这些改进大多集中在正则化和训练方法上，以及最重要的数据集规模，在科研上能挖掘的点并不是很多，但可能是笔者的工程能力太过欠缺，没能体会到这些技术的美，还是要继续努力学习呀！

笔者水平有限，心里也知道这篇写的很垃圾，估计对读者大佬们也没什么帮助，而且写的时候已经是深夜了，以后会提升认知水平，写的更加精彩的！
