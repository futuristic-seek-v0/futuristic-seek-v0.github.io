# (ICCV 2021) [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
![结构图](images/2025-07-16-Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers/main.png)
## Preface （前言）
这篇是提出DINO模型结构的论文，来自Facebook AI，笔者站在2025年的时间学习记录这篇ICCV 2021的论文，主要是为之后学习DINOv2进行铺垫，熟悉一下ViT模型的发展过程。

## Research Objectives（研究目标）
这篇文章里明确提过，想探究Vision Transformer为什么会有比CNN更加优秀的性能，以及自监督学习是否能帮助Vision Transformer提升。此外，收到Transformer在NLP领域里Self-Supervied Learning（自监督学习）的启发，将Vision Transformer与Self-Supervised Learning结合起来，再结合上Knowledge Distillation（知识蒸馏），得到最终的DINO（self-**di**stillation with **no** labels）模型，并在多个下游任务上进行了实验验证。

## Method（方法）
### Self-Supervised Learning（自监督学习）
自监督学习相信大家都比较熟悉，简单来说就是不需要标签，模型仅从数据本身学习语义特征，不需要人工标注。例如K-Means聚类算法，只需要数据点，不需要标注，就能通过迭代实现准确分类。详情可阅读[这篇文章](https://www.cnblogs.com/pinard/p/6164214.html)。

![kmeans](images/2025-07-16-Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers/kmeans.png)

那么，在DINO中，自监督学习是如何体现的呢？答案是Self-Distillation（自融合蒸馏），使用Teacher-Student网络架构。

### Teacher-Student Network
Teacher-Student架构是一种经典的蒸馏学习的架构，Student模型通过模仿Teacher模型的预测结果来进行学习。下图便是DINO中使用的Teacher-Student结构，也是DINO的核心所在。

一般情况下，Teacher是一个能力较强的训练好的模型，比如在ImageNet上训练好的ViT-L，而Student是一个能力较弱的模型，比如ViT-S，对于相同的输入，将Teacher和Student的输出（比如说是图像分类预测）进行交叉熵计算，然后将loss用于Student模型的参数更新，从而让Student的输出向Teacher靠拢。
![distillation](images/2025-07-16-Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers/distillation.png)


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
### ImageNet
这篇论文将DINO与多种方法，在多种架构上进行了比较，验证了DINO采用Self-Supervised Learning的性能。
![imagenet](images/2025-07-16-Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers/imagenet.png)
### Image Retrival
这个表是DINO在Image Retrieval上的实验结果。
![image retrieval](images/2025-07-16-Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers/image-retrieval.png)

### Copy Detection
笔者不知道这是什么实验，总之也记录一下吧。
![copy detection](images/2025-07-16-Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers/copy-detection.png)

### Segmentation
这部分是笔者最感兴趣的实验，因为第一次接触到DINO就是听大佬讲到DINO可以无监督segmentation，而且质量非常惊艳，可以说是有效展示了ViT架构的强大实力。
![segmentation](images/2025-07-16-Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers/segmentation.png)
## Conclusion（结论）
笔者读过这篇论文后，其实最大的收获是借机学明白了Teacher-Student架构的Self-Distillation方法，同时了解到DINO的self-attention map和无监督segmentation能力很强，可以作为一个强力的工具用于如今这个大模型遍地走的时代，从而能发散思维，在这个基础上做一些更有意思的事情。

笔者水平有限，文章也没写多长，只是基于自己的学习收获草草记录了一下，如果有什么错误，还请各位大佬见谅！
