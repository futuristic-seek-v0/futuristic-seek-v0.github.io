## 论文笔记（2025.07.19）(ICLR 2025 oral) Differential Transformer+Transformer回顾与解析

![main](images/2025-07-19-Differential%20Transformer/main.png)

### 前言

自从2017年《Attention is All You Need》的发布，Transformer架构横空出世，并主导了人工智能的大发展，目前的各类模型的基础架构都是基于 Transformer 进行搭建。Transformer最初是在NLP领域大放异彩，随着Vision Transformer的提出，Transformer架构也在视觉领域发力，逐渐追赶甚至超越了卷积神经网络。毫无疑问，Transformer就是当下最红火的架构。

但是，Transformer架构也有它的局限性，比如这篇由微软研究院和清华大学发表的论文《Differential Transformer》，就指出Transformer架构“注意力不够集中”的问题，并借助电子信息领域的差分去噪方式改进了Transformer结构，这篇论文也在ICLR 2025上作为oral文章发表。

### Transformer 回顾

在介绍这篇论文前，鉴于Transformer的地位，有必要再来回顾一下Transformer架构，再加深一下理解。当然，网上写Transformer的blog多到数不清，笔者还是按自己的理解来记录一下。

#### Attention机制

Attention机制，中文里叫做注意力机制，是Transformer的核心。用一道简单的完型填空就可以形象地介绍这一机制：“马路上有很多（）”。这个括号里的词大概率是“汽车”，作为人类，我们是如何做出这个判断呢，便是基于这句话中“马路”这个词，也就是把注意力放到了这个词上，而对于“有很多”这样的字词关注程度则没有那么高。Transformer正是通过合理的设计实现了这一机制。



从数学的角度上来说，对于一个序列$\bold{X} =[x_1,x_2,\cdots,x_n]^T$，有三个神经网络，对于$x_i$可以提取出$q_i,k_i,v_i$，分别对应query、key和value，其中query和key用来衡量注意力分配的权重，value则可以理解为当前token的一个语义信息特征，可以类比word embedding向量。

> [!NOTE]
>
> 为了方便，默认这里的$x_i$都已经是embedding形式了，可以理解为一系列的 word embedding 向量。

对于一个查询向量$q'$，通过$dot(q',k_i)$来计算当前查询向量对$[x_1,x_2,\cdots,x_n]$的关注程度，即
$$
a_i = \frac{\exp(dot(q',k_i)/\tau)}{\sum_{j=1}^{n}\exp(dot(q',k_j)/\tau)},
$$
$a_i$则表示对$x_i$的关注程度，再将关注程度作为注意力的权重对value进行加权平均，得到当前查询向量的attention输出：
$$
\operatorname{Attn}(q',\bold{x})=\sum_{i=1}^{n}a_iv_i
$$
这样的Attention机制完美实现了对注意力的分配，只要q，k，v设计的合理，完全可以做到让模型将注意力投放到“马路”上，从而做出正确的预测。

那么，q，k，v该怎么计算呢？当然是用learnable的矩阵了！对于维度为d的一个token embedding，比如$x_i$，分别定义3个矩阵$\bold{W}_Q,\bold{W}_K,\bold{W}_V$，形状为$d\times n$，则
$$
q_i=x_i^T\bold{W}_Q\\
k_i=x_i^T\bold{W}_K\\
v_i=x_i^T\bold{W}_V,
$$
然后可以得到三个矩阵$\bold{Q}=\bold{X}\bold{W}_Q=[q_1,q_2,\cdots,q_n],\bold{K}=\bold{X}\bold{W}_K=[k_1,k_2,\cdots,k_n],\bold{V}=\bold{X}\bold{W}_V=[v_1,v_2,\cdots,v_n]$。

那么，$\bold{Q}\bold{K}^T$就可以看成是这n个token互相进行查询，再乘上$\bold{V}$，就得到了最终的输出：
$$
\operatorname{Attn}(\bold{Q},\bold{K},\bold{V})=softmax(\frac{\bold{Q}^T\bold{K}}{\sqrt{d}})\bold{V}
$$
这就构成了一个Attention层，**<font color='red'>不同于普通的神经网络层，这一结构的每一个输出都充分整合了输入的token之间的关系，充分利用了相似度，并以此得到最后的输出特征。</font>**





### Multi-Head Attention

了解了Attention机制后，需要再介绍Multi-Head Attention机制。如下图右所展示。

![attention](images/2025-07-19-Differential%20Transformer/attention.png)

核心思想和Attention机制是一模一样的，不一样的是，会把Q、K、V三个矩阵分成多个部分，文章里也叫做多个head，每个head里独立进行Attention的计算，最后再组合起来，通过线性层得到最后的输出。



### Transformer架构——Training

利用Multi-Head Attention机制，引出Transformer的训练架构图：

![transformer](images/2025-07-19-Differential%20Transformer/transformer.png)

左侧是Encoder，输入经过多层Multi-Head Attention层提取出特征，其中的Add&Norm是一种类似ResNet的设计，防止梯度消失，Positional Encoding是为了给输入的序列添加位置信息，让模型明白输入序列的顺序信息。

右侧则是Decoder，其中的Masked Multi-Head Attention的目的是只让Decoder的输入读取到之前已经生成的内容，因为做Next Token Prediction的时候，只能获取到前文信息。

> [!NOTE]
>
> 在训练中，我们会把output完全输入进去并行计算，比如是将'I love you'翻译成'我爱你'，训练时'我爱你'这几个token会同时喂给模型，如果不mask，decoder会直接读取到它后续要生成的内容，所以为了防止模型作弊，需要mask。实际的代码里，Masked Multi-Head Attention是利用下三角矩阵进行实现。

整个的流程如下，比如说现在要翻译一句话，Inputs是“我是一只猫”，decoder的Outputs会先输入一个<BOS>（表示开始）的特殊token，它经过Masked Multi-Head Attention后，将输出作为query，然后将Encoder传过来的特征作为key和value，输出通过query计算attention分数，然后相应地提取value信息，然后后续再经过Multi-Head Attention进行操作，得到对下一时刻输出的预测概率，让输出和ground truth匹配，以此为目标进行训练，比如让模型在这里输出'I'的概率最大化，随后把ground truth token，也就是'I'，添加到下一时刻decoder的outputs输入里去，进行后续的训练。需要再次强调的是，实际中这个过程是并行完成的。

在Inference阶段，则是把上一时刻的输出添加到下一时刻decoder的outputs输入，而且是串行操作。这个过程用言语实在是不太好表达，更多可视化的内容，详情可参考[这篇blog](https://zhuanlan.zhihu.com/p/338817680)。（https://zhuanlan.zhihu.com/p/338817680）



## Differential Transformer

### Method

讲了这么多，终于进入正题了。这篇论文首先通过实验，发现Transformer架构的一个问题，就是对于不重要内容的attention程度还是太高了，比如之前提到的“马路上有很多（）”，模型把attention过多的分散到了“有很多”上（私下感觉这个例子不太妥当，但是想表达的意思还是很明确的），为此作者做了实验证实了这一现象。

![noise](images/2025-07-19-Differential%20Transformer/noise.png)

这些不必要的attention被称之为attention noise，如何消除attention noise能让模型性能更优异呢？作者借用了电路中差分去噪的思路，简单来说就是把attention一分为二，将差作为最后的attention。

笔者这里偷懒直接搬论文截图了（），一看就能懂：

![method](images/2025-07-19-Differential%20Transformer/method.png)

这个$\lambda$参数为什么这么设置，笔者水平有限看不太懂，不过不影响总体的理解。

![structure](images/2025-07-19-Differential%20Transformer/structure.png)

结构非常好理解，但是确实管用，相减对attention noise的抑制效果会强于对有用信息的attention score的影响，归一化后就能起到很好的去噪效果。



### Experiments

这篇论文的实验主要是四个部分，一是下游任务的比较，二是测试Differential Transformer的长序列信息理解能力，三是具体的一些评估指标，四是说明Differential Transformer能有效降低activation中的异常值。

![exp1](images/2025-07-19-Differential%20Transformer/exp1.png)

![exp2](images/2025-07-19-Differential%20Transformer/exp2.png)

![exp3](images/2025-07-19-Differential%20Transformer/exp3.png)

![exp4](images/2025-07-19-Differential%20Transformer/exp4.png)

![exp5](images/2025-07-19-Differential%20Transformer/exp5.png)



## Conclusion

这篇ICLR 2025 oral的文章充分诠释了大道至简，如此简单的修改能带来性能的显著提升，让笔者感触最深的是，工程上的许多方法和思路，在人工智能领域可能都有大放异彩的潜力，扩宽知识面也是很重要的！！！
