## (ICLR 2024 oral) Vision Transformers Need Registers

![main](images/2025-07-25-Vision%20Transformers%20Need%20Registers/main.png)

### Preface（前言）

这篇来自Meta和阿尔卑斯大学的论文聚焦于一个非常比较有意思的领域，在Transformer架构如火如荼的时候，他们深挖架构本身，对ViT模型进行了结构性的研究探索，发现了ViT训练过程中存在异常token会显著影响模型的表现，从而进一步影响下游任务和特征图的解释性，同时提出了Register Token的解决方案，是一篇标准的“发现问题并提出解决方案的”高水平论文，不愧是Meta团队的产出。

话不多说，马上介绍这篇文章的核心内容。

### Problem Formulation

这篇论文从一个比较反常理的现象切入，即LOST算法和DINO结合做Object Discovery的效果非常好，毕竟DINO模型可以无监督的产生质量较高的attention maps，对下游任务肯定是有很大帮助的。

可是，DINOv2可以理解为DINO的pro max版，提取出的attention maps按理来说效果应该更好，结果DINOv2和POST算法的结合效果反而不如DINO了。这一现象说明，DINOv2和DINO的运行机制出现了偏差，DINOv2的attention maps具有一些特性是DINO没有的，这些特性导致了最终的结果偏差。

论文以这个切入点深入挖掘，并提出了他们的发现：在ViT中的一部分token（大约是2%）具有高于正常值10倍的范数，同时这些token经常出现在ViT的中间层，而且是在充分训练过后才会出现，尤其重要的是，这些异常token出现在与邻近patch相似度高的patch中，也就是出现在信息较为冗余的patch中。

论文里给出了一张图，非常直观的展示了这个现象。

![norm](images/2025-07-25-Vision%20Transformers%20Need%20Registers/norm.png)

对于这张狗而言，DINO所给出的local feature norms质量非常好，在狗出现的部分范数高，而在空白部分则较低，可以很清晰的看出给予attention的区域；而DINOv2反而做不到这么清晰，甚至观察到，在白色的空白部分，DINOv2的local feature norms出现了零星的异常patch，范数非常高。

是不是只有DINOv2这种自监督训练的Vision Transformer才有呢？并不是，这种现象非常普遍，无论是监督还是自监督，都免不了出现异常。

![norm2](images/2025-07-25-Vision%20Transformers%20Need%20Registers/norm2.png)

此外，论文做了一系列实验，研究了这一现象发生的时机以及这一现象出现的区域特性：

![time](images/2025-07-25-Vision%20Transformers%20Need%20Registers/time.png)

![region](images/2025-07-25-Vision%20Transformers%20Need%20Registers/region.png)

这些实验验证了前文记录的发现，即异常token多出现在中间层，多出现在训练时间1/3之后，多出现在近邻相似度较高的patch中。

除了探究异常token出现的时机、位置，这篇论文也分享了一个更加有趣的现象：这些异常的高范数tokens，包含更多的全局信息而不是局部信息，论文通过将这些token和普通token用于image classification来验证这一发现：

![classification](images/2025-07-25-Vision%20Transformers%20Need%20Registers/classification.png)

结果显示，异常token的分类效果显著好于一般的token。

一句话概括上述的异常现象：**大型、经过充分训练的模型学会识别冗余token，并将它们用作存储、处理和检索全局信息的地方**。

### Method

发现了现象，也就给出了解决问题的可能方案。这篇论文就提出了一个比较有意思的idea，既然Vision Transformer会把冗余token用于存储全局信息，那么可以专门引入新的token用来存储全局信息，类似[CLS] token一样，设计一个[reg] token，最后再把这个token抛弃掉就行了。

![method](images/2025-07-25-Vision%20Transformers%20Need%20Registers/method.png)

### Experiments

这么一弄效果可以说立竿见影。

![exp1](images/2025-07-25-Vision%20Transformers%20Need%20Registers/exp1.png)

加上这些[reg] tokens后，高范数的异常token数量立刻就降下去了。

![exp2](images/2025-07-25-Vision%20Transformers%20Need%20Registers/exp2.png)

可以看到，加入[reg] tokens后，在背景部分的异常patch也快速消失。

![exp3](images/2025-07-25-Vision%20Transformers%20Need%20Registers/exp3.png)

提取出[reg]里的token，也可以发现，[reg]里存储的确实都是些图片里无关紧要的区域，真正对分类有价值的信息都在[CLS]里，但有意思的是，它们却能自动聚焦到一定区域上，比如说图中的这把勺子。



### Conclusion

通篇写下来，发现并没有多少篇幅，笔者觉得原因是这篇论文发现的现象、实验验证和最后的方法确实都非常浅显易懂，改进结果也非常的直观，不禁感慨大牛的论文就是强。