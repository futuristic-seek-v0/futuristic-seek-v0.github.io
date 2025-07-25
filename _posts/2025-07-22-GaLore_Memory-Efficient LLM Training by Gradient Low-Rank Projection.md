## (ICML 2024 oral) GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection

![main](images/2025-07-22-GaLore_Memory-Efficient%20LLM%20Training%20by%20Gradient%20Low-Rank%20Projection/main.png)

### Preface（前言）

相信部署过大模型的佬们都知道，大模型的微调和训练有多么消耗资源，其中训练时存储的梯度信息占据了较大比例。为解决这一问题，类似LoRA这样的高效微调方法得到了快速发展，但那是在网络结构上进行改进，来自加州理工学院、Meta、德克萨斯大学和CMU的研究者们提出了GaLore，也就是对梯度存在的Low-Rank现象进行分析和利用，大幅降低了训练存储占用，甚至能在一张4090上训练LLaMA 7B。

### Low-Rank Adaptation

这篇文章采用的方法为Gradient Low-Rank Projection，但是更广为人知的是Low-Rank  Adaptation，这两者的大体思路其实是相同的，笔者先简要介绍Low-Rank Adaptation，也就是大名鼎鼎的LoRA。

![lora](images/2025-07-22-GaLore_Memory-Efficient%20LLM%20Training%20by%20Gradient%20Low-Rank%20Projection/lora.png)

大语言模型的能力非常强大，但在特定任务上，很多时候模型只有一部分参数起到关键作用，这一点也是比较好理解的，为了提升模型在特定任务上的性能，可以用数据集只对这一部分参数进行微调，而通过低秩的限制，可以有效做到这一点，LoRA由此产生。

LoRA 是一种参数高效（Parameter-Efficient）的适配（Adapter）微调方法，它不修改原始模型参数，而是通过添加一对低秩矩阵来学习权重更新。

### Low-Rank Gradient

微调时参数有低秩特性相信很容易理解，但是大部分人（也可能只是笔者太菜）应该都不知道梯度其实也是有低秩特性的，即Low-Rank Gradient，但是这一性质只在特定条件下成立，即在Reverse Network中成立。
