## (ICLR 2025 oral) Data Selection via Optimal Control for Language Models

![main](images/2025-07-21-Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/main.png)

### 前言

Data Selection一直是AI领域的重要内容，毕竟数据的质量直接决定了模型的能力。在现在的大模型时代，模型预训练规模急剧膨胀，但是高质量训练数据则相对稀缺，未区分质量的海量数据会导致训练效率低、成本高，Data Selection更显得不可或缺。

常用的Data Selection方法有Deduplication（去重）、Pattern-Based Filtering（正则关键词清洗）、Importance Resampling、Gradient Influence Methods，笔者会在后面简单讲述下这几种方法的基本原理和各个方向上的进展。

这篇论文巧妙的从最优控制的视角审视Data Selection，用Pontryagin Maximum Principle的方法为Data Selection提供了新的解决方案，并通过实验验证了有效性，斩获了ICLR 2025的oral，无论是思路还是PMP工具的应用，都值得笔者写一篇blog记录一下。

### Data Selection常用方法

#### Deduplication（去重）



