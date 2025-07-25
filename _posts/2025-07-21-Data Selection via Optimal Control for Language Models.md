## (ICLR 2025 oral) Data Selection via Optimal Control for Language Models

![main](images/2025-07-21-Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/main.png)

### Preface（前言）

Data Selection一直是AI领域的重要内容，毕竟数据的质量直接决定了模型的能力。在现在的大模型时代，模型预训练规模急剧膨胀，但是高质量训练数据则相对稀缺，未区分质量的海量数据会导致训练效率低、成本高，Data Selection更显得不可或缺。

常用的Data Selection方法有Deduplication（去重）、Pattern-Based Filtering（正则关键词清洗）、Importance Resampling、Gradient Influence Methods，笔者会在后面简单讲述下这几种方法的基本原理。

这篇论文巧妙的从最优控制的视角审视Data Selection，用Pontryagin Maximum Principle的方法为Data Selection提供了新的解决方案，并通过实验验证了有效性，斩获了ICLR 2025的oral，无论是思路还是PMP工具的应用，都值得笔者写一篇blog记录一下。

### Data Selection常用方法

#### Deduplication（去重）
顾名思义，就是在大规模数据集中识别并删除重复记录，以此提升存储效率和数据质量。一般采取的方法有哈希、模糊匹配（比如通过Levenshtein距离判定）等方法，这些方法都比较好理解，也更接近人工去重的思维。
#### Pattern-Based Filtering
Pattern-Based Filtering 指的是使用可识别的数据模式（如正则表达式、重复 n‑gram、触发模板等），在大规模数据集中自动过滤掉低质量、重复性高或不相关的文本样本，从而提升训练语料的整体质量，常见做法包括：
- **重复字符串模式**： 比如“ha ha ha”
- **特定模板/触发词**：比如特定错误提示
- **长度/字符比例异常**：比如极短文本、不含字母的代码片段
- **过度重复项过滤**：比如文档里特定词组的出现频率超过一定阈值，就会被丢弃

#### Importance Resampling
Importance Resampling（IR） 在数据选择中指的是按照某种“重要性权重”对原始数据进行抽样，以得到一个近似符合目标分布（target distribution）的子集。其背景是针对预训练语言模型，需要从超大规模（raw）未标注语料中抽出样本，使其在某特征空间上的分布尽量与目标分布一致，一般包含三大步骤：
- 构建特征空间并估计分布
- 计算重要性权重
- 按权重重抽样

这里如何构造分布、如何计算重要性权重、如何重抽样，都是值得研究的问题。

#### Gradient Influence Methods

**Gradient Influence Methods** 指的是利用训练过程中的梯度信息，估计每个训练样本对目标任务（例如特定验证集上的 loss 或性能）的影响力，以此指导选择对模型提升最有效的样本。代表性方法包括 **影响函数（Influence Functions）**、**TracIn / Representer Point**、**DataInf** 以及最近的一些与聚类梯度选择相关的工作，如 [**ClusterUCB**](https://arxiv.org/abs/2506.10288)、[**Influence Distillation**](https://arxiv.org/abs/2505.19051) 。



### Method

接下来将讲解这篇论文的方法，首先我们来了解一下Pontryagin Maximum Principle。

#### Pontryagin Maximum Principle

Pontryagin Maximum Principle（庞特里亚金极大值原理，简称 PMP）是**最优控制理论**的核心结果之一，它提供了一种寻找控制系统最优解的必要条件。

我们考虑一个**动力系统控制问题**：

> 找一个控制 $$ u(t) \in U $$，使得系统状态 $$ x(t) $$ 满足动力学约束并使得某个性能指标最小（或最大）。

典型形式如下：

**状态方程（动力学约束）：**
$$
\dot{x}(t) = f(x(t), u(t), t), \quad x(t_0) = x_0
$$
**目标函数（或者叫性能指标）：**
$$
J(u) = \phi(x(t_f)) + \int_{t_0}^{t_f} L(x(t), u(t), t)\,dt
$$
其中$$ x(t) \in \mathbb{R}^n $$表示为系统状态，$$ u(t) \in U \subseteq \mathbb{R}^m $$是控制变量（需选取的对象），$$ f $$表示系统的动态函数，$$ L $$：即时成本函数，$$ \phi $$：终端成本函数（举个比较幽默的例子，比如计算开车的总开销，系统状态是速度，控制变量就是踩油门的力度，即时成本就是瞬时油耗，终端成本就是到停车场收的停车费）。

接下来，我们定义几个方程式：

**Hamiltonian 函数：**
$$
H(x, u, p, t) = p^\top f(x, u, t) + p^0 L(x, u, t)
$$
**协态方程（伴随方程）:**
$$
\dot{p}(t) = - \frac{\partial H}{\partial x}(x^*(t), u^*(t), p(t), t)
$$
上述两个方程式里的$$p$$和$$p^0$$分别是一个伴随变量和一个常数，$$ u^*(t)$$ 是一个使得目标函数$$J(u)$$极小（或极大）的最优控制，$$x^*(t)$$是最优控制对应的状态，则存在一个伴随变量（或协态变量）$$p(t) \in \mathbb{R}^n$$ 和一个常数 $$ p^0 \leq 0 $$，使得如下条件满足：

 **极大值（或极小值）条件：**：

对于所有允许的控制 $$ u \in U $$，最优控制 $$ u^*(t) $$ 应满足：

$$
u^*(t) = \arg\max_{u \in U} H(x^*(t), u, p(t), t)
$$
或者（极小问题）：

$$
u^*(t) = \arg\min_{u \in U} H(x^*(t), u, p(t), t)
$$
且
$$
\dot{x}^*(t) = \frac{\partial H}{\partial p}(x^*(t), u^*(t), p(t), t)
$$
若 $$ t_f $$ 是自由变量，则还需要满足：

$$
H(x^*(t_f), u^*(t_f), p(t_f), t_f) = 0
$$
这些公式每一个都还不是那么难理解，但是组合在一起难免变得抽象，笔者举了一个例子供参考。

比如说$$x(t)$$表示一辆车的路程，$$u(t)$$表示速度，可以理解为和油耗成线性正比，那么最小化燃料消耗等价于：
$$
\min \int_0^T u(t)^2 \, dt
$$
在
$$
\dot{x}(t) = u(t), \quad x(0) = 0, \quad x(T) = x_T
$$
的约束下进行求解。

按照前文讲述的方法，先构造Hamiltonian方程：
$$
H = p(t) u(t) + u(t)^2
$$
然后定义状态和协态方程：
$$
\dot{x}(t) = u(t)
$$

$$
\dot{p}(t) = -\frac{\partial H}{\partial x} = 0 \Rightarrow p(t) = \text{const},
$$

极小值条件为
$$
\frac{\partial H}{\partial u} = p + 2u = 0 \Rightarrow u^*(t) = -\frac{p}{2},
$$
代入求解：
$$
\dot{x}(t) = -\frac{p}{2}, \quad x(0) = 0 \Rightarrow x(t) = -\frac{p}{2} t
$$

$$
x(T) = x_T = -\frac{p}{2} T \Rightarrow p = -\frac{2x_T}{T}
$$

$$
\Rightarrow u^*(t) = \frac{x_T}{T}
$$

得到最优控制方法就是匀速行驶。

### PDS(PMP-based Data Selection)

介绍完了PMP算法，接下来就可以直入主题，讲解论文所采用的PMP算法了。

首先需要明确，论文是怎样将Data Selection看作是一个控制问题的。既然是Data Selection，**控制变量**设计为数据集中各个样本的权重$$\gamma = [\gamma_1, \dots, \gamma_N]$$，状态变量设计为模型的参数$$\theta_t$$，优化目标则是累积下游验证损失，即
$$
\min_{\gamma \in U} \sum_{t=1}^T J(\theta_t)
   \quad \text{s.t.} \quad \theta_{t+1} = \theta_t - \eta \nabla L(\theta_t, \gamma),
$$

$$
L(\theta_t, \gamma) = \sum_{n=1}^{|D_{trn}|} \gamma_n \ell(x_n, \theta_t),
$$

其中$$U$$是可行域，$$J(\theta_t)$$是在验证集上的损失函数，$$\ell(x_n, \theta)=-\log(p_{\theta}(x_n))$$是单样本的对数损失。

控制问题设计好后，接下来就是用PMP进行求解了，引用论文里的符号，令最优控制为 $$\gamma^*$$，状态轨迹为 $$\{\theta_t^*\}$$，则存在伴随向量 $$\{\lambda_t^*\}$$ 使得：

**前向状态演化**（参数更新）：  
$$
\theta_{t+1}^* = \theta_t^* - \eta \nabla L(\theta_t^*, \gamma^*)
$$
**反向伴随递推**：
$$
\lambda_t^* = \lambda_{t+1}^* + \nabla J(\theta_t^*) - \eta \nabla^2 L(\theta_t^*, \gamma^*) \, \lambda_{t+1}^*
$$
**最大化 Hamiltonian ⇒ 数据选择准则**：
每一步 $$t$$ 应选取 $$\gamma$$ 最大化 Hamiltonian，也就是选择样本使$$<\lambda_{t+1}^*, \nabla \ell(x_i, \theta_t^*)>$$最大， 因此，给样本打分为：$$s_i \propto \langle \lambda_{t+1}^*, \nabla \ell(x_i, \theta_t^*) \rangle$$

最终的算法如图所示：

![algorithm](images/2025-07-21-Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/algorithm.png)

我知道这部分公式出现的非常唐突，笔者把论文里的证明过程贴在这里，其中Theorem B.1的证明跟笔者所写的一致，只是符号不同：

![proof1](images/2025-07-21-Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/proof1.png)

![proof2](images/2025-07-21-Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/proof2.png)

![proof3](images/2025-07-21-Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/proof3.png)

![proof4](images/2025-07-21-Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/proof4.png)

>[!NOTE]
>
>真实训练中，$$\gamma$$是离线且固定的，因此这个scorer实在小模型和代理语料上进行代理训练，随后在大语料上用scorer输出样本的质量分数，然后筛选最终的训练数据集。

![framework](images/2025-07-21-Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/framework.png)

### Experiments

论文作者采用与Mistral相同的LM架构，在160M、470M、1B和1.7B四个参数规模上都进行了Pre-Training，数据集为从Redpajama里分离出的CommonCrawl。

首先，论文比较了各类Data Selection的方法，采用OLMo论文里的多个数据集进行了0-shot accuracy测试，结果均有提升：

![exp1](images/2025-07-21-Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/exp1.png)

此外，作者在DCLM语料库上记录了Test Loss，相比较下，采用PDS的loss值下降的也更快：

![exp2](images/2025-07-21-Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/exp2.png)

同时，实验表示该方法在大模型上依旧有效，验证了Scaling Law：

![exp3](images/2025-07-21-Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/exp3.png)



### Conclusion

笔者本科就是学自动化的，从来没想过可以这样把控制算法用在Data Selection里，这么看PMP也应该是一件比较趁手的武器，举一反三，感觉在其它领域、其它的控制算法也有不小的应用空间（看样子本科学自动化也不错doge）。
