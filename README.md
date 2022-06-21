# [External-Attention-tensorflow](https://github.com/ccfco-Ivan/External-Attention-tensorflow)



## Contents
- [Attention Series](#attention-series)
  - [1. Residual Attention Usage](#1-residual-attention-usage)
  - [2. External Attention Usage](#2-external-attention-usage)
  - [3. Self Attention Usage](#3-self-attention-usage)
  - [4. Simplified Self Attention Usage](#4-simplified-self-attention-usage)
  - [5. Squeeze-and-Excitation Attention Usage](#5-squeeze-and-excitation-attention-usage)
  - [6. SK Attention Usage](#6-sk-attention-usage)
  - [7. CBAM Attention Usage](#7-cbam-attention-usage)
  - [8. BAM Attention Usage](#8-bam-attention-usage)
  - [9. ECA Attention Usage](#9-eca-attention-usage)

## Attention Series

### 1. Residual Attention Usage
#### 1.1. Paper
[Residual Attention: A Simple but Effective Method for Multi-Label Recognition---ICCV2021](https://arxiv.org/abs/2108.02456)

#### 1.2 Overview
![](attention/img/ResAtt.png)

> Only 4 lines of code consistently leads to improvement of multi-label recognition, across many diverse pretrained models and datasets, even without any extra training.
（在许多不同的预训练模型和数据集上，即使没有任何额外的训练，只用4行代码也可以提高多标签识别的准确率）

#### 1.3. UsageCode
```python
from attention.ResidualAttention import ResidualAttention
import tensorflow as tf

input = tf.random.normal(shape=(50, 7, 7, 512))
resatt = ResidualAttention(num_class=1000, la=0.2)
output = resatt(input)
print(output.shape)
```

***

### 2. External Attention Usage
#### 2.1. Paper
["Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks"](https://arxiv.org/abs/2105.02358)

#### 2.2. Overview
![](attention/img/External_Attention.png)

>主要解决的Self-Attention(SA)的两个痛点问题：（1）O(n^2)的计算复杂度；(2)SA是在同一个样本上根据不同位置计算Attention，忽略了不同样本之间的联系。因此，本文采用了两个串联的MLP结构作为memory units，使得计算复杂度降低到了O(n)；此外，这两个memory units是基于全部的训练数据学习的，因此也隐式的考虑了不同样本之间的联系。

#### 2.3. UsageCode
```python
from attention.ExternalAttention import ExternalAttention
import tensorflow as tf

input = tf.random.normal(shape=(50, 49, 512))
ea = ExternalAttention(d_model=512, S=8)
output = ea(input)
print(output.shape)
```

***

### 3. Self Attention Usage
#### 3.1. Paper
["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf)

#### 3.2. Overview
![](attention/img/SA.png)

>这是Google在NeurIPS2017发表的一篇文章，在CV、NLP、多模态等各个领域都有很大的影响力，目前引用量已经4.5w+。Transformer中提出的Self-Attention是Attention的一种，用于计算特征中不同位置之间的权重，从而达到更新特征的效果。首先将input feature通过FC映射成Q、K、V三个特征，然后将Q和K进行点乘的得到attention map，再将attention map与V做点乘得到加权后的特征。最后通过FC进行特征的映射，得到一个新的特征。

#### 3.3. UsageCode
```python
from attention.SelfAttention import ScaledDotProductAttention
import tensorflow as tf

input = tf.random.normal((50, 49, 512))
sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
output = sa(input, input, input)
print(output.shape)
```

***

### 4. Simplified Self Attention Usage
#### 4.1. Paper
[None]()

#### 4.2. Overview
![](attention/img/SSA.png)

#### 4.3. UsageCode
```python
from attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
import tensorflow as tf

input = tf.random.normal((50, 49, 512))
ssa = SimplifiedScaledDotProductAttention(d_model=512, h=8)
output = ssa(input, input, input)
print(output.shape)
```

***

### 5. Squeeze-and-Excitation Attention Usage
#### 5.1. Paper
["Squeeze-and-Excitation Networks"](https://arxiv.org/abs/1709.01507)

#### 5.2. Overview
![](attention/img/SE.png)

>这是CVPR2018的一篇文章，是做通道注意力的，因其简单的结构和有效性，将通道注意力掀起了一波小高潮。大道至简，这篇文章的思想非常简单，首先将spatial维度进行AdaptiveAvgPool，然后通过两个FC学习到通道注意力，并用Sigmoid进行归一化得到Channel Attention Map,最后将Channel Attention Map与原特征相乘，就得到了加权后的特征。

#### 5.3. UsageCode
```python
from attention.SEAttention import SEAttention
import tensorflow as tf

input = tf.random.normal((50, 7, 7, 512))
se = SEAttention(channel=512, reduction=8)
output = se(input)
print(output.shape)
```

***

### 6. SK Attention Usage
#### 6.1. Paper
["Selective Kernel Networks"](https://arxiv.org/pdf/1903.06586.pdf)

#### 6.2. Overview
![](attention/img/SK.png)

>这是CVPR2019的一篇文章，致敬了SENet的思想。在传统的CNN中每一个卷积层都是用相同大小的卷积核，限制了模型的表达能力；而Inception这种“更宽”的模型结构也验证了，用多个不同的卷积核进行学习确实可以提升模型的表达能力。作者借鉴了SENet的思想，通过动态计算每个卷积核得到通道的权重，动态的将各个卷积核的结果进行融合。

>本文的方法分为三个部分：Split,Fuse,Select。Split就是一个multi-branch的操作，用不同的卷积核进行卷积得到不同的特征；Fuse部分就是用SE的结构获取通道注意力的矩阵(N个卷积核就可以得到N个注意力矩阵，这步操作对所有的特征参数共享)，这样就可以得到不同kernel经过SE之后的特征；Select操作就是将这几个特征进行相加。

#### 6.3. UsageCode
```python
from attention.SKAttention import SKAttention
import tensorflow as tf

input = tf.random.normal((50, 7, 7, 512))
se = SKAttention(channel=512, reduction=8)
output = se(input)
print(output.shape)
```

***

### 7. CBAM Attention Usage
#### 7.1. Paper
["CBAM: Convolutional Block Attention Module"](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)

#### 7.2. Overview
![](attention/img/CBAM1.png)

![](attention/img/CBAM2.png)

>这是ECCV2018的一篇论文，这篇文章同时使用了Channel Attention和Spatial Attention，将两者进行了串联（文章也做了并联和两种串联方式的消融实验）。

>Channel Attention方面，大致结构还是和SE相似，不过作者提出AvgPool和MaxPool有不同的表示效果，所以作者对原来的特征在Spatial维度分别进行了AvgPool和MaxPool，然后用SE的结构提取channel attention，注意这里是参数共享的，然后将两个特征相加后做归一化，就得到了注意力矩阵。

>Spatial Attention和Channel Attention类似，先在channel维度进行两种pool后，将两个特征进行拼接，然后用7x7的卷积来提取Spatial Attention（之所以用7x7是因为提取的是空间注意力，所以用的卷积核必须足够大）。然后做一次归一化，就得到了空间的注意力矩阵。

#### 7.3. Usage Code
```python
from attention.CBAM import CBAMBlock
import tensorflow as tf

input = tf.random.normal((50, 7, 7, 512))
kernel_size = input.get_shape()[1]
cbam = CBAMBlock(channel=512, reduction=16, kernel_size=kernel_size)
output = cbam(input)
print(output.shape)
```

***

### 8. BAM Attention Usage
#### 8.1. Paper
["BAM: Bottleneck Attention Module"](https://arxiv.org/pdf/1807.06514.pdf)

#### 8.2. Overview
![](attention/img/BAM.png)

>这是CBAM同作者同时期的工作，工作与CBAM非常相似，也是双重Attention，不同的是CBAM是将两个attention的结果串联；而BAM是直接将两个attention矩阵进行相加。

>Channel Attention方面，与SE的结构基本一样。Spatial Attention方面，还是在通道维度进行pool，然后用了两次3x3的空洞卷积，最后将用一次1x1的卷积得到Spatial Attention的矩阵。

>最后Channel Attention和Spatial Attention矩阵进行相加（这里用到了广播机制），并进行归一化，这样一来，就得到了空间和通道结合的attention矩阵。

#### 8.3. Usage Code
```python
from attention.BAM import BAMBlock
import tensorflow as tf

input = tf.random.normal((50, 7, 7, 512))
bam = BAMBlock(channel=512, reduction=16, dia_val=2)
output = bam(input)
print(output.shape)
```

***

### 9. ECA Attention Usage
#### 9.1. Paper
["ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"](https://arxiv.org/pdf/1910.03151.pdf)

#### 9.2. Overview
![](attention/img/ECA.png)

>这是CVPR2020的一篇文章。 如上图所示，SE实现通道注意力是使用两个全连接层，而ECA是需要一个的卷积。作者这么做的原因一方面是认为计算所有通道两两之间的注意力是没有必要的，另一方面是用两个全连接层确实引入了太多的参数和计算量。

>因此作者进行了AvgPool之后，只是使用了一个感受野为k的一维卷积（相当于只计算与相邻k个通道的注意力），这样做就大大的减少的参数和计算量。(i.e.相当于SE是一个global的注意力，而ECA是一个local的注意力)。

#### 9.3. Usage Code
```python
from attention.ECAAttention import ECAAttention
import tensorflow as tf

input = tf.random.normal((50, 7, 7, 512))
eca = ECAAttention(kernel_size=3)
output = eca(input)
print(output.shape)

```

***



***

参考：小马[External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)
侵权我删谢谢