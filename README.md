# [External-Attention-tensorflow](https://github.com/ccfco-Ivan/External-Attention-tensorflow)

[![OSCS Status](https://www.oscs1024.com/platform/badge/ccfco-Ivan/External-Attention-tensorflow.svg?size=small)](https://www.oscs1024.com/project/ccfco-Ivan/External-Attention-tensorflow?ref=badge_small)

## Contents
- [Attention Series](#attention-series)
  - [1. Residual Attention Usage---ICCV2021](#1-residual-attention-usage)
  - [2. External Attention Usage---arXiv 2021.05.05](#2-external-attention-usage)
  - [3. Self Attention Usage---NIPS2017](#3-self-attention-usage)
  - [4. Simplified Self Attention Usage](#4-simplified-self-attention-usage)
  - [5. Squeeze-and-Excitation Attention Usage---CVPR2018](#5-squeeze-and-excitation-attention-usage)
  - [6. SK Attention Usage---CVPR2019](#6-sk-attention-usage)
  - [7. CBAM Attention Usage---ECCV2018](#7-cbam-attention-usage)
  - [8. BAM Attention Usage---BMCV2018](#8-bam-attention-usage)
  - [9. ECA Attention Usage---CVPR2020](#9-eca-attention-usage)
  - [10. DANet Attention Usage---CVPR2019](#10-danet-attention-usage)
  - [11. Pyramid Split Attention Usage---arXiv 2021.05.30](#11-pyramid-split-attention-usage)
  - [12. Efficient Multi-Head Self-Attention Usage---arXiv 2021.05.28](#12-efficient-multi-head-self-attention-usage)
  - [13. Shuffle Attention Usage---arXiv 2021.01.30](#13-shuffle-attention-usage)
  - [14. MUSE Attention Usage---arXiv 2019.11.17](#14-muse-attention-usage)
  - [15. SGE Attention Usage---arXiv 2019.05.23](#15-sge-attention-usage)

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
>
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
>
>Channel Attention方面，大致结构还是和SE相似，不过作者提出AvgPool和MaxPool有不同的表示效果，所以作者对原来的特征在Spatial维度分别进行了AvgPool和MaxPool，然后用SE的结构提取channel attention，注意这里是参数共享的，然后将两个特征相加后做归一化，就得到了注意力矩阵。
>
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
>
>Channel Attention方面，与SE的结构基本一样。Spatial Attention方面，还是在通道维度进行pool，然后用了两次3x3的空洞卷积，最后将用一次1x1的卷积得到Spatial Attention的矩阵。
>
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
>
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
### 10. DANet Attention Usage
#### 10.1. Paper
["Dual Attention Network for Scene Segmentation"](https://arxiv.org/pdf/1809.02983.pdf)

#### 10.2. Overview
![](attention/img/danet.png)![](attention/img/danet2.png)
>这是CVPR2019的文章，思想上就是将self-attention用到场景分割的任务中，不同的是self-attention是关注每个position之间的注意力，而本文将self-attention做了一个拓展，还做了一个通道注意力的分支，操作上和self-attention一样，不同的通道attention中把生成Q，K，V的三个Linear去掉了。最后将两个attention之后的特征进行element-wise sum。

#### 10.3. Usage Code
```python
from attention.DANet import DAModule
import tensorflow as tf

input = tf.random.normal((50, 7, 7, 512))
danet = DAModule(d_model=512, kernel_size=3, H=7, W=7)
print(danet(input).shape)
```
***

### 11. Pyramid Squeeze Attention Usage

#### 11.1. Paper
["EPSANet: An Efficient Pyramid Squeeze Attention Block on Convolutional Neural Network"](https://doi.org/10.48550/arXiv.2105.14447)

#### 11.2. Overview
![Pyramid Squeeze Attention (PSA) module](attention/img/psa.jpg)![A detailed illustration of Squeeze and Concat(SPC) module](attention/img/psa2.jpg)
>这是深大2021年5月30日在arXiv上上传的一篇文章，本文的目的是如何获取并探索不同尺度的空间信息来丰富特征空间。网络结构相对来说也比较简单，主要分成四步，第一步，将原来的feature根据通道分成n组然后对不同的组进行不同尺度的卷积，得到新的特征W1；第二步，通过使用SE权重模块提取不同尺度的特征图的注意力，得到channel-wise attention向量；第三步，对不同组进行softmax；第四步，将获得channel-wise attention与原来的特征W1相乘。

#### 11.3. Usage Code
```python
from attention.PSA import PSA
import tensorflow as tf

input = tf.random.normal((50, 7, 7, 512))
psa = PSA(channel=512, reduction=8)
output = psa(input)
print(output.shape)
```

***

### 12. Efficient Multi-Head Self-Attention Usage

#### 12.1. Paper
["ResT: An Efficient Transformer for Visual Recognition"](https://arxiv.org/abs/2105.13677)

#### 12.2. Overview
![](attention/img/EMSA.jpg)
>这是南大5月28日在arXiv上上传的一篇文章。本文解决的主要是SA的两个痛点问题：（1）Self-Attention的计算复杂度和n呈平方关系；（2）每个head只有q,k,v的部分信息，如果q,k,v的维度太小，那么就会导致获取不到连续的信息，从而导致性能损失。这篇文章给出的思路也非常简单，在SA中的FC之前，用了一个卷积来降低了空间的维度，从而得到空间维度上更小的K和V。

#### 12.3. Usage Code
```python
from attention.EMSA import EMSA
import tensorflow as tf

input = tf.random.normal((50, 64, 512))
emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8, H=8, W=8, ratio=2, apply_transform=True)
output = emsa(input, input, input)
print(output.shape)
```
***


### 13. Shuffle Attention Usage

#### 13.1. Paper
["SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS"](https://arxiv.org/pdf/2102.00240.pdf)

#### 13.2. Overview
![](attention/img/ShuffleAttention.png)
>采用Shuffle Units将两种注意力机制有效结合。具体来说，SA首先将通道维度分组为多个子特征，然后并行处理它们。其次，对于每个子特征，SA使用Shuffle Unit来描述空间和通道维度上的特征依赖关系。最后，对所有子特征进行聚合，并采用“channel shuffle”算子来实现不同子特征之间的信息通信。

#### 13.3. Usage Code
```python
from  attention.ShuffleAttention import ShuffleAttention
import tensorflow as tf

input = tf.random.normal((50, 7, 7, 512))
se = ShuffleAttention(channel=512, G=8)
output = se(input)
print(output.shape)
```

***


### 14. MUSE Attention Usage

#### 14.1. Paper
["MUSE: Parallel Multi-Scale Attention for Sequence to Sequence Learning"](https://arxiv.org/abs/1911.09483)

#### 14.2. Overview
![](./attention/img/MUSE.png)
>这是北大团队2019年在arXiv上发布的一篇文章，主要解决的是Self-Attention（SA）只有全局捕获能力的缺点。如下图所示，当句子长度变长时，SA的全局捕获能力变弱，导致最终模型性能变差。因此，作者在文中引入了多个不同感受野的一维卷积来捕获多尺度的局部Attention，以此来弥补SA在建模长句子能力的不足。
![](attention/img/MUSE2.jpg)
> 
>实现方式如模型结构所示的那样，将SA的结果和多个卷积的结果相加，不仅进行全局感知，还进行局部感知。最终通过引入多尺度的局部感知，使模型在翻译任务上的性能得到了提升。

#### 14.3. Usage Code
```python
from attention.MUSEAttention import MUSEAttention
import tensorflow as tf

input = tf.random.normal((50, 49, 512))
sa = MUSEAttention(d_model=512, d_k=512, d_v=512, h=8)
output = sa(input, input, input)
print(output.shape)
```
***


### 15. SGE Attention Usage

#### 15.1. Paper
[Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks](https://arxiv.org/pdf/1905.09646.pdf)

#### 14.2. Overview
![](attention/img/SGE.jpg)
>这篇文章是[SKNet](#6-sk-attention-usage)作者在19年的时候在arXiv上挂出的文章，是一个轻量级Attention的工作，从核心代码中可以看出，引入的参数真的非常少，self.weight和self.bias都是和groups呈一个数量级的（几乎就是常数级别）。
>
>这篇文章的核心点是用局部信息和全局信息的相似性来指导语义特征的增强，总体的操作可以分为以下几步：
>>1. 将特征分组，每组feature在空间上与其global pooling后的feature做点积（相似性）得到初始的attention mask 
>>2. 对该attention mask进行减均值除标准差的normalize，并同时每个group学习两个缩放偏移参数使得normalize操作可被还原
>>3. 最后经过sigmoid得到最终的attention mask并对原始feature group中的每个位置的feature进行scale
>
> 实验部分，作者也是在分类任务（ImageNet）和检测任务（COCO）上做了实验，能够在比[SK](#6-sk-attention-usage)、[CBAM](#7-cbam-attention-usage)、[BAM](#8-bam-attention-usage)等网络参数和计算量更小的情况下，获得更好的性能，证明了本文方法的高效性。


#### 15.3. Usage Code
```python
from attention.SGE import SpatialGroupEnhance
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
sge = SpatialGroupEnhance(groups=8)
output=sge(input)
print(output.shape)

```


***

参考：小马[External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)
侵权我删谢谢