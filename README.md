# External-Attention-tensorflow


### 1.Residual Attention Usage
#### 1.1. Paper
[Residual Attention: A Simple but Effective Method for Multi-Label Recognition---ICCV2021](https://arxiv.org/abs/2108.02456)

#### 1.2 Overview
![](attention/img/ResAtt.png)

#### 1.3. UsageCode
```python
from ResidualAttention import ResidualAttention
import tensorflow as tf

input = tf.random.normal(shape=(50, 7, 7, 512))
resatt = ResidualAttention(num_class=1000, la=0.2)
output = resatt(input)
print(output.shape)
```

***

### 2.External Attention Usage
#### 2.1. Paper
["Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks"](https://arxiv.org/abs/2105.02358)

#### 2.2. Overview
![](attention/img/External_Attention.png)

#### 2.3. UsageCode
```python
from ExternalAttention import ExternalAttention
import tensorlow as tf

input = torch.randn(50, 49, 512)
ea = ExternalAttention(d_model=512, S=8)
output = ea(input)
print(output.shape)
```






参考：小马[External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)
侵权我删谢谢