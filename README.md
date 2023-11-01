# FaceNet-xxs
Reference：https://github.com/bubbliiiing/facenet-pytorch/tree/bilibili
## TODO List
模型构建
数据加载器和迭代器
损失函数、优化器、分布式训练
训练、结果与权重存储
测试、验证。


## done List

**`2023-11-01`**: 模型训练

**`2023-10-29`**: 模型构建

**`2023-10-28`**: 构建仓库

## 文件结构
```python

FaceNet-xxs
|-- datasets (用于存放训练数据集，一个子文件夹是一个人对应的图像)
|-- img      (人脸验证用 的数据集）
|-- lfw (Labled Faces in the Wild数据集) 
|-- logs (存放**权重**和**结果**)
|-- nets (网络模型结构)
```  