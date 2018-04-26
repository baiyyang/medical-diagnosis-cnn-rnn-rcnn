# medical-diagnosis-cnn-rnn-rcnn
分别使用rnn/cnn/rcnn模型来实现根据患者描述，进行疾病诊断

## Describe
分别使用rnn，cnn，rcnn模型来实现医疗疾病诊断，即文本分类工作
1. rnn模型如下，经过LSTM取最后一刻的输出，之后经过softmax函数分类，模型的结构如下图所示
2. cnn模型参考Yoon Kim论文《》实现，模型结构图如下所示。通过实验研究发现，由于该网络中，只取最大的权重作为最后的分类，容易存在过拟合。因此，我们改进了原始的TextCNN，提出了：
  - 基于平均特征层的卷积神经网络(Mean Features Convolutional Nerual Network for Sentence Classification，MF-TextCNN)
  - 基于全特征相连层下的卷积神经网络文本分类模型(All Features Concat Convolutional Nerual Network for Sentence Classification，AFC-TextCNN)
3. rcnn为Bi-LSTM后接高度为1，宽度为2*embedding的卷积核，之后取max-pooling，最后经过softmax函数。模型的结构如图所示。

## Requirements
python 3
tensorflow >= 1.5
numpy，zhon，jieba

## Result
