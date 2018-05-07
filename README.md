# medical-diagnosis-cnn-rnn-rcnn
分别使用rnn/cnn/rcnn模型来实现根据患者描述，进行疾病诊断

## Describe
分别使用rnn，cnn，rcnn模型来实现医疗疾病诊断，即文本分类工作
1. rnn模型如下，经过LSTM取最后一刻的输出，之后经过softmax函数分类，模型的结构如下图所示。
![](https://github.com/baiyyang/medical-diagnosis-cnn-rnn-rcnn/blob/master/images/rnn.png)
2. cnn模型参考Yoon Kim论文[《Convolutional Neural Networks for Sentence Classification》](https://arxiv.org/abs/1408.5882)实现，模型结构图如下所示。
![](https://github.com/baiyyang/medical-diagnosis-cnn-rnn-rcnn/blob/master/images/textcnn.png)

通过实验研究发现，由于该网络中，只取最大的权重作为最后的分类，容易存在过拟合。因此，我们改进了原始的TextCNN，提出了：
  - 基于平均特征层的卷积神经网络(Mean Features Convolutional Nerual Network for Sentence Classification，MF-TextCNN)
  ![](https://github.com/baiyyang/medical-diagnosis-cnn-rnn-rcnn/blob/master/images/mp-textcnn.jpg)
  - 基于全特征相连层下的卷积神经网络文本分类模型(All Features Concat Convolutional Nerual Network for Sentence Classification，AFC-TextCNN)
  ![](https://github.com/baiyyang/medical-diagnosis-cnn-rnn-rcnn/blob/master/images/afc-textcnn.jpg)
3. rcnn为Bi-LSTM后接高度为1，宽度为2*hidden_dim的卷积核，之后取max-pooling，最后经过softmax函数。模型的结构如图所示。
![](https://github.com/baiyyang/medical-diagnosis-cnn-rnn-rcnn/blob/master/images/rcnn.png)
## Requirements
- python 3
- tensorflow >= 1.5
- numpy
- zhon
- jieba

## Result
|models|presicion|note|
|:----:|:--------:|:----:|
|rnn|0.78|收敛较慢|
|textcnn|0.82|收敛较快，容易过拟合|
|MF-textcnn|0.80|收敛较慢，欠拟合|
|AFC-textcnn|0.88|收敛速度一般，准确率较高|
|rcnn|0.84|收敛较快|

## Others
欢迎各位大佬指正！
