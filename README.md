# Scikit-Learn库使用练习

版本：`0.22.1`


![Scikit-Learn](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/sklearn.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


![Scikit-Learn-With-Python](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/sklearn-python.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## 线性回归
简单线性回归模型：![](http://latex.codecogs.com/gif.latex?y={\alpha}+{\beta}x)

残差平方和：![](http://latex.codecogs.com/gif.latex?SS_{res}=\sum\limits_{i=1}^{n}(y_{i}-f(x_{i}))^2)

方差计算：![](http://latex.codecogs.com/gif.latex?var(x)=\frac{\sum\limits_{i=1}^{n}(x_{i}-\bar{x})^2}{n-1})

协方差计算：![](http://latex.codecogs.com/gif.latex?cov(x,y)=\frac{\sum_{i-1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{n-1})

![](http://latex.codecogs.com/gif.latex?\beta=\frac{cov(x,y)}{var(x)})

![](http://latex.codecogs.com/gif.latex?\alpha=\bar{y}-\beta\bar{x})

## 开发机器学习系统

### 机器学习适合的问题情形
1. 针对大量数据，能够高效稳定的做出判断
2. 允许预测结果存在一定数量的错误

### 开发流程
1. 明确定义问题
2. 考虑非机器学习的方法
3. 进行系统设计，考虑错误修正方法
4. 选择算法
5. 确定特征、训练数据和日志
6. 执行前处理
7. 学习与参数调整
8. 系统实现

### 可能遇到的困难
1. 存在概率性处理，难以进行自动化测试
2. 长期运行后，由于存储趋势变化而导致输入出现倾向性变化
3. 处理管道复杂化
4. 数据的依赖关系复杂化
5. 残留的实验代码或参数增多
6. 开发和实际的语言/框架变得支离破碎

### 设计要点
1. 如何利用预测结果
2. 弥补预测错误(系统整体如何弥补错误、是否需要人工确认或修正、必要的话去那里弥补预测错误)

### 针对可能出现问题的应对
1. 人工准备标准组对，以监控预测性能 => 1、2、4
2. 预测模型模块化，以便能够进行算法的A/B测试 => 2
3. 模型版本管理，随时可以回溯 => 4、5
4. 保管每个数据处理管道 => 3、5
5. 统一开发/实际环境的语言/框架 => 6

## 机器学习成果评价
常见基础指标：
- 准确率(Accuracy)
- 查准率(Precision)
- 召回率(Recall)
- F值(F-measure)

考虑上述指标的概念：
- 混淆矩阵(Confusion Matrix)
- 微平均(Micro-average)
- 宏平均(Macro-average)

其他指标：
- ROC曲线
- 基于ROC的AUG
- ……

更多相关内容欢迎参阅:watermelon:书

![](http://latex.codecogs.com/gif.latex?F-measure=\frac{2}{\frac{1}{Precision}+\frac{1}{Recall}})

![](http://latex.codecogs.com/gif.latex?Precision_{micro-average}=\frac{TP_{1}+TP_{2}+TP_{3}}{TP_{1}+TP_{2}+TP_{3}+FP_{1}+FP_{2}+FP_{3}})

![](http://latex.codecogs.com/gif.latex?Precision_{macro-average}=\frac{Prec_{1}+Prec_{2}+Prec_{3}}{3})

### 回归的评价
均方根误差(![](http://latex.codecogs.com/gif.latex?RMSE))：
![](http://latex.codecogs.com/gif.latex?RMSE=\sqrt{\frac{\sum_{i}(predict_{i}-actual_{i})^{2}}{N}})

```python
from math import sqrt

def rmse(predicts, actuals):
    sum = 0
    for predict, actual in zip(predicts, actuals):
        sum += (predict - actual)**2
    return sqrt(sum / len(predicts))
```

现成的函数是：`sklearn.metrics.mean_squard_error`

可决系数(![](http://latex.codecogs.com/gif.latex?R^{2}))
![](http://latex.codecogs.com/gif.latex?R^{2}=1-\frac{\sum_{i}(predict_{i}-actual_{i})^{2}}{\sum_{i}(predict_{i}-\bar{actual_{i}})^{2}})

## 札记
1. 损失函数L(x)和成本函数J(x)的区别：损失函数针对单个样本，成本函数针对整个数据集。
2. 数据预处理占了整个数据分析业务的大半时间。
3. 必须要想明白什么问题是机器学习能解决的，什么是机器学习不能解决的；还要思考一个问题用什么样的思路去训练模型更适合。
4. 训练出来的机器学习模型必须不断地维护。
5. 机器学习项目产生预想不到的预测结果的风险始终存在着。
6. 具有机器学习模块的系统的开发，实际上就是一个试错式的不断迭代的过程。
7. 开发机器学习系统可以先开发一个MVP(最简可行产品)，如先利用统计分析库或已有功能模块开发得到初步产品。
8. 即使是启动伊始就决定使用机器学习的项目，在发现没有绝对必要的情况下，也可以放弃使用机器学习。
9. 对于监督学习，如何获得高质量的标签是十分重要的，正解标签的质量直接决定问题能否顺利地得到解决。
10. 所谓数据预处理，往往要把混乱的数据整理为RDB能表现的表形式的数据，如处理加工缺失数据、剔除异常数值，克服数值范围影响、文本分词、文本统计、文本低频次剔除、图像灰度化、分类变量转换为虚拟变量等。
11. 机器学习需要不断调整算法参数，直到得到更优解。最初目标是，以由认为给出的或者根据规则库推理得到的正确结果作为基准预测性能值，然后想办法超过它。
12. 如果刚写成的预测程序畅通无阻地得到了超高预测性能，一般不是好事，可能遇到了过拟合或者数据泄露。
13. 防范过拟合的三种方法：交叉验证调整参数、正则化处理、参照学习曲线。
14. 机器学习的预测结果含有概率性处理内容，采用确定的测试是难以验证某些特定的预测结果的。
15. 通过用仪表盘监控预测性能，超出设定的阈值可发出警报，以感知长期运行时的倾向性变化。
16. 当模型经过了充分的模块化处理，就可以采用多个预测模型，组成多种互换组合，进行A/B测试(对照实验/随机实验)，更容易对多个模型进行性能预测。
17. 最好是开发过程的源代码、模型、数据的变更都能得到版本管理，随时可以回溯，辅以文档就更好了。
18. 随着开发的进行，参数可能不断增加，数据会变得越来越复杂，导致开发时和实际运行时的参数不一致，结果造成无法达到预想性能的事发生。
19. 解决实际问题时，需要先考虑好是重视精确率还是重视召回率，一般最好是确定最低限度以后再进行调整。
20. 执行数值计算时尽量不在Python系统进行处理是确保执行高速的重要措施。
