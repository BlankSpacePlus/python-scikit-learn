# 机器学习与Scikit-Learn

版本：`0.22.1`


![](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/sklearn.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


![](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/sklearn-python.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


> 说好的只研究Scikit-Learn，终究还是把这里变成了深入学习机器学习的笔记资料库。


# 监督学习、无监督学习、半监督学习、强化学习
| 类型 | 描述 |
|:---:|:---:|
| 监督学习   | 监督学习是最常见的机器学习类别，这种方法是一种函数逼近。我们尝试将一些数据点映射到一些模糊函数。通过智能优化，我们努力拟合出一个最接近将来要使用的数据的函数。|
| 无监督学习 | 无监督学习只是分析数据而没有任何类别的Y来映射，算法不知道输出应该是什么，需要自己推理。 |
| 强化学习   | 类似于监督学习，只是每个步骤都会产生反馈而已。 |

强化学习的话，举个好玩儿的例子吧：<br/>
想想我们想要训练小鼠，使其学会XX，一般会怎么做？<br/>
一般是给奖惩机制，做对给奖励(强化=>形成依赖)，做错给惩罚(电击之类的=>形成恐惧经验)<br/>
其实，强化学习就和这种操作差不多，只不过“折磨”的是计算机……<br/>
![](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/小鼠.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


![监督学习、无监督学习、强化学习](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/机器学习分类.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


半监督学习允许使用少量的已标注数据为大量的数据生成标签，它在处理大量数据时可能比较实用。<br/>
半监督学习建立的类别可能不是任务中想要的那些类别。

# 开发机器学习系统

## 机器学习适合的问题情形
1. 针对大量数据，能够高效稳定的做出判断
2. 允许预测结果存在一定数量的错误

## 开发流程
1. 明确定义问题
2. 考虑非机器学习的方法
3. 进行系统设计，考虑错误修正方法
4. 选择算法
5. 确定特征、训练数据和日志
6. 执行前处理
7. 学习与参数调整
8. 系统实现

## 可能遇到的困难
1. 存在概率性处理，难以进行自动化测试
2. 长期运行后，由于存储趋势变化而导致输入出现倾向性变化
3. 处理管道复杂化
4. 数据的依赖关系复杂化
5. 残留的实验代码或参数增多
6. 开发和实际的语言/框架变得支离破碎

## 设计要点
1. 如何利用预测结果
2. 弥补预测错误(系统整体如何弥补错误、是否需要人工确认或修正、必要的话去那里弥补预测错误)

## 针对可能出现问题的应对
1. 人工准备标准组对，以监控预测性能 => 1、2、4
2. 预测模型模块化，以便能够进行算法的A/B测试 => 2
3. 模型版本管理，随时可以回溯 => 4、5
4. 保管每个数据处理管道 => 3、5
5. 统一开发/实际环境的语言/框架 => 6

## 监督学习的训练数据
基本包含两类信息：
- 输入信息：从访问日志等提取出的特征
- 输出信息：分类标签或预测值

输出的标签或值可以采用以下方法添加：
- 开发服务日志获取功能模块，从日志中获取（全自动）
- 人工浏览内容，然后添加（人工）
- 自动添加信息，由人确认（自动+人工）

## 获取训练数据的途径
- 利用公开的数据集或模型
- 开发者自己创建训练模型
- 他人帮忙输入数据
- 数据创建众包
- 集成于服务中，由用户输入

# 机器学习数据集的划分
监督学习中，数据集常被分为`训练集、测试集`或者`训练集、测试集、验证集`：
- 训练集用于训练模型的子集，得到模型的未知参数。
- 验证集用于评估训练集的效果，用于在训练过程中检验模型的状态、收敛情况。验证集通常用于调整超参数，根据几组模型上的表现决定哪组超参数拥有较好的性能。
- 测试集用于测试训练后模型的子集，测试集用来评价模型的泛化能力，即之前模型使用验证集来确定了超参数，使用训练集调整了参数，最后使用一个从没有见过的数据集来判断这个模型的性能。

# 机器学习成果评价
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

## 回归的评价
均方根误差(![](http://latex.codecogs.com/gif.latex?RMSE))：<br/>
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

可决系数(![](http://latex.codecogs.com/gif.latex?R^{2}))：<br/>
![](http://latex.codecogs.com/gif.latex?R^{2}=1-\frac{\sum_{i}(predict_{i}-actual_{i})^{2}}{\sum_{i}(predict_{i}-\bar{actual_{i}})^{2}})

# 批量处理、实时处理、批次学习、逐次学习
- 处理方式：
    - 批量处理：成批处理某事物。
    - 实时处理：对随时传来的传感数据或日志数据的逐次处理。
- 学习方式（需要的数据群不同，学习的优化方针不同）：
    - 批次学习：权重计算需要所有的训练数据，运用所有数据才算出最优权重。
    - 逐次学习：给定一个训练数据，就立即计算一次权重。
- 可行的组合：
    - 采取批量处理方式进行批次学习（使用Web应用或者API）
    - 采用批量处理方式进行逐次学习（使用数据库）
    - 采用实时处理方式进行逐次学习

# 相关和回归
和相关回归都是研究两个变量相互关系的分析方法。

相关分析研究两个变量之间相关的方向和密切程度，但不能指出两个变量关系的具体形式，也不能从一个变量的变化来推测另一个变量的变化关系。

回归方程则是通过一定的数学方程来反映变量之间相互关系的具体形式，以便从一个已知量来推测另一个未知量。回归分析是估算预测的一种重要方法。

相关和回归的具体区别有：
1. 相关分析中变量之间处于平等地位；回归分析中因变量处于被解释的地位，自变量用于预测因变量的变化。
2. 相关分析中不必确定自变量和因变量，所涉及的变量可以都是随机变量；回归分析则必须事先确定具有相关关系的变量中，哪个是自变量，哪个是因变量。一般来说，回归分析中因变量是随机变量，而把自变量作为研究时给定的非随机变量。
3. 相关分析研究变量之间相关的方向和程度，但相关分析不能根据一个变量的变化来推测另一个变量的变化情况；回归分析是研究变量之间相互关系的具体表现形式，根据变量之间的关系来确定一个相关的数学表达式，从而可以从已知量来推测未知量。
4. 对于两个变量来说，相关分析只能计算出一个相关系数；而回归分析有时候可以根据研究目的的不同建立两个不同的回归方程。

相关分析和回归分析是广义相关分析的两个阶段，有着密切的联系：
1. 相关分析是回归分析的基础和前提，回归分析则是相关分析的深入和继续。相关分析需要依靠回归分析来表现变量之间数量变化的相关程度。只有当变量之间高度相关时，进行回归分析寻求其相关的具体形式才有意义。如果在没有对变量之间是否相关、相关方向和程度之间做出正确判断之前，就进行回归分析，很容易造成“虚假回归”。
2. 由于相关性分析只研究变量之间相关的方向和程度，不能推断变量之间相互关系的具体形式，也无法从一个变量的变化来推测另一个变量的变化情况。因此在具体应用过程中，只有把相关分析和回归分析结合起来，才能达到研究和分析的目的。

# 机器学习算法评述

## 分类

### 感知机
感知机利用了布尔逻辑的思想，更进一步的点在于其包含了更多的模糊逻辑，它通常需要基于某些阈值的达到与否来决定返回值。

感知机的思想：将线性可分的数据集中的正负样本点精确划分到两侧的超平面。

感知机的特点：
- 在线学习
- 预测性能一般，但学习效率高
- 感知机的解可以有无穷多个(它只要求训练集中所有样本都能正确分类即可)
- 易导致过拟合(甚至可以说没怎么考虑模型的泛化能力)
- 只能解决线性可分问题(决策边界是直线)

显然，感知机不能处理XOR（典型线性不可分）。

感知机算法结构：<br/>
![](http://latex.codecogs.com/gif.latex?sum=b+w[0]*x[0]+w[1]*x[1])


感知机的参数的权重向量常用随机梯度下降法(SGD)来确定。

感知机的激活函数应该选择类似于阶跃函数的、能将输出值进行非线性变换的函数。

感知机的损失函数一说是Hinge函数max(0, -twx)，一说是被误分类的样本点到当前分离超平面的相对距离的总和：<br/>
![](http://latex.codecogs.com/gif.latex?L(w,b,x,y)=-\sum\limits_{x_{i}\in{E}}{y_{i}(w\cdot{x_{i}}+b)})<br/>
这个式子也可以表示为(由于y=±1)：<br/>
![](http://latex.codecogs.com/gif.latex?L(w,b,x,y)=\sum\limits_{x_{i}\in{E}}{\vert{w\cdot{x_{i}}+b}\vert})<br/>
我们应该让这个损失函数尽可能小。值得一提的是，![](http://latex.codecogs.com/gif.latex?L(w,b,x,y))所描述的相对距离和真正意义上的欧氏距离不同。<br/>
损失函数对w和b求偏导数，就可以写出感知机模型梯度下降算法。

无论学习速率多少，感知机算法在M足够大的情况下一定能训练处一个使得E=φ的分离超平面(Novikoff定理)。

感知机和SVM类似，可以使用核技巧。核感知机的对偶算法比较简单，只需要直接使用核函数替换掉相应内积即可。

### (用于分类的)线性模型

#### 二分类线性模型
线性模型用于回归问题时，y是特征的线性函数(直线/平面/超平面)；而用于分类时，决策边界是输入的线性函数(二元线性分类器是利用直线、平面或超平面来分开两个类别的超平面)

学习线性模型有很多算法，区别在于：
- 系数和截距的特定组合对训练数据拟合好坏的度量方法（损失函数）（此点对很多应用来说不那么重要）
- 是否使用正则化，以及使用哪种正则化方法

最常见的两种线性分类算法：
- Logistic回归
- 线性支持向量机

用于分类的线性模型在低维空间看起来可能非常受限，因为决策边界只能是直线或者平面。

对线性模型系数的解释应该始终持保留态度。

##### 逻辑回归
逻辑回归命名为“回归”，由于引入了回归函数，所以也可以作为分类算法，它常被用作比较各种机器学习算法的基础算法。

线性回归中，因变量是连续的，它可以是无限数量的可能值中的任何一个；而在逻辑回归中，因变量只有有限数量的可能值。

逻辑回归与感知机相似，它的特点是：
- 除了输出以外，还给出输出类别的概率值
- 既可以在线学习也可以批量学习
- 预测性能一般，但学习速度快
- 为防止过拟合，可以添加正则化项（这点比感知机好）
- 只能分离线性可分数据，决策边界也是直线

逻辑回归的激活函数是Sigmoid函数(Logistic函数)，损失函数是交叉熵误差函数。

关于Logistic函数，可以认为它是一种“可导的阶跃函数”(单纯的阶跃函数是不连续的所以肯定不可导故不能用到机器学习中)，进而将其作为连接线性连续值和阶跃离散值的桥梁。
它的表达式是：<br/>
![](http://latex.codecogs.com/gif.latex?Sigmoid(x)=\frac{1}{1+e^{-x}})

`目标函数 = 所有数据的损失函数总和 + 正则化项`

- L1正则化项(用于Lasso回归)：![](/gif.latex?\lambda\sum\limit_{i=1}^{m}{w_{i}^{2}})
- L2正则化项(用于Ridge回归)：![](http://latex.codecogs.com/gif.latex?\lambda\sum\limit_{i=1}^{m}{|w_{i}|})
- ElasticNet回归混合了Lasso回归和Ridge回归

#### 多分类线性模型
除了逻辑回归，绝大多数的线性分类模型只能适用于二分类问题，而不直接适用于多分类问题，推广的办法是“一对其余”。<br/>
也就是说，每个类别都学习一个二分类模型，将这个类别与其他的类别尽量分开，这样就产生了与类别个数一样多的二分类模型。<br/>
至于预测如何运行，显然是在测试点上运行所有的二分类器来预测，在对应类别上“分数最高”的分类器“胜出”，将这个类别的标签返回作为预测结果。

多分类的逻辑回归虽然不是“一对其余”，但每个类别也都有一个系数向量和一个截距，预测方法其实相同。

### 支持向量机SVM
SVM的特点：
- 可以通过超平面到点集的间隔最大化，学习光滑的超平面
- 使用核(Kernel)函数，能分离非线性数据(能处理线性可分数据和线性不可分数据)
- 如果是线性核，即使是高维稀疏数据也能学习
- 既可批量学习也可在线学习
- SVM是计算最优的，因为它映射到二次规划(凸函数最优化)

损失函数是Hinge函数，与感知机类似，不同的是与横轴交点位置不同。由于相交点不同，对于在决策边界附近勉强取得正确解的数据施加微弱的惩罚项作用，也可以对决策边界产生间隔。

SVM的核心特色：
- 间隔最大化，它与正则化一样可以抑制过拟合，含义是要考虑如何拉超平面使得两个类别分别于最近的数据(支持向量)的距离最大化，其实质是决定使得间隔最大化的超平面的拉拽方法相对已知数据创造出间隔。
- 核函数方法，它针对线性不可分数据虚构地添加特征，使得数据成为高维向量，最终变成线性可分数据。（如一维线性不可分数据，升为二维，可能就变成了线性可分的）

在NLP领域，SVM比较适合用于处理情感分析的任务。

SVM处理分类问题要做的事：最大化两个(或多个)分类之间的距离的宽度。

#### 线性SVM
只要训练集D线性可分，则SVM算法对应的优化问题的解就存在且唯一。

原始算法核心步骤：<br/>
![](http://latex.codecogs.com/gif.latex?w\leftarrow{w+\eta{y_{i}x_{i}}})<br/>
![](http://latex.codecogs.com/gif.latex?b\leftarrow{b+\eta{y_{i}}})

注意对偶形式的训练过程常常会重复用到大量的样本点之间的内积，我们应该提前将样本点两两之间的内积计算出来并存储到一个矩阵(Gram矩阵)中：<br/>
![](http://latex.codecogs.com/gif.latex?G=(x_{i}\cdot{x_{j}})_{N\times{N}})

#### 非线性SVM
非线性SVM需要使用核技巧(注意核技巧重原理，核方法重应用)。

核技巧具有合理性、普适性和高效性的特点，简单来说就是奖一个低维的线性不可分的数据映射到一个高维的空间(这么说不是特别严谨，比如RBF核映射后的空间就是无限维的)，并期望映射后的数据在高维空间里是线性可分的。

当空间的维数d越大时，其中N个点线性可分的概率越大，这点曾被证明过，构成了核技巧的理论基础之一。

核技巧巧妙的地方在于：它通过定义核函数来避免显示定义映射φ，对构造低维到高维映射并使数据线性可分的过程进行转化，从而简化问题求解。<br/>
换句话说，核技巧使用核函数![](http://latex.codecogs.com/gif.latex?K(x_{i},y_{i})=\phi(x_{i})\cdot{\phi(x_{j})})替换掉算法中出现的内积![](http://latex.codecogs.com/gif.latex?x_{i}\cdot{x_{j}})来完成将数据从低维映射到高维的过程。

通过将原始问题转化为对偶问题能够非常简单地对核技巧进行应用。

核技巧的思想是：
- 将算法表述为样本点内积的组合(对偶实现)
- 设法找到核函数![](http://latex.codecogs.com/gif.latex?K(x_{i},y_{i}))，它能返回样本点![](http://latex.codecogs.com/gif.latex?x_{i})、![](http://latex.codecogs.com/gif.latex?,y_{i})被![](http://latex.codecogs.com/gif.latex?\phi)映射后的内积
- 用![](http://latex.codecogs.com/gif.latex?K(x_{i},y_{i}))替换![](http://latex.codecogs.com/gif.latex?\phi(x_{i})\cdot{\phi(x_{j})})，完成低维到高维的映射，也完成了从线性算法到非线性算法的转换
 
SVM的核函数举例：
- 多项式核(异质和均匀)：<br/>![](http://latex.codecogs.com/gif.latex?K(x_{i},x_{j})=(x_{i}\cdot{x_{j}}+1)^p)
- 径向基核：<br/>![](http://latex.codecogs.com/gif.latex?K(x_{i},x_{j})=e^{-\gamma||x_{i}-x_{j}||^2})
- 高斯核：<br/>![](http://latex.codecogs.com/gif.latex?K(x_{i},x_{j})=e^{-\frac{||x_{i}-x_{j}||^2}{2\sigma^2}})

SVM使用核函数的缺点是容易对数据过拟合，此时避免过拟合的一个方法是引入松弛。

### 神经网络
神经网络在很多方面都是接近完美的机器学习结构，它可以将输入数据映射到某种常规的输出，具备识别模式以及从以前的数据中学习的能力。

学习神经网络之前，可以先学习一下感知机，二者有某些相似之处。但与感知机不同的是，神经网路可以用于模拟复杂的结构。

M-P模型：<br/>
![](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/神经网络-MP模型.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)<br/>
![](http://latex.codecogs.com/gif.latex?y=\Phi(\sum\limits_{i=1}^n{w_{i}x_{i}+b}))<br/>
参数值说明：
- w能把从激活函数得到的函数值线性映射到另一个维度的空间上
- b能在此基础上再进行一步平移的操作

神经网络特别的一点在于它使用了一层隐藏的加权函数，称为神经元。在此隐藏层中，我们可以有效地构建一个使用了许多其他函数的网络。而如若没有隐藏层的这些函数，神经网络只是一组简单的加权函数罢了。

神经网络可以使用每层神经元的数量表示。<br/>
例如，输入层20个神经元，隐藏层10个神经元，输出层5个神经元，则可称之为20-10-5网络。

输入层：神经网络的入口点，是模型设置的输入数据的地方，这一层无神经元，因为它的主要目的是作为隐藏层的导入渠道。

神经网络的输入类型是一定要标记的，然而输入数据类型只有两种：
- 对称型：标准输入的值的范围在0和1之间，如果输入数据比较稀疏，结果可能出现偏差(甚至有崩溃的风险)。
- 标准型：对称输入的值的范围在-1和1之间，有利于防止模型因为清零效应而崩溃。它较少强调数据分布的中间位置，不确定数据可映射为0并被忽略掉。

如果没有隐藏层，神经网络将是一些线性加权线性函数的组合；隐藏层的存在给神经网络赋予了为非线性数据建模的能力。实际的神经网络的隐藏层可能有N个。<br/>
每个隐藏层包含一组神经元，这些神经元的输出将传递给输出层。

神经元是封装在激活函数中的线性加权组合。加权线性组合是将从前面所有神经元得到的数据聚合成一个数据，用于作为下一层的输入。<br/>
当神经网络一层一层输送信息时，它将前面一层的输入聚合为一个加权和。

神经网路的基本组成单元是层(Layer)而不是神经元节点，层![](http://latex.codecogs.com/gif.latex?L_{i})与层![](http://latex.codecogs.com/gif.latex?L_{i+1})之间通过权值矩阵![](http://latex.codecogs.com/gif.latex?w^{i})和偏置量![](http://latex.codecogs.com/gif.latex?b^{i})来连接。其中![](http://latex.codecogs.com/gif.latex?w^{i})能将结果从原来的维度空间映射到新的维度空间，![](http://latex.codecogs.com/gif.latex?b^{i})则能打破对称性。

神经网络模型的每一层![](http://latex.codecogs.com/gif.latex?L_{i})都有一个激活函数![](http://latex.codecogs.com/gif.latex?\Phi_{i})，激活函数是在标准范围或对称范围之间规范化数据的方式，是模型的非线性扭曲力。根据如何在训练算法中决定权重，我们需要选择不同的激活函数：

| 名称 | 标准型输入 | 对称型输入 |
|:---:|:---:|:---:|
| Sigmoid | ![](http://latex.codecogs.com/gif.latex?\frac{1}{1+e^{-2\cdot{sum}}) | ![](http://latex.codecogs.com/gif.latex?{\frac{2}{1+e^{-2\cdot{sum}}}-1) |
| Cosine | ![](http://latex.codecogs.com/gif.latex?{\frac{cos(sum)}{2}}+0.5) | ![](http://latex.codecogs.com/gif.latex?cos(sum)) |
| Sine | ![](http://latex.codecogs.com/gif.latex?{\frac{sin(sum)}{2}}+0.5) | ![](http://latex.codecogs.com/gif.latex?sin(sum)) |
| Gaussian | ![](http://latex.codecogs.com/gif.latex?\frac{1}{e^{sum^{2}}}) | ![](http://latex.codecogs.com/gif.latex?{\frac{2}{e^{sum^{2}}}}-1) |
| Elliott | ![](http://latex.codecogs.com/gif.latex?{\frac{{0.5}\cdot{sum}}{1+\vert{sum}\vert}}+0.5) | ![](http://latex.codecogs.com/gif.latex?\frac{sum}{1+\vert{sum}\vert}) |
| Linear | ![](http://latex.codecogs.com/gif.latex?sum>1?1:(sum<0:sum)) | ![](http://latex.codecogs.com/gif.latex?sum>1?1:(sum<-1?-1:sum)) |
| Threshold | ![](http://latex.codecogs.com/gif.latex?sum<0?0:1) | ![](http://latex.codecogs.com/gif.latex?sum<0?-1:1) |

其中Sigmoid是与神经元一起使用的默认函数，因为其有能力做平滑决策。

使用激活函数的最大优点在于，它们可以作为缓存每层输入值的一种方式。这一点很有用处，因为神经网络可以借此寻找模式和忽略噪声。

激活函数有两个主要类别：
- 倾斜函数：很好的默认选择
- 周期函数：用于给大量噪声的数据建模

输出层具有神经元，这是模型给出数据的地方。与输入层一样，输出的数据也是对称型或标准型的。输出层有多少个输出，是正在建模的函数决定的。

每个神经元的权重源自训练算法。训练算法通过迭代(epoch)，给每个神经元找到最优的权重值。<br/>
每个epoch中，算法遍历整个神经网络，并将其与预期的结果进行比较。如果比较结果是错误的，算法将对神经元的权重值进行相应的调整。<br/>
训练算法有很多种，比较典型的是：
- 反向传播算法：<br/>![](http://latex.codecogs.com/gif.latex?\Delta{w(t)}=-\alpha{(t-y)}\phi'{x_{i}}+\epsilon\Delta{w(t-1)})
- QuickProp算法：<br/>![](http://latex.codecogs.com/gif.latex?\Delta{w(t)}=\frac{S(t)}{S(t-1)-S(t)}\Delta{w(t-1)})
- Rprop算法(用的较多)：不根据公式计算权重变化，它仅使用权重改变量的符号，以及一个增加因子和减小因子。

这些算法有一个共同点：它们试图找到一个凸误差表面的最优解，即梯度下降。

迭代运算计算权重会比较快，这样做不试图计算关于权重的误差函数的导数，而是计算每个神经元权重的权重变化，此为delta规则：<br/>
![](http://latex.codecogs.com/gif.latex?\Delta{w_{ji}}=\alpha{(t_{j}-\phi(h_{j}))\phi'(h_{j})}x_{i})<br/>
这说明神经元j的第i个权重变化为：`alpha * (expected - calculated) * derivative_of_calculated * input_at_i`

神经网络算法流程：
1. 通过向前传导算法获取各层的激活值。
2. 通过输出层的激活值![](http://latex.codecogs.com/gif.latex?v^{m})和损失函数来做决策并获得损失。损失函数：<br/>![](http://latex.codecogs.com/gif.latex?L^{*}(x)=L(y,v^{(m)}))
3. 通过反向传播算法算出各个Layer的局部梯度：<br/>![](http://latex.codecogs.com/gif.latex?\delta^{(i)}=\frac{\partial{L(x)}}{\partial{u^{(i)}}})
4. 使用各种优化器优化参数。

构建神经网络之前，要思考到这样三个问题：
- 使用多少隐藏层？隐藏层的数量没有明确的限制，有三种启发式方法可提供帮助：
    - 不要使用两个以上的隐藏层，否则可能出现数据过拟合的情况。如果隐藏层数量太多，神经网路就会开始记忆训练数据。
    - 通常来说，一个隐藏层的工作，近似于对数据做一个连续映射。大多数神经网络中都只有一个隐藏层而已。
    - 两个隐藏层之间可能不做连续映射。这种情况比较罕见，如果不想做连续映射，可以使用两个隐藏层。
- 每个隐藏层有多少神经元？由于强调的是应该聚合而非扩展，有三种启发式方法可提供帮助：
    - 隐藏神经元的数量应该在输入神经元数量和输出神经元数量之间。
    - 隐藏神经元的数量应该为输入神经元数量的三分之二，加上输出神经元的数量。
    - 隐藏神经元的数量应该小于输入神经元的数量。
- 神经网络的容错率和最大epoch是多少？
    - 容错率太小的话，训练时间会变长。
    - 一般可以选择1000个epoch或迭代作为训练周期的起点。如此，可以建立一些复杂模型，却也不用担心训练过度。
    - 最大epoch和最大误差共同定义了解决方案的收敛点，它们作为一种信号，告诉我们可以什么时候停止算法的训练，生成一个神经网络。

损失函数：
- 距离损失函数(最小平方误差准则)：<br/>![](http://latex.codecogs.com/gif.latex?L(y,G(x))={\Vert{y-G(x)}\Vert}^{2}=[y-G(x)]^{2})
- 交叉熵损失函数(交叉熵是信息论中的概念，此函数要求G(x)的每一位取值都在(0,1)中)：<br/>![](http://latex.codecogs.com/gif.latex?L(y,G(x))=-[ylnG(x)+(1-y)ln(1-G(x))])
- log-likelihood损失函数(要求G(x)是一个概率向量，假设y∈ck)：<br/>![](http://latex.codecogs.com/gif.latex?L(y,G(x))=-lnv_{k})

神经网络向前传导和反向传播的区别：向前传导是由后往前将`激活值`一路传导，反向传播则是由前往后将`梯度`一路传播。<br/>
什么是前？什么是后？<br/>
前与后是由Layer与输出层的相对位置给出的，越靠近输出层的Layer越前，反之越靠后。

#### 前馈神经网络
对于三层前馈神经网络，其按输入层、中间层、输出层的顺序对输入和权重进行乘积加权，用所谓softmax函数对输出层最后的计算值进行正规化处理，得到概率结果。<br/>
前馈神经网络利用误差反向传播算法进行学习。根据随机初始化得到的权重值，沿着网络正向，计算输出值。再沿着网路反向，计算输出值与正确值的误差，从而修正权重值。如果权重值的修正量小于某个规定值，或者达到预设的循环次数，学习结束。

我们可能会认为简单的增加中间层的层数就使神经网路变得更加复杂，更加有效。而事实上，单纯地增加神经网络的中间层的层数，是会导致Backpropagation无法进行学习的情况。

### KNN
KNN是一种搜索最邻近的算法。当输入一个未知的数据时，该算法根据邻近的K个已知数据所属类别，以多数表决的方式确定输入数据的类别。它不仅仅可以用作一个分类器，还可以用于搜索类似项。

K值通过交叉验证予以确定，确定距离则多使用欧氏距离，有时也会使用马哈拉诺比斯距离等。

NLP等具有高维稀疏数据的领域(维度灾难)，常常无法直接使用KNN获得良好的预测性能(高维空间过于巨大以以至于其中的点根本不会表现出彼此邻近)，需要进行降维处理。

KNN容易理解，实现简单，不需要太多的调节就容易取得不错的性能，只要距离得当就比较容易取得不错的应用效果。

KNN的规则比朴素贝叶斯简单很多，因为只涉及特征与类别之间的简单关联。

KNN的两个核心参数：
- K值
- 数据点之间距离的度量方法(距离的计量方法)

K值的选择：
- 猜测式选择
- 启发式选择
    - 挑选互质的类和K值
    - 选择大于或等于类数+1的K
    - 选择足够低的K值以避免噪声(如果K值等于整个数据集则相当于默认选择最常见的类)
- 使用搜索算法选择：
    - 暴力搜索
    - 遗传算法(相关论文已附)
    - 网格搜索
    - ……

距离的度量：
- 几何距离(直观地测量一个物体上从一个点到另一个点有多远)
    - 闵可夫斯基距离：<br/>![](http://latex.codecogs.com/gif.latex?d_{p}(x,y)=(\sum\limits_{i=0}^{n}|x_{i}-y_{i}|^{p})^{\frac{1}{p}})
    - 欧几里得距离(p=2)：<br/>![](http://latex.codecogs.com/gif.latex?d(x,y)=\sqrt{\sum\limits_{i=0}^{n}(x_{i}-y_{i})^{2}})
    - 余弦距离(计算稀疏矢量之间的距离的速度快)：<br/>![](http://latex.codecogs.com/gif.latex?d(x,y)=\frac{x.y}{\Vert{x}\Vert\Vert{y}\Vert})
- 计算距离
    - 曼哈顿距离(适用于诸如图的遍历、离散优化之类的有边沿约束的问题)：<br/>![](http://latex.codecogs.com/gif.latex?d(x,y)=\sum\limits_{i=0}^{n}\vert{x_{i}-y_{i}}\vert)
    - Levenshtein距离(工作原理类似于通过改变一个邻居来制作另一个邻居的精确副本，需要改变的步骤数就是Levenshtein距离，它用于自然语言处理)：<br/>![](http://latex.codecogs.com/gif.latex?\lambda\sum\limits_{i=1}^{m}{|w_{i}|})
- 统计距离
    - 马哈拉诺比斯(Mahalanobis)距离(取成对的数据点并测量平方差)：<br/>![](http://latex.codecogs.com/gif.latex?d(x,y)=\sqrt{\sum\limits_{i=1}^{n}{\frac{(x_{i}-y_{i})^{2}}{s_{i}^{2}}}})
    - Jaccard距离(考虑到数据分类的重叠，可用于快速确定文本的相似程度)：<br/>![](http://latex.codecogs.com/gif.latex?J(X,Y)=\frac{\vert{X\cap{Y}}\vert}{\vert{X\cup{Y}}\vert})

Levenshtein距离的Python代码表示(这是递归版的，实际应用需要改成DP版的)：
```python
def lev(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    return min(lev(a[1:],b[1:])+(a[0]!=b[0]), lev(a[1:],b)+1, lev(a,b[1:])+1)
```

KNN的核心问题（导致实际用的少）：
- 预测速度慢
- 不能处理具有很多特征的数据集

### 决策树、随机森林、GBDT
决策树是机器学习领域树状算法的代表，发展得到了随机森林、梯度提升决策树等新的算法。

#### 决策树
决策树的特点：
- 学习模型对人类而言容易理解（IF-THEN规则）
- 不需要对输入数据进行正规化处理(比如归一化或标准化等)，这是因为每个特征被单独处理且数据划分不依赖于缩放
- 即使输入类的变量和残缺值（由于测量遗漏等原因无值）等，也能在内部进行自行处理
- 在特定条件下存在容易过拟合的趋势（由于条件划分，所以树的深度越深，数据量越不足，更易导致过拟合，减小深度和适当剪枝对此有帮助；特征太多也易导致过拟合，可以事先进行降维或特征选择）
- 能够处理非线性分离问题，但不擅长处理线性可分离问题（通过不断进行区域划分来生成决策边界，因此决策边界不是直线）
- 不擅长处理对每个类别的数据有依赖的数据
- 数据的微小变化容易导致预测结果出现显著改变
- 预测性能比较一般
- 只能进行批量学习
- 学习模型可视化程度高
- 可以广泛应用于分类和回归任务

决策树确实可以用树状图可视化表示，每个结点表示一个问题或一个包含答案的叶子结点。树的边将问题的答案与将问的下一个问题连接起来。

为了构造决策树，不断地对数据进行递归划分，直到划分后的每个叶子结点只包含单一的目标值（叶子结点是纯的），每一步的划分能够使得当前的信息增益达到最大。

决策树节点分裂的三种常用指标：
- 信息增益(Information gain)
- 基尼不纯度(GINI impurity)
- 方差缩减(Variance reduction)

信息增益的公式：<br/>
![](http://latex.codecogs.com/gif.latex?Gain=H_{new}-H_{prev}=H(T)-H(T|A))

基尼不纯度的公式：<br/>
![](http://latex.codecogs.com/gif.latex?I_{G}(f)=\sum\limits_{i=1}^{m}p(f_{i})(1-p(f_{i}))=1-\sum\limits_{i=1}^{m}p(f_{i})^{2})

方差缩减的公式：<br/>
![](http://latex.codecogs.com/gif.latex?\xi=E(X_{1j})-E(X_{2j})=\mu_{1}-\mu_{2})

学习决策树之前需要学习一些信息论的知识。<br/>
信息论中，熵用于度量从一个可能的结果集合中选择一个结果时所涉及的平均不确定性。

虽然划分的规则是根据数据给出的，但是划分本身其实是针对整个输入空间进行划分的。由于是NP完全问题，所以可以采取启发式方法来近似求解。

决策树常见的生成算法：
- ID3算法(交互式二分法)：比较朴素，使用互信息作为信息增益的度量，划分离散型数据，可以二分也可以多分。
- C4.5算法：使用信息增益比作为信息增益的度量，给出了混合型数据分类的解决方案。
- CART算法：规定生成出来的决策树为二叉树，且一般使用基尼增益作为信息增益的度量，可以做分类也可以做回归。

决策树的算法结构：<br/>
先根据训练数据确定条件式。在预测时，从树根依序追溯条件分支，直到叶子结点，然后返回预测结果。利用不纯度的基准，学习条件分支，尽量使相同类别的数据聚拢在一起。不纯度可以使用信息增益或者基尼系数（这些东西与信息论相关），数据的分割应该使得不纯度降低。利用决策树算法，应该能够得到从数据顺利分理出的IF-THEN规则树。

决策树生成过程：
1. 向根结点输入数据。
2. 依据信息增益的度量，选择数据的某个特征来把数据划分成互不相交的好几份并分别喂给一个新的Node。
3. 如果分完数据后发现：
    1. 某份数据的不确定较小，亦即其中某一类别的样本已经占了大多数，此时就不再对这份数据继续进行划分，将对应的Node转化为叶结点。
    2. 某份数据的不确定性仍然较大，那么这份数据就要继续分割下去。

防止决策树过拟合的两种策略：
1. 预剪枝：及早停止树的生长，限制条件可能包括限制树的最大深度、限制叶子节点的最大数目、规定一个结点中数据点的最小数目。
2. (后)剪枝：先构造树，构造完后删除或折叠信息量很少的结点。

对于决策树本身来说，即使做了预剪枝，它也常常会过拟合，泛化性能差，应该使用集成学习方法代替简单的决策树方法。

Scikit-Learn的决策树实现是DecisionTreeRegressor和DecisionTreeClassifier，其中只实现了预剪枝而没有后剪枝。

事实上，决策树虽然不像朴素贝叶斯一样几乎只有计数与统计，但决策树核心的内容——对各种信息不确定性的度量仍逃不出“计数”的范畴。

### 集成学习
集成学习是合并多个机器学习模型来构建更强大的机器学习模型的方法，它针对多个学习结果进行组合。这样的模型有很多，已证明有两种集成模型对大量分类和回归的数据集都有效，那就是随机森林和梯度提升决策树，它们都以决策树为基础。

随机森林和梯度提升决策树(集成学习)模型的预测性能可能比单纯的决策树好一些。

#### 随机森林
随机森林的本质是许多决策树的集合，其中每棵树都与其他树有所不同。

随机森林背后的思想是：每棵树的预测性能都可能较好，但可能对部分数据过拟合。如果构造多棵树，且每棵树的预测可能都相对较好，但都以不同方式拟合，那么我们就可以对这些树的结果综合处理(如取平均)来降低多拟合的影响。

数学上可以证明，随机森林既能减少过拟合，又能保持树的预测能力，所以随机森林是不错的算法。

随机森林算法先准备若干个特征组合，以多数投票表决的方式将性能好的学习机得到的多个预测结果进行集成。为了完成多棵决策树的独立学习，可采取并行学习。<br/>
随机森林算法不需要剪枝处理，所以主要参数只有两个，比GBDT少，但是随机森林容易过拟合。<br/>
随机森林预测性能比决策树高，参数少，调整起来简单顺手，决策边界看起来也很像决策树。<br/>
单纯的决策树随着数据的增加，其学习结果大为改变，而随机森林的学习结果较为稳定。

#### 梯度提升决策树(GBDT)
如果说随机森林算法通过并行学习利用预测结果，那么GBDT采用的则是针对采样数据串行学习浅层树的梯度提升方法。<br/>
它将预测值与实际值的偏差作为目标变量，一边弥补缺点，另一边由多个学习器进行学习。<br/>
由于穿性学习比较耗时，参数也比随机森林多，所以调整代价较大，不过能获得比随机森林更好的预测性能。

### 贝叶斯(分类)算法
贝叶斯算法是一类算法的总称，这类算法均以贝叶斯算法为基础，所以统称贝叶斯分类算法。

贝叶斯定理我们也都知道，它是基于事件发生的条件概率而构建的一种判定方法。所以，贝叶斯分类算法就是一种以概率论为基础的分类算法。

从分类问题的角度，贝叶斯算法足以与决策树算法和神经网络算法相媲美。基于贝叶斯算法的分类器以完善的贝叶斯定理为基础，有较强的模型表示、学习和推理能力，训练过程容易理解，且对于大型数据库也表现出不错的准确率与速度。它的不足之处是泛化能力比线性分类器（前面提到的逻辑回归和线性支持向量机等）差。

贝叶斯分类器的代表有：
- 朴素贝叶斯分类器
- 贝叶斯网络分类器
- 树扩展的朴素贝叶斯分类模型TAN分类器

#### 朴素贝叶斯分类器
朴素贝叶斯分类常被称为生成式分类器，它是基于概率特征来实现分类的；而逻辑回归常被称为判别式分类器，其中的概率只是一种分类的判别方式。<br/>
解释一下贝叶斯分类器被称为生成式分类器的原因：它仅仅是对输入的训练数据集进行了若干“计数”操作。

朴素贝叶斯是一种简单且高效的算法。<br/>
其高效的原因是：它通过单独查看每个特征来学习参数，并从每个特征中手机简单的类别数据，并从每个特征中收集简单的类别统计数据。

朴素贝叶斯模型对高维稀疏数据的效果很好，对参数的鲁棒性也相对较好。它是很好的基准模型，常用于非常大的数据集（在这些数据集上即使训练线性模型也需要花费大量时间）。

朴素贝叶斯可以处理离散型、连续型、混合型的数据。其中，离散型的朴素贝叶斯不仅能对离散数据进行分类，还能进行特征提取和可视化。

朴素贝叶斯的损失函数是0-1函数下的贝叶斯决策。

朴素贝叶斯的基本假设是条件独立性假设，但由于条件较为苛刻，所以可以通过另外两种贝叶斯分类器算法(半朴素贝叶斯和贝叶斯网)来弱化。

scikit-learn实现了三种朴素贝叶斯分类器：
- GaussianNB：可用于任意连续数据（主要用于高维数据），保存每个类别中每个特征的数学期望和标准差
- BernoulliNB：假定输入数据为二分类数据，广泛用于稀疏数据，计算每个类别中每个特征不为0的元素个数
- MultinomialNB：假定输入数据为计数数据（即每个特征代表某个对象的整体计数），广泛用于稀疏数据，统计每个类别中每个特征的数学期望


MultinomialNB和BernoulliNB都只有一个参数alpha，它用于控制模型的复杂度。alpha的工作原理是，向算法中添加alpha这么多的虚拟数据点，这些点对所有的特征都取正值，这样可以将统计数据“平滑化”。<br/>
alpha越大，平滑化越强，模型复杂度越低。算法性能对alpha的鲁棒性较好（alpha值对模型性能影响不大），但调整此参数可能使精度略有提高。

实践中，为了避免概率连乘导致数值误差过大(计算机不擅长处理浮点数，特别是接近于0的乘除)，可以转换：<br/>
![](http://latex.codecogs.com/gif.latex?y={\alpha}+{\beta}x)

## 回归
回归问题一共有以下几种典型的模型来处理：
- 线性回归（直线）、多项式回归（曲线）
- Lasso回归（L1）、Ridge回归（L2）、ElasticNet（L1、L2）
- 回归树（如CART，基于决策树，能处理非线性数据）
- 支持向量回归（SVR，基于SVM，能处理非线性数据）

### 简单线性回归
基于回归的线性模型可以表示为这样的回归模型：对单一特征的预测结果是一条直线，两个特征时则是一个平面，在更高维度时是一个超平面。

对于有多个特征的数据集而言，线性模型可以非常强大。特别地，如果特征数量大于训练数据点的数量，则任何目标y都可以在训练集上用线性函数完美拟合。

线性回归算法的优点：
- 简单易用
- 可解释性强
- 容易理解和实现

线性回归算法的缺点：
- 不能表达复杂的模式
- 对非线性问题表现不佳

简单线性回归模型：<br/>
![](http://latex.codecogs.com/gif.latex?y={\alpha}+{\beta}x)

残差平方和：<br/>
![](http://latex.codecogs.com/gif.latex?SS_{res}=\sum\limits_{i=1}^{n}(y_{i}-f(x_{i}))^2)

方差计算：<br/>
![](http://latex.codecogs.com/gif.latex?var(x)=\frac{\sum\limits_{i=1}^{n}(x_{i}-\bar{x})^2}{n-1})

协方差计算：<br/>![](http://latex.codecogs.com/gif.latex?cov(x,y)=\frac{\sum_{i-1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{n-1})

参数求解：<br/>
![](http://latex.codecogs.com/gif.latex?\beta=\frac{cov(x,y)}{var(x)})

![](http://latex.codecogs.com/gif.latex?\alpha=\bar{y}-\beta\bar{x})

线性回归的损失函数使用L2范数的平方来度量：<br/>
![](http://latex.codecogs.com/gif.latex?L(x)=\Vert{\hat{y}-{y}}\Vert_{2}^{2})<br/>
这种选择与欧式距离的选取有关。<br/>
L2范数加的这个平方与原本求解欧氏距离的根号抵消，实则是对计算结果进行了同步放大。这种放大的合理之处在于：原本误差小的会更小，而原本误差大的会更大，不会出现大小混乱的情况。

优化方法的表达式：<br/>
![](http://latex.codecogs.com/gif.latex?\min\limits_{w,b}{\Vert{\hat{y}-y}\Vert}_{2}^{2})

根据优化方法调参的方法：<br/>
![](http://latex.codecogs.com/gif.latex?w'=w-(LearningRate)*(Loss))

学习率是一个超参数，由外部输入给定，决定了每次的调整幅度。学习率低则调整慢，学习率高则可能错过最佳收敛点。

L1范数：<br/>
![](http://latex.codecogs.com/gif.latex?\Vert{x}\Vert_{1}=\sum\limits_{i=1}^{n}{\vert{x_{i}}\vert})<br/>
L2范数：<br/>
![](http://latex.codecogs.com/gif.latex?\Vert{x}\Vert_{2}=\sqrt{\sum\limits_{i=1}^{n}{x_{i}^{2}})<br/>
L0范数：向量中非0元素的个数。

简单线性回归选择最小二乘法与极大似然估计有关。<br/>
假设数据样本![](http://latex.codecogs.com/gif.latex?v_{1},\cdots{,v_{n}})服从由未知参数θ确定的概率分布：<br/>
![](http://latex.codecogs.com/gif.latex?p(v_{1},\cdots{,v_{n}}\vert{\theta})})<br/>
尽管不知道θ的值，但可以回过头来通过给定样本与θ的相似度来考量这个参数：<br/>
![](http://latex.codecogs.com/gif.latex?L(\theta\vert{v_{1},\cdots{,v_{n}}}))<br/>
按照这种方法，θ最可能的值就是最大化这个似然函数的值，即能够以最高概率产生观测数据的值。在具有概率分布函数而非概率密度函数的连续分布的情况下，我们也可以做到同样的事情。<br/>
而对于这里的简单线性回归模型，通常假设回归误差是正态分布的，其数学期望为0且标准差为σ，若是如此，可用下面的似然函数描述α和β产生(![](http://latex.codecogs.com/gif.latex?x_{i},y_{i}))的可能性大小：<br/>
![](http://latex.codecogs.com/gif.latex?L(\alpha,\beta\vert{x_{i},y_{i},\sigma})=\frac{1}{\sqrt{2\pi\sigma}}{e^{-\frac{(y_{i}-\alpha-\beta{x_{i}})^{2}}{2{\sigma}^{2}}})<br/>
由于待估计的参数产生整个数据集的可能性为产生各个数据的可能性之积，因此令误差平方和最小的α和β最有可能是我们所求的。<br/>
换而言之，在此情况下(包括这些假设)，最小化误差的平方和等价于最大化产生观测数据的可能性。

### 多重线性回归
![](http://latex.codecogs.com/gif.latex?y={\alpha}+{\beta_{1}}x_{1}+\cdots{+{\beta_{n}}x_{n}})

需要满足的额外条件：
- x的各列是线性无关的，即任何一列绝不会是其他列的加权和。
- x的各列与误差ε无关，否则会产生系统性错误。

我们应该把模型的系数看作在其他条件相同的情况下每个变量因子的影响力的大小，但切记它没有反映出变量因子之间的任何相互作用。

回归模型怕遇到的情况之一是变量是自相关的，因为那样的话系数没有意义。

这里可以关注一下无偏估计和有偏估计的概念。

### Ridge回归
Ridge回归和Lasso回归是添加了正则化的线性回归，Ridge回归使用L2范数，Lasso回归使用L1范数。<br/>
所谓正则化，指的是我们给误差项添加一个惩罚项，该惩罚项会随着β的增大而增大，能够抑制某个系数可能过大的影响(最终形成有偏估计)。<br/>
我们试图使误差项和惩罚项的组合值最小化，得到所谓最优解。

通过实测可以知道，随着模型可用的数据越来越多，两个模型的性能都在提升，最终线性回归的性能追上了岭回归。如果有足够多的训练数据，正则化变得不那么重要，线性回归和岭回归将具有相同的性能，但线性回归的训练性能在下降（如果添加更多数据，模型将更难以过拟合或记住所有数据）。

### Lasso回归
关于Lasso回归使用的L1范数，其实就是系数的绝对值之和，其结果是：使用Lasso时的某些系数刚好为0，这说明某些特征会被模型完全忽略。这个过程可以看做一个自动化的选择过程。

对比一下Ridge回归和Lasso回归：
- 关于惩罚项，Ridge回归使用L2范数(平方)，Lasso回归使用L1范数(绝对值)。
- Ridge回归的惩罚项会缩小系数，而Lasso回归的惩罚项却趋近于迫使某些系数变为0。
- 实际应用中选择Ridge回归的比Lasso回归的多，但Lasso回归适合处理稀疏模型。如果特征很多但只有几个是重要的，那选Lasso回归可能更好。
- Lasso回归可能比Ridge回归更容易理解，毕竟只用了部分参数。

### ElasticNet
这个东西综合考虑了Lasso回归的L1和Ridge回归的L2，需要调节这两个正则化项的参数。

### 回归树
预测的方法是：<br/>
基于每个结点的测试对树进行遍历，最终找到新数据点所属的叶子结点。这一数据点的输出即为此叶子结点中所有训练点的平均目标值。

回归树不能外推，也不能在训练数据范围外进行预测。一旦输入超过了训练模型数据的范围，模型就只能持续预测最后一个已知的数据点，这是所有树模型的缺陷(但是这不意味着树模型是一种很差的模型，事实上它可能会做出非常好的预测)。

## 序列归纳(序列分类)
序列分类方法可以归纳为两种基本类别：
- 基于特征的分类：将每个序列转换为一个特征向量，然后使用传统的分类方法对向量进行分类。
- 基于模型的分类：建立一个序列概率分布的内在模型，如隐马尔可夫模型(HMM)、最大熵马尔可夫模型(MEMM)、条件随机场(CRF)等统计学模型。

MEMM可作为HMM的替代方法。这个模型中，转换和观测概率矩阵由每个状态的最大熵分类器替代。它允许对概率分布进行编码从而可以在已知观测数据的情况下产生状态到状态的转换。此模型在NLP领域已应用到信息抽取和语义角色标注。

CRF模型作为一种切分序列数据的新方法被提出。它相对于HMM的一个优点是可以放松HMM和用于序列标注的随机模型中所做出的严格独立性假设。该模型可以克服MEMM等模型无法避免的标注偏见问题(网络的一种性质，导致状态转换时偏向于出度较少的状态，从而可能对序列概率分布的计算带来副作用)。<br/>
CRF中，不同状态的不同特征的权重可以相互影响。

## 聚类
聚类算法是无监督学习应用的典型，它的输入数据没有分类相关的初始信息，从这种数据中发现自然的分组以便将相似的数据项归入对应的集合中。

本质上，聚类算法能自主发现数据集自身的特征。

聚类包含为：
- 互斥聚类
- 重叠聚类
- 层次聚类
- 概率聚类
- ……

### K均值聚类
K均值聚类算法使用一个随机预定义的K(要把数据分类成的群集数量)，它将最终找到一个最优的聚类质心。

K均值聚类算法的流程：在一个数据集中选择K个随机点，定义为质心。接下来将每个点分配到最接近质心的群集中，并给它们赋予群集编号，得到一个基于原始随机质心的聚类。接下来使用数据的均值更新质心的位置。重复这一过程，直到质心不再移动。

KNN用的距离度量，诸如曼哈顿距离、欧几里得距离、闵可夫斯基距离、马哈拉诺比斯距离，这里都可以用。

k-Means的优点：
- 群集是严格的、天然的球形
- 能收敛出解决方案

K-Means的缺陷：
- 所有的分类必须具有一个硬边界，它意味着任何数据点只能在一个聚类中，而不能在它们的边界上
- 由于多使用欧几里得距离，所以更适合球面型数据，一旦位置居中的数据可以被分配到任何一个方向，就不是很适合

### 最大期望聚类
相较于关注寻找一个质心及相关数据点的K均值聚类算法，EM聚类算法关注数据是否可以分散到任意一个群集中而不关心是否有一定的模糊性。<br/>
使用EM算法的我们并不想要精确的分配，更想知道数据点分配到每个群集中的概率。

最大期望(EM)聚类算法是一个收敛于映射群集的迭代过程。它在每次迭代中进行期望和最大化。

期望是关于更新模型的真实性并查看映射情况如何。它使用TDD的方法去建立群集，我们要验证模型跟踪数据的效果如何。从数学上来看，对于数据中的每一行，我们都根据它先先前的值去估计一个概率向量。<br/>
![](http://latex.codecogs.com/gif.latex?Q(\theta\Vert\theta_{t})=E_{Z\Vert{X,\theta_{t}}}logL(\theta;X,Z))

我们还需要使用期望函数的最大值，即寻找一个θ使得对数似然度最大化：
![](http://latex.codecogs.com/gif.latex?\theta_{t}=arg{max__{\theta}Q(\theta\Vert\theta_{t})})

EM聚类的不足：当数据映射成奇异协方差矩阵时，可能不会收敛且不稳定。

### 不可能性定理
不可能性定理告诉我们，在聚类时不可能同时满足以下三个条件中的两个以上：
- 丰富性：存在一个能产生所有不同类型分区的距离函数(聚类算法可以创建数据点到群集的分配的所有类型的映射)。
- 尺度不变性：如果所有的数据都放大一个倍数，它们仍能在同一个群集中。
- 一致性：缩小群集内点与点之间的距离，然后再扩展这个距离，群集应该产生同样的结果。

## 其他
- 推荐
- 异常检测
- 频繁模式挖掘
- 强化学习
- ……

# 降维
降维指的是将高维数据在尽可能保留信息的情况下转换为低维数据。

降维不仅有助于可视化，还可以将稀疏数据变为密集数据从而实现数据压缩。

# 梯度下降
对损失函数求导，得到一个损失函数增长最快的方向，若沿其反向而行，可以以最快的速度减少，此之谓“梯度下降法”。

梯度下降法核心是求导，还要关注步长，通过调整和优化改该参数的表现，常能推导出原理一致但表现迥异的算法：
- 随机梯度下降算法(SGD)：每次迭代中只随机使用一个样本来进行参数更新
- 小批量梯度下降算法(MBGD)：每次迭代中同时选用多个样本来进行参数更新
- 批量梯度下降算法(BGD)：每次迭代中同时选用所有样本来进行参数更新

步长的选择：
- 使用固定步长
- 随时间增长逐步减小步长
- 在每一步中通过最小化目标函数的值来选择合适的步长

# A/B测试
简而言之，A/B测试就是：<br/>
在用户不知情的情况下，为选中的一部分用户提供使用算法A的网站或服务，而为其余用户提供算法B。对于这两组用户，我们在一段时间内记录相关的成功指标，然后对算法A和算法B的指标进行对比，根据这些指标作出更优的选择。

使用A/B测试让我们能够在实际情况下评估算法，这可能有助于我们发现用户与模型交互的意外后果。

通常，A是新模型，而B是已建立的系统。

当然，还有比A/B测试更复杂的机制，如bandit算法等。

# 机器学习算法的选择矩阵
选择什么样的机器学习算法与实际问题有关，我们要关注两个方面：
1. 限定偏置 => 对算法进行根本的限制，即用有些算法不可能完成这种任务
2. 优选偏置 => 表示的是算法更适合解决什么问题，这是优选的概念，首先得满足限定偏置的要求

| 算法 | 类型 | 分类 | 限定偏置 | 优选偏置 |
|:---:|:---:|:---:|:---:|:---:|
| KNN | 监督学习 | 基于实例 | 一般来说，KNN适用于测量距离相关的近似值，它受维数灾难的影响 | 在解决距离相关的问题时优先考虑该算法 |
| 朴素贝叶斯 | 监督学习 | 概率 | 适用于输入相互独立的问题 | 当每个类的概率总大于0的问题优先考虑此算法 | 当每个类的概率总大于0的问题优先考虑该算法 |
| 决策树/随机森林 | 监督学习 | 树 | 对于低协方差的问题不太有用 | 在解决分类数据问题时优先考虑该算法 |
| SVM | 监督学习 | 决策边界 | 作用于两个分类之间有明显区别时 | 在解决二分类问题时优先考虑该算法 |
| 神经网络 | 监督学习 | 非线性函数 | 几乎没有限定偏置 | 在输入总是二进制时优先考虑此算法 |
| HMM(隐马尔科夫模型) |监督学习/无监督学习 | 马尔科夫方法 | 一般适用于符合马尔科夫的系统信息 | 在处理时间序列数据和无记忆信息时优先考虑该算法 |
| 聚类算法 | 无监督学习 | 聚类 | 没有限定偏置 | 在数据是给定的某种形式的距离分组(欧几里得、曼哈顿或其他)时优先考虑该算法 |
| 特征选择 | 无监督学习 | 矩阵分解 | 没有限定偏置 | 依赖于算法，可以优先考虑高交互性的数据 |
| 特征转换 | 无监督学习 | 矩阵分解 | 必须是一个非退化矩阵 | 将会更好地工作于没有反转问题的矩阵 |
| 装袋法 | 元启发式算法 | 元启发式算法 | 几乎作用于任何事情 | 对于不是高度可变的数据很有效 |

# 文本标注任务及相关的机器学习算法
| 算法 | 任务 |
|:---:|:---:|
| 聚类 | 题材分类、垃圾邮件标注 |
| 决策树 | 语义类型或本体类别标注、指代消解 |
| 朴素贝叶斯 | 情感分类、语义类型或本体类别标注 |
| 最大熵(MaxEnt) | 情感分类、语义类型或本体类别标注 |
| 结构化模式归纳(HMM、CRF等) | 词性标注、情感分类、词义消歧 |

# 核外学习与集群上的并行化

## 核外学习
核外学习指的是在单台机器上从无法保存到主存储器的数据中学习。

数据从硬盘或网络等源进行读取，一次读一个样本或多个样本组成的数据块，这样每个数据块都可以读入RAM，然后处理这个数据子集并更新模型。<br/>
当这个数据块的内容处理好之后，就舍弃它，读取下一个数据块……<br/>

## 集群上的并行化
这种扩展策略是将数据分配给计算机集群的多台计算机，让每台计算机处理部分数据。

这种方式很可能会快很多，而受限的主要是集群大小，此外这种模式的基础架构比较复杂。

最常用的分布式计算平台之一是Hadoop之上构建的Spark平台。

# 数据对算法结果的影响
1. 机器学习中最重要的不是算法而是数据，数据决定了算法的能力上限，而算法只是逼近这个上限。
2. 机器学习模型从特征中进行学习，特征有多少价值，机器才可能习得多少价值。即便使用同样的模型算法，但如果用于学习的特征不同，所得结果的价值也不同，选取的特征越合适，含金量越高，最后学习到的结果也越有价值。

# 概率分布可视化
摘自 => [Here](https://github.com/graykode/distribution-is-all-you-need)

![重要的概率分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/overview.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)

## 均匀分布
均匀分布在 [a，b] 上具有相同的概率值，是简单概率分布。

![均匀分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/uniform.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## 伯努利分布
先验概率 p(x) 不考虑伯努利分布。因此，如果我们对最大似然进行优化，那么我们很容易被过度拟合。

利用二元交叉熵对二项分类进行分类。它的形式与伯努利分布的负对数相同。

![伯努利分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/bernoulli.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## 二项分布
参数为 n 和 p 的二项分布是一系列 n 个独立实验中成功次数的离散概率分布。<br/>
二项式分布是指通过指定要提前挑选的数量而考虑先验概率的分布。

![二项分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/binomial.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## 多伯努利分布(分类分布)
多伯努利称为分类分布。交叉熵和采取负对数的多伯努利分布具有相同的形式。

![多伯努利分布(分类分布)](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/categorical.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## 多项式分布
多项式分布与分类分布的关系与伯努尔分布与二项分布的关系相同。

![多项式分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/multinomial.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## β分布
β分布与二项分布和伯努利分布共轭。

利用共轭，利用已知的先验分布可以更容易地得到后验分布。

当β分布满足特殊情况（α=1，β=1）时，均匀分布是相同的。

![β分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/beta.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## Dirichlet分布
dirichlet 分布与多项式分布是共轭的。

如果 k=2，则为β分布。

![Dirichlet分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/dirichlet.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## γ分布
如果 gamma(a，1)/gamma(a，1)+gamma(b，1)与 beta(a，b) 相同，则 gamma 分布为β分布。

指数分布和卡方分布是伽马分布的特例。

![γ分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/gamma.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## 指数分布
指数分布是 α=1 时 γ 分布的特例。

![指数分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/exponential.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## 高斯分布
高斯分布是一种非常常见的连续概率分布。

![高斯分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/gaussian.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## 正态分布(标准高斯分布)
正态分布为标准高斯分布，数学期望为0，标准差为1。

![正态分布(标准高斯分布)](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/normal.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## 卡方分布
k 自由度的卡方分布是 k 个独立标准正态随机变量的平方和的分布。<br/>
卡方分布是 β 分布的特例。

![卡方分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/chi-squared.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## t分布
t分布是对称的钟形分布，与正态分布类似，但尾部较重，这意味着它更容易产生远低于平均值的值。

![t分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/student_t.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


# 神经网络激活函数可视化

## 逻辑函数Sigmoid
![](http://latex.codecogs.com/gif.latex?\Phi(x)=Sigmoid(x)=\frac{1}{1+e^{-x}})

![](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/激活函数/sigmoid.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## 正切函数Tanh
![](http://latex.codecogs.com/gif.latex?\Phi(x)=tanh(x)=\frac{1-e^{-2x}}{1+e^{-2x}})

![](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/激活函数/tanh.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## 线性整流函数ReLU
![](http://latex.codecogs.com/gif.latex?\Phi(x)=ReLU(x)=max(0,x))

![](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/激活函数/relu.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## ELU函数
![](http://latex.codecogs.com/gif.latex?\Phi(\alpha,x)=ELU(\alpha,x)=\begin{cases}{\alpha(e^{x}-1),x<0}\\{x,x\geq{0}}\end{cases})

![](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/激活函数/tanh.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## Softplus函数
![](http://latex.codecogs.com/gif.latex?\Phi(x)=ln(1+e^x))

![](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/激活函数/softplus.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


## 恒同映射Identity
![](http://latex.codecogs.com/gif.latex?\Phi(x)=x)

![](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/激活函数/identity.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


# 札记
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
13. 防范过拟合的方法：交叉验证调整参数、正则化处理、参照学习曲线、减少特征、选择比过拟合算法更简单的算法。
14. 机器学习的预测结果含有概率性处理内容，采用确定的测试是难以验证某些特定的预测结果的。
15. 通过用仪表盘监控预测性能，超出设定的阈值可发出警报，以感知长期运行时的倾向性变化。
16. 当模型经过了充分的模块化处理，就可以采用多个预测模型，组成多种互换组合，进行A/B测试(对照实验/随机实验)，更容易对多个模型进行性能预测。
17. 最好是开发过程的源代码、模型、数据的变更都能得到版本管理，随时可以回溯，辅以文档就更好了。
18. 随着开发的进行，参数可能不断增加，数据会变得越来越复杂，导致开发时和实际运行时的参数不一致，结果造成无法达到预想性能的事发生。
19. 解决实际问题时，需要先考虑好是重视精确率还是重视召回率，一般最好是确定最低限度以后再进行调整。
20. 执行数值计算时尽量不在Python系统进行处理是确保执行高速的重要措施。
21. 可用作特征或训练数据的信息大致有：用户信息、内容信息、用户操作日志。
22. 数据源的存储可以选择分布式RDBMS、分布式基础架构Hadoop聚类器的HDFS、对象存储，但尽量都要能使用SQL访问到数据。
23. 要理解线性可分问题和线性不可分问题，简单说，用直线无法清楚切分出来的数据可称为线性不可分数据。
24. 导致欠拟合的可能原因：有某些特征未考虑、模型表现力不足、正则化项影响过强。
25. 有关超平面的概念，《线性代数》里已经提及过，起码要知道是怎么回事。
26. 数据分析可能使用R或Python之类的编程语言，但生产团队可能会选择C++、Scala、Go、Java这样的语言来构建鲁棒性更好的可扩展系统，对此可以使用一种高性能语言在更大的框架内重新实现数据分析团队找到的解决方案。
27. 在决定开展一个机器学习的项目之前一定要有充分的思考和研究。
28. 如果将机器学习系统融入到生产环境中，要考虑到软件工程的方方面面的因素，而非简单的一次性分析。
29. 正常情况下，个人训练的数据集很难超过几百GB，此时，扩展内存或者从云端供应商租用机器成为了可行的解决方案。但是，如果处理的数据时TB级起步的，那就需要另当别论了。可以选择核外学习与集群上的并行化。
30. 正则化是指对模型做的显式约束，使得每个特征对输出的影响尽可能小，以尽量避免过拟合。
31. 复杂度小的模型意味着在训练集上的性能更差，但泛化性能可能更好，我们要在模型的简单性（系数都接近于0）与训练集性能之间做出权衡。
32. 使用Scikit-Learn训练线性模型的时候，建议指定`max_iter`。
33. 机器学习中使用优化方法的目的只有一个：通过调整假设函数的参数，另损失函数的损失值降到最小。
34. 根据NFL定律，在所有机器学习算法中，不存在最厉害的算法，所有的算法都是“平等”的，有得必有失罢了。我们能做的，是具体问题具体分析，进而找到最适合问题描述的模型。（证明相关详见:watermelon:书）
35. 回归问题和分类问题的最大区别在于预测结果是连续的还是离散的，前者预测结果是连续的，后者则是离散的。
36. 选择模型的关键不在于模型的复杂程度，而在于数据分布。
37. 机器学习可以说“在错误中学习”，包含两个内容：偏差度量(损失函数)、权值调整(优化方法)。
38. 机器学习中预测值使用![](http://latex.codecogs.com/gif.latex?\hat{y})表示，而真实值才是![](http://latex.codecogs.com/gif.latex?y)。
39. 对越高维度的空间进行建模，距离近似就会变得越不准确(维度灾难)。某一维度的指定距离在被投射到低维时会变小或不变，在投射到高维空间时会变大或不变，一旦不能保持某一点相对另一个点的距离不变，会影响所有基于距离的模型，因为所有的数据点将变得混乱且彼此远离。
40. 挖掘数据集的特征对于创建弹性模型是至关重要的。
41. 机器学习算法还是时常遇到递归的情况，这时建议使用动态规划优化算法的实现。
42. 学习是指一个系统可以从经验中改善其性能的过程。学习包括改进一个任务T，相对于一个性能度量标准P，基于经验E。
43. 机器学习算法学习目标函数的近似值，将输入数据映射到期望的输出值。
44. 一个集合或者类型的特征是否优于另一个取决于学习任务的目标函数是什么。
45. 一个机器学习算法是否优于另一个取决于为分类器产生的特征的数据对应的分类任务的范围(这里以分类为例)。
46. 想要使用梯度下降法，损失函数要处处可导。
47. 对于机器学习原始算法，可能需要使用其对偶形式来简化问题。
48. 只有对应的一位为1而其他位都是0的编码被称为one-hot representation(独热编码)。
49. 梯度是函数值上升最快的方向，所以负梯度是函数下降最快的方向。
50. 模型实际上是针对于存在于不同变量之间的数学(或概率)联系的一种规范。

# Python第三方库的下载
Python第三方库的下载遇到超时失败时，可以加上两个参数：<code>--no-cache-dir</code>和<code>--default-timeout=1000</code>。<br/>
示例：
```text
pip --no-cache-dir --default-timeout=1000 install xxx
```

# AI荐读数目(仅介绍本人接触过的)

## 数学向
- 《统计学习方法》

## 半数学半理论向
- 《机器学习》
- 《机器学习算法的数学解析与Python实现》

## 理论导论向
- 《人工智能》

## 半理论半验证向（使用现成的库）
- 《Introduction to Machine Learning with Python》（《Python机器学习基础教程》）
- 《大话Python机器学习》

## 半理论半实现向（自己造轮子）
- 《Python与机器学习实战-决策树、集成学习、支持向量机与神经网络算法详解及编程实现》

## 专题向
- 《Bayesian Analysis with Python》（《Python贝叶斯分析》）

## 应用系统向
- 《Machine Learning at Work》（《机器学习应用系统设计》） 
- 《Python人脸识别从入门到工程实践》
- 《自然语言处理Python进阶》
- 《机器学习-使用OpenCV和Python进行智能图像处理》
- 《Thoughtful Machine Learning with Python: A Test-Driven Approach》（《Python机器学习实践-测试驱动的开发方法》）

## 数据分析向
- 《Foundations for Analysis with Python》（《Python数据分析基础》）

## 工具库训练向
- 《Python数据可视化之matplotlib实践》
- 《Python数据可视化之matplotlib进阶》
- 《Python数据可视化编程实战》
- 《Mastering Machine Learning with scikit-learn》（《scikit-learn机器学习》）

## 深度学习
- 《Deep Learning from Scratch》（《深度学习入门》）
- 《零起点TensorFlow快速入门》
