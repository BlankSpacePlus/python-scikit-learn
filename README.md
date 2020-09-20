# Scikit-Learn库使用练习

版本：`0.22.1`


![Scikit-Learn](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/sklearn.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


![Scikit-Learn-With-Python](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/sklearn-python.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


> 说好的只研究Scikit-Learn，终究还是把这里变成了深入学习机器学习的笔记资料库。


## 监督学习、无监督学习、强化学习
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

### 监督学习的训练数据
基本包含两类信息：
- 输入信息：从访问日志等提取出的特征
- 输出信息：分类标签或预测值

输出的标签或值可以采用以下方法添加：
- 开发服务日志获取功能模块，从日志中获取（全自动）
- 人工浏览内容，然后添加（人工）
- 自动添加信息，由人确认（自动+人工）

### 获取训练数据的途径
- 利用公开的数据集或模型
- 开发者自己创建训练模型
- 他人帮忙输入数据
- 数据创建众包
- 集成于服务中，由用户输入

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

## 批量处理、实时处理、批次学习、逐次学习
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

## 机器学习算法评述

### 分类

#### 感知机
感知机利用了布尔逻辑的思想，更进一步的点在于其包含了更多的模糊逻辑，它通常需要基于某些阈值的达到与否来决定返回值。

感知机的特点：
- 在线学习
- 预测性能一般，但学习效率高
- 易导致过拟合
- 只能解决线性可分问题(决策边界是直线)


显然，感知机不能处理XOR（典型线性不可分）。

感知机算法结构：<br/>
![](http://latex.codecogs.com/gif.latex?sum=b+w[0]*x[0]+w[1]*x[1])


参数的权重向量的确定常用随机梯度下降法(SGD)。

感知机的激活函数应该选择类似于阶跃函数的、能将输出值进行非线性变换的函数。

感知机的损失函数是Hinge函数max(0, -twx)

#### (用于分类的)线性模型

##### 二分类线性模型
线性模型用于回归问题时，y是特征的线性函数(直线/平面/超平面)；而用于分类时，决策边界是输入的线性函数(二元线性分类器是利用直线、平面或超平面来分开两个类别的超平面)

学习线性模型有很多算法，区别在于：
- 系数和截距的特定组合对训练数据拟合好坏的度量方法（损失函数）（此点对很多应用来说不那么重要）
- 是否使用正则化，以及使用哪种正则化方法

最常见的两种线性分类算法：
- Logistic回归
- 线性支持向量机

用于分类的线性模型在低维空间看起来可能非常受限，因为决策边界只能是直线或者平面。

对线性模型系数的解释应该始终持保留态度。

###### 逻辑回归
逻辑回归命名为“回归”，由于引入了回归函数，所以也可以作为分类算法，它常被用作比较各种机器学习算法的基础算法。

逻辑回归与感知机相似，它的特点是：
- 除了输出以外，还给出输出类别的概率值
- 既可以在线学习也可以批量学习
- 预测性能一般，但学习速度快
- 为防止过拟合，可以添加正则化项（这点比感知机好）
- 只能分离线性可分数据，决策边界也是直线


逻辑回归的激活函数是Sigmoid函数，损失函数是交叉熵误差函数。

![](http://latex.codecogs.com/gif.latex?Sigmoid(x)=\frac{1}{1+e^{-x}})


`目标函数 = 所有数据的损失函数总和 + 正则化项`

- L1正则化项(用于Lasso回归)：![](http://latex.codecogs.com/gif.latex?\lambda\sum\limit_{i=1}^{m}{w_{i}^{2}})
- L2正则化项(用于Ridge回归)：![](http://latex.codecogs.com/gif.latex?\lambda\sum\limit_{i=1}^{m}{|w_{i}|})
- ElasticNet回归混合了Lasso回归和Ridge回归

##### 多分类线性模型
除了逻辑回归，绝大多数的线性分类模型只能适用于二分类问题，而不直接适用于多分类问题，推广的办法是“一对其余”。<br/>
也就是说，每个类别都学习一个二分类模型，将这个类别与其他的类别尽量分开，这样就产生了与类别个数一样多的二分类模型。<br/>
至于预测如何运行，显然是在测试点上运行所有的二分类器来预测，在对应类别上“分数最高”的分类器“胜出”，将这个类别的标签返回作为预测结果。

多分类的逻辑回归虽然不是“一对其余”，但每个类别也都有一个系数向量和一个截距，预测方法其实相同。

#### 支持向量机SVM

SVM的特点：
- 可以通过间隔最大化，学习光滑的超平面
- 使用核(Kernel)函数，能分离非线性数据(能处理线性可分数据和线性不可分数据)
- 如果是线性核，即使是高维稀疏数据也能学习
- 既可批量学习也可在线学习


损失函数是Hinge函数，与感知机类似，不同的是与横轴交点位置不同。由于相交点不同，对于在决策边界附近勉强取得正确解的数据施加微弱的惩罚项作用，也可以对决策边界产生间隔。

SVM的核心特色：
- 间隔最大化，它与正则化一样可以抑制过拟合，含义是要考虑如何拉超平面使得两个类别分别于最近的数据(支持向量)的距离最大化，其实质是决定使得间隔最大化的超平面的拉拽方法相对已知数据创造出间隔。
- 核函数方法，它针对线性不可分数据虚构地添加特征，使得数据成为高维向量，最终变成线性可分数据。（如一维线性不可分数据，升为二维，可能就变成了线性可分的）

SVM比较适合用于处理情感分析的任务。

SVM处理分类问题要做的事：最大化两个(或多个)分类之间的距离的宽度。

SVM是计算最优的，因为它映射到二次规划(凸函数最优化)。

SVM的核函数举例：
- 多项式核(异质和均匀)
- 径向基函数
- 高斯内核

SVM使用核函数的缺点：容易对数据过拟合。此时避免过拟合的一个方法是引入松弛。

#### 神经网络
神经网络在很多方面都是接近完美的机器学习结构，它可以将输入数据映射到某种常规的输出。

学习神经网络之前，可以先学习一下感知机，二者有某些相似之处。但与感知机不同的是，神经网路可以用于模拟复杂的结构。

神经网络具备识别模式以及从以前的数据中学习的能力。

对于三层前馈神经网络，其按输入层、中间层、输出层的顺序对输入和权重进行乘积加权，用所谓softmax函数对输出层最后的计算值进行正规化处理，得到概率结果。<br/>
前馈神经网络利用误差反向传播算法进行学习。根据随机初始化得到的权重值，沿着网络正向，计算输出值。再沿着网路反向，计算输出值与正确值的误差，从而修正权重值。如果权重值的修正量小于某个规定值，或者达到预设的循环次数，学习结束。

我们可能会认为简单的增加中间层的层数就使神经网路变得更加复杂，更加有效。而事实上，单纯地增加神经网络的中间层的层数，是会导致Backpropagation无法进行学习的情况。

#### KNN
KNN是一种搜索最邻近的算法。当输入一个未知的数据时，该算法根据邻近的K个已知数据所属类别，以多数表决的方式确定输入数据的类别。它不仅仅可以用作一个分类器，还可以用于搜索类似项。

K值通过交叉验证予以确定，确定距离则多使用欧氏距离，有时也会使用马哈拉诺比斯距离等。

NLP等具有高维稀疏数据的领域，常常无法直接使用KNN获得良好的预测性能，需要进行降维处理。

KNN容易理解，实现简单，不需要太多的调节就容易取得不错的性能，只要距离得当就比较容易取得不错的应用效果。

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


KNN的核心问题（导致实际用的少）：
- 预测速度慢
- 不能处理具有很多特征的数据集

#### 决策树、随机森林、GBDT
决策树是机器学习领域树状算法的代表，发展得到了随机森林、梯度提升决策树等新的算法。

##### 决策树
决策树的特点：
- 学习模型对人类而言容易理解（IF-THEN规则）
- 不需要对输入数据进行正规化处理
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

为了构造决策树，不断地对数据进行递归划分，直到划分后的每个叶子结点只包含单一的目标值（叶子结点是纯的）。

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


决策树的算法结构：<br/>
先根据训练数据确定条件式。在预测时，从树根依序追溯条件分支，直到叶子结点，然后返回预测结果。利用不纯度的基准，学习条件分支，尽量使相同类别的数据聚拢在一起。不纯度可以使用信息增益或者基尼系数（这些东西与信息论相关），数据的分割应该使得不纯度降低。利用决策树算法，应该能够得到从数据顺利分理出的IF-THEN规则树。

防止决策树过拟合的两种策略：
1. 预剪枝：及早停止树的生长，限制条件可能包括限制树的最大深度、限制叶子节点的最大数目、规定一个结点中数据点的最小数目。
2. (后)剪枝：先构造树，然后删除或折叠信息量很少的结点。

Scikit-Learn的决策树实现是DecisionTreeRegressor和DecisionTreeClassifier，其中只实现了预剪枝而没有后剪枝。

##### 随机森林和GBDT
随机森林算法先准备若干个特征组合，以多数投票表决的方式将性能好的学习机得到的多个预测结果进行集成。为了完成多棵决策树的独立学习，可采取并行学习。<br/>
随机森林算法不需要剪枝处理，所以主要参数只有两个，比GBDT少，但是随机森林容易过拟合。<br/>
随机森林预测性能比决策树高，参数少，调整起来简单顺手，决策边界看起来也很像决策树。

如果说随机森林算法通过并行学习利用预测结果，那么GBDT采用的则是针对采样数据串行学习浅层树的梯度提升方法。<br/>
它将预测值与实际值的偏差作为目标变量，一边弥补缺点，另一边由多个学习器进行学习。<br/>
由于穿性学习比较耗时，参数也比随机森林多，所以调整代价较大，不过能获得比随机森林更好的预测性能。

随机森林、GBDT这种针对多个学习结果进行组合的方法称为集成学习。单纯的决策树随着数据的增加，其学习结果大为改变，而随机森林的学习结果较为稳定。此外，集成学习的预测性能也会更好一些……

#### 贝叶斯(分类)算法
贝叶斯算法是一类算法的总称，这类算法均以贝叶斯算法为基础，所以统称贝叶斯分类算法。

贝叶斯定理我们也都知道，它是基于事件发生的条件概率而构建的一种判定方法。所以，贝叶斯分类算法就是一种以概率论为基础的分类算法。

从分类问题的角度，贝叶斯算法足以与决策树算法和神经网络算法相媲美。基于贝叶斯算法的分类器以完善的贝叶斯定理为基础，有较强的模型表示、学习和推理能力，训练过程容易理解，且对于大型数据库也表现出不错的准确率与速度。它的不足之处是泛化能力比线性分类器（前面提到的逻辑回归和线性支持向量机等）差。

贝叶斯分类器的代表有：
- 朴素贝叶斯分类器
- 贝叶斯网络分类器
- 树扩展的朴素贝叶斯分类模型TAN分类器

##### 朴素贝叶斯分类器
朴素贝叶斯分类常被称为生成式分类器，它是基于概率特征来实现分类的；而逻辑回归常被称为判别式分类器，其中的概率只是一种分类的判别方式。

朴素贝叶斯高效的原因是：它通过单独查看每个特征来学习参数，并从每个特征中手机简单的类别数据，并从每个特征中收集简单的类别统计数据。

朴素贝叶斯模型对高维稀疏数据的效果很好，对参数的鲁棒性也相对较好。它是很好的基准模型，常用于非常大的数据集（在这些数据集上即使训练线性模型也需要花费大量时间）。

scikit-learn实现了三种朴素贝叶斯分类器：
- GaussianNB：可用于任意连续数据（主要用于高维数据），保存每个类别中每个特征的数学期望和标准差
- BernoulliNB：假定输入数据为二分类数据，广泛用于稀疏数据，计算每个类别中每个特征不为0的元素个数
- MultinomialNB：假定输入数据为计数数据（即每个特征代表某个对象的整体计数），广泛用于稀疏数据，统计每个类别中每个特征的数学期望


MultinomialNB和BernoulliNB都只有一个参数alpha，它用于控制模型的复杂度。alpha的工作原理是，向算法中添加alpha这么多的虚拟数据点，这些点对所有的特征都取正值，这样可以将统计数据“平滑化”。<br/>
alpha越大，平滑化越强，模型复杂度越低。算法性能对alpha的鲁棒性较好（alpha值对模型性能影响不大），但调整此参数可能使精度略有提高。

### 回归
回归问题一共有以下几种典型的模型来处理：
- 线性回归（直线）、多项式回归（曲线）
- Lasso回归（L1）、Ridge回归（L2）、ElasticNet（L1、L2）
- 回归树（基于决策树，能处理非线性数据）
- 支持向量回归（基于SVM，能处理非线性数据）

#### 线性回归
基于回归的线性模型可以表示为这样的回归模型：对单一特征的预测结果是一条直线，两个特征时则是一个平面，在更高维度时是一个超平面。

对于有多个特征的数据集而言，线性模型可以非常强大。特别地，如果特征数量大于训练数据点的数量，则任何目标y都可以在训练集上用线性函数完美拟合。

简单线性回归模型：![](http://latex.codecogs.com/gif.latex?y={\alpha}+{\beta}x)

残差平方和：![](http://latex.codecogs.com/gif.latex?SS_{res}=\sum\limits_{i=1}^{n}(y_{i}-f(x_{i}))^2)

方差计算：![](http://latex.codecogs.com/gif.latex?var(x)=\frac{\sum\limits_{i=1}^{n}(x_{i}-\bar{x})^2}{n-1})

协方差计算：![](http://latex.codecogs.com/gif.latex?cov(x,y)=\frac{\sum_{i-1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{n-1})

![](http://latex.codecogs.com/gif.latex?\beta=\frac{cov(x,y)}{var(x)})

![](http://latex.codecogs.com/gif.latex?\alpha=\bar{y}-\beta\bar{x})

线性回归的损失函数使用L2范数来度量：<br/>
![](http://latex.codecogs.com/gif.latex?L(x)=\Vert{\hat{y}-{y}}\Vert_{2}^{2})<br/>
这种选择与欧式距离的选取有关。

#### Ridge回归
通过实测可以知道，随着模型可用的数据越来越多，两个模型的性能都在提升，最终线性回归的性能追上了岭回归。如果有足够多的训练数据，正则化变得不那么重要，线性回归和岭回归将具有相同的性能，但线性回归的训练性能在下降（如果添加更多数据，模型将更难以过拟合或记住所有数据）。

#### Lasso回归
关于Lasso回归使用的L1范数，其实就是系数的绝对值之和，其结果是：使用Lasso时的某些系数刚好为0，这说明某些特征会被模型完全忽略。这个过程可以看做一个自动化的选择过程。

实际应用选择Ridge回归的比Lasso回归的多。如果特征很多但只有几个是重要的，那选Lasso回归可能更好。

Lasso回归可能比Ridge回归更容易理解，毕竟只用了部分参数。

#### ElasticNet
这个东西综合考虑了Lasso回归的L1和Ridge回归的L2，需要调节这两个正则化项的参数。

#### 回归树
预测的方法是：<br/>
基于每个结点的测试对树进行遍历，最终找到新数据点所属的叶子结点。这一数据点的输出即为此叶子结点中所有训练点的平均目标值。

### 聚类
- 层次聚类
- K-Means

### 其他
- 推荐
- 异常检测
- 频繁模式挖掘
- 强化学习
- ……

## 机器学习算法的选择矩阵
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

## 降维
降维指的是将高维数据在尽可能保留信息的情况下转换为低维数据。

降维不仅有助于可视化，还可以将稀疏数据变为密级数据从而实现数据压缩。

## A/B测试
简而言之，A/B测试就是：<br/>
在用户不知情的情况下，为选中的一部分用户提供使用算法A的网站或服务，而为其余用户提供算法B。对于这两组用户，我们在一段时间内记录相关的成功指标，然后对算法A和算法B的指标进行对比，根据这些指标作出更优的选择。

使用A/B测试让我们能够在实际情况下评估算法，这可能有助于我们发现用户与模型交互的意外后果。

通常，A是新模型，而B是已建立的系统。

当然，还有比A/B测试更复杂的机制，如bandit算法等。

## 核外学习与集群上的并行化

### 核外学习
核外学习指的是在单台机器上从无法保存到主存储器的数据中学习。

数据从硬盘或网络等源进行读取，一次读一个样本或多个样本组成的数据块，这样每个数据块都可以读入RAM，然后处理这个数据子集并更新模型。<br/>
当这个数据块的内容处理好之后，就舍弃它，读取下一个数据块……<br/>

### 集群上的并行化
这种扩展策略是将数据分配给计算机集群的多台计算机，让每台计算机处理部分数据。

这种方式很可能会快很多，而受限的主要是集群大小，此外这种模式的基础架构比较复杂。

最常用的分布式计算平台之一是Hadoop之上构建的Spark平台。

## 数据对算法结果的影响
1. 机器学习中最重要的不是算法而是数据，数据决定了算法的能力上限，而算法只是逼近这个上限。
2. 机器学习模型从特征中进行学习，特征有多少价值，机器才可能习得多少价值。即便使用同样的模型算法，但如果用于学习的特征不同，所得结果的价值也不同，选取的特征越合适，含金量越高，最后学习到的结果也越有价值。

## 重要的概率分布
摘自 => [Here](https://github.com/graykode/distribution-is-all-you-need)

![重要的概率分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/overview.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)

### 均匀分布
均匀分布在 [a，b] 上具有相同的概率值，是简单概率分布。

![均匀分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/uniform.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


### 伯努利分布
先验概率 p(x) 不考虑伯努利分布。因此，如果我们对最大似然进行优化，那么我们很容易被过度拟合。

利用二元交叉熵对二项分类进行分类。它的形式与伯努利分布的负对数相同。

![伯努利分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/bernoulli.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


### 二项分布
参数为 n 和 p 的二项分布是一系列 n 个独立实验中成功次数的离散概率分布。<br/>
二项式分布是指通过指定要提前挑选的数量而考虑先验概率的分布。

![二项分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/binomial.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


### 多伯努利分布(分类分布)
多伯努利称为分类分布。交叉熵和采取负对数的多伯努利分布具有相同的形式。

![多伯努利分布(分类分布)](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/categorical.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


### 多项式分布
多项式分布与分类分布的关系与伯努尔分布与二项分布的关系相同。

![多项式分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/multinomial.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


### β分布
β分布与二项分布和伯努利分布共轭。

利用共轭，利用已知的先验分布可以更容易地得到后验分布。

当β分布满足特殊情况（α=1，β=1）时，均匀分布是相同的。

![β分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/beta.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


### Dirichlet分布
dirichlet 分布与多项式分布是共轭的。

如果 k=2，则为β分布。

![Dirichlet分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/dirichlet.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


### γ分布
如果 gamma(a，1)/gamma(a，1)+gamma(b，1)与 beta(a，b) 相同，则 gamma 分布为β分布。

指数分布和卡方分布是伽马分布的特例。

![γ分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/gamma.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


### 指数分布
指数分布是 α=1 时 γ 分布的特例。

![指数分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/exponential.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


### 高斯分布
高斯分布是一种非常常见的连续概率分布。

![高斯分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/gaussian.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


### 正态分布(标准高斯分布)
正态分布为标准高斯分布，数学期望为0，标准差为1。

![正态分布(标准高斯分布)](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/normal.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


### 卡方分布
k 自由度的卡方分布是 k 个独立标准正态随机变量的平方和的分布。<br/>
卡方分布是 β 分布的特例。

![卡方分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/chi-squared.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


### t分布
t分布是对称的钟形分布，与正态分布类似，但尾部较重，这意味着它更容易产生远低于平均值的值。

![t分布](https://github.com/ChenYikunReal/python-scikit-learn-training/blob/master/images/概率分布/student_t.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)


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

## Python下载库
Python第三方库的下载遇到超时失败时，可以加上两个参数：<code>--no-cache-dir</code>和<code>--default-timeout=1000</code>。<br/>
示例：
```text
pip --no-cache-dir --default-timeout=1000 install xxx
```

## AI荐读数目(仅介绍本人接触过的)

### 数学向
- 《统计学习方法》

### 半数学半理论向
- 《机器学习》
- 《机器学习算法的数学解析与Python实现》

### 理论导论向
- 《人工智能》

### 半理论半验证向（使用现成的库）
- 《Introduction to Machine Learning with Python》（《Python机器学习基础教程》）
- 《大话Python机器学习》

### 半理论半实现向（自己造轮子）

### 专题向
- 《Bayesian Analysis with Python》（《Python贝叶斯分析》）

### 应用系统向
- 《Machine Learning at Work》（《机器学习应用系统设计》） 
- 《Python人脸识别从入门到工程实践》
- 《自然语言处理Python进阶》
- 《机器学习 使用OpenCV和Python进行智能图像处理》

### 数据分析向
- 《Foundations for Analysis with Python》（《Python数据分析基础》）

### 工具库训练向
- 《Python数据可视化之matplotlib实践》
- 《Python数据可视化之matplotlib进阶》
- 《Python数据可视化编程实战》
- 《Mastering Machine Learning with scikit-learn》（《scikit-learn机器学习》）

### 深度学习
- 《Deep Learning from Scratch》（《深度学习入门》）
- 《零起点TensorFlow快速入门》
