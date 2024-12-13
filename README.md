数学建模-基于决策树的深度学习编译优化-实验报告

#  实验一

1、在服务器上，根据提供的数据集（不同调度的pickel格式文件），使用训练数据对指定的Cost Model进行训练(XGBoost)。完成训练后，使用测试集测得性能预测结果(吞吐量)。

2、在提交平台上，上传10组性能预测结果，通过自动评分程序处理，得到RMSE的值。

## 实验目的

1、了解深度学习模型的训练过程及编译方法。

2、学习构造基于决策树的张量优化模型。

3、熟悉深度学习模型在服务器的应用方法。



## 实验原理

本实验搭建一个深度学习模型Cost-Model，实现一个预测模型以预测调度的性能变化。

Cost Model是一个预测模型，它可以根据算子的不同配置和调度来预测性能。通过训练数据（实际执行时间），Cost Model学习张量程序的性能估计，以自动适配到对应的硬件。

XGBoost是一种改进的决策树增强算法，具有训练速度快、精确的特征提取和快速预测的优点。使用XGBoost来作为基础预测模型，预测相应的吞吐量。

## 实验设置

硬件设置为课程提供的服务器，设置步骤为：

1. 按照课件中的步骤进行相关的操作，设置相应的训练数据和环境
2. 为了减少不必要的数据搬运，创建相应的软连接

 软件设置：本实验不需要设置

总的命令行执行语句为：

```shell
mkdir /home/user124/scripts
cd scripts
find /home/ubuntu/tenset/scripts -type f -exec ln -fs {} /home/user124/scripts/ \;
find /home/ubuntu/tenset/scripts -type d -exec ln -fs {} /home/user124/scripts/ \;
rm /home/user124/scripts/train_model.py
cp /home/ubuntu/tenset/scripts/train_model.py /home/user124/scripts
ln -s /home/ubuntu/tenset/scripts/dataset /home/user124/scripts
```

## 实验过程（步骤、方法和中间结果输出）

- 登录服务器，修改密码，进入工作目录

- 拷贝数据集和脚本到自己的目录下。为避免不必要的数据移动，实际过程中为创建了到目标文件的软链接

- 运行`train_model.py`脚本，训练Cost Model，并输出验证集中的调度伪代码、成本模型对程序吞吐率的预测、调优加速比和均方根误差RMSE

  运行的命令行语句为：`python train_model.py --number=124`

### 实验的输入为

 本次实验中不需要额外的输入，只需要能够运行相应的模型训练文件即可。

相应的命令行执行语句为：`python train_model.py --number=124`

### 得到的实验输出为

 得到的实验输出为训练后的相应数据，为训练得到的模型在测试集上的测试效果。

```
=========== Valid ============
RMSE: 0.0912313940187328
 PREDS: [0.56446628 0.20632023 0.18618622 0.19983302 0.28089275 0.29460123
 0.19431119 0.16374024 0.32930012 0.20408858]
xgb {'RMSE': '0.091231', 'R^2': '0.625968', 'pairwise comparision accuracy': '0.802964', 'mape': '10272841837.321886', 'average peak score@1': '0.804566', 'average peak score@5': '0.854054'}
```

### 计算截图为

得到的计算截图为：

![实验一截图](https://pigkiller-011955-1319328397.cos.ap-beijing.myqcloud.com/img/202412131739005.gif)

## 实验总结

 在本实验中，完成了对Cost-Model的运行，了解了整体的实验思路。

也带来了对实验的思考：如何提高模型的预测效果，怎么寻找一组最好的参数，能否使用如强化学习等算法来优化最优参数的寻找过程等。

# 实验二

1、分析数据特征，调整XGBoost模型参数并重新训练Cost Model。使用验证集验证精度，并使用测试集给出性能预测结果。

2、在提交平台上，上传10组性能预测结果，通过自动评分程序处理，得到RMSE的值。

## 实验目的

1、熟悉决策树模型的训练过程。

2、掌握深度学习模型的调参和模型评价方法。 

## 实验原理

本实验在给定的默认参数基础上，实现了一个**基于ε-greedy策略强化学习模型**，在参数空间内进行搜索，实现了对最优参数的高效搜索。

本次实验中调节的参数有：

1. 决策树的最大深度max_depth
2. 节点分裂的最小损失减少值gamma
3. 子节点所需的最小样本权重和min_child_weight
4. 学习率eta

### 强化学习建模

因为本实验的任务设置并不复杂，不需要额外的强化学习环境设置，故可以设置为：

- 状态空间：当前的参数配置组合
- 动作空间：参数的调整方式
  - 局部搜索概率为0.3
  - 局部搜索半径为0.2
- 奖励信号：-RMSE

在参数空间的探索过程中，为了避免无用的消耗，根据贝叶斯

### 强化学习搜索优化

为了减少无意义的搜索，主要进行了一下几种优化：

1. 贝叶斯优化
2. 参数重要性
3. 自适应局部搜索
4. 搜索早停机制

#### 贝叶斯优化

使用高斯过程对参数空间进行建模，预测未探索区域的性能。

主要的有点在于：可以利用历史搜索经验，减少对已经探索过的参数的重复搜索，以此能够平衡探索与利用。比最初的比随机搜索更有效率。

代码实现为：

```py
def get_next_params(self):
    if len(self.history_params) < 10:
        return self._get_random_params()
    # 使用高斯过程回归
    X = np.vstack([self._params_to_vector(p) for p in self.history_params])
    y = np.array(self.history_scores)
    self.gp.fit(X, y)
    # 使用期望改进(EI)作为采集函数
    n_candidates = 1000
    X_candidates = np.random.uniform(0, 1, size=(n_candidates, 4))
    ei = self._acquisition_function(X_candidates)
    
    best_candidate_idx = np.argmax(ei)
    return self._vector_to_params(X_candidates[best_candidate_idx])
```

#### 参数重要性设置

在学习过程中，并不是所有的参数都是具有相同的重要性的。通过为参数设置相应的重要性权重，可以对重要的参数进行更多搜索，以达成更好的搜索效果，减少在不重要参数上的搜索时间，以提高搜索效率。

并且，在函数运行过程中支持**动态对参数重要性进行调节**，以达成更好的搜索效果。

对参数的重要性初始化设置为：

```py
# 参数重要性
self.param_importance = {
    'max_depth': 1.0,
    'min_child_weight': 1.0,
    'gamma': 1.0,
    'eta': 1.0
}
```

相应的动态权重调节搜索代码为：

```py
def update_param_importance(self):
    """更新参数重要性权重"""
    if len(self.history_params) < 10:
        return
        
    X = np.vstack([self._params_to_vector(p) for p in self.history_params])
    y = np.array(self.history_scores)
    
    # 计算参数与性能的相关性
    correlations = np.abs(np.corrcoef(X.T, y)[-1, :-1])
    total = np.sum(correlations)
    
    # 更新每个参数的重要性权重
    for i, param in enumerate(['max_depth', 'min_child_weight', 'gamma', 'eta']):
        self.param_importance[param] = correlations[i] / total if total > 0 else 0.25
```

#### 自适应局部搜索机制

自适应局部搜索机制能够根据历史探索结果自动调整搜索方向，平衡了探索与利用的关系，相比网格搜索更高效，可以在有限步数内找到较好的参数组合。

但是这样确实也存在着一些问题，如可能陷入局部最优解的情况，导致对参数空间探索不够充分。需要在此基础上引入一定的额外随机性进行参数空间的探索。

相应的代码为：

```py
def local_search(self, base_params):
    """在最优参数附近进行局部搜索"""
    new_params = {}
    # 搜索半径随训练进度逐渐减小
    radius = self.local_search_radius * (1 - len(self.history_params) / 1000)
    
    for param, value in base_params.items():
        # 离散参数使用整数调整
        if param in ['max_depth', 'min_child_weight']:
            delta = random.choice([-1, 0, 1])
            new_value = max(min(value + delta, 
                              self.dynamic_ranges[param][1]), 
                          self.dynamic_ranges[param][0])
            new_params[param] = int(new_value)
        # 连续参数使用比例调整
        else:
            scale = self.param_importance[param]
            delta = random.uniform(-radius, radius) * scale * value
            new_value = max(min(value + delta, 
                              self.dynamic_ranges[param][1]), 
                          self.dynamic_ranges[param][0])
            new_params[param] = new_value
    
    return new_params
```

#### 早停机制

早停机制避免了无效的参数探索，在多轮无明显提高，即收敛后自动停止运行，减少不必要的实验探索。

```python
def train(self, dataset, max_episodes=1000):
    for episode in range(max_episodes):
        # ... 训练代码 ...
        
        # 早停检查
        if current_rmse < self.best_rmse:
            self.patience_counter = 0
        else:
            if abs(current_rmse - self.best_rmse_patience) < self.min_delta:
                self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            print(f"\nEarly stopping triggered after {episode + 1} episodes")
            break
```

## 实验设置

### 硬件设置

硬件设置和实验一相同，均为使用课程组提供的服务器，无需进行其他修改。

### 软件设置

对于软件设置，为了支持强化学习相关搜索过程中的贝叶斯优化，需要安装python软件包scikit-learn。

对强化学习过程封装了一个类`XGBParamsOptimizer`，支持调用`train_model`中实现的`train_zero_shot`函数作为反馈，实现强化学习的训练效果。具体的软件更改可以见附件中的`train_model_rl.py`文件。以下介绍几个核心方法：

- `get_next_params`：决定下一组要尝试的参数组合，整合贝叶斯优化、局部搜索和随机探索
- `local_search`：在当前最优参数附近进行局部搜索，考虑参数重要性和搜索半径
- `update_param_importance`：基于历史数据更新各参数的重要性权重
- `update_dynamic_ranges`：根据历史表现动态调整参数的搜索范围
- `train`：执行完整的参数优化过程，包括参数选择、模型训练、结果评估和早停检查
- `check_early_stopping`：检查是否满足早停条件

## 实验过程（步骤、方法和中间结果输出）

总的实验步骤分为如下几步：

1. 对原始的实验代码进行修改，支持更好的函数化封装，是得能够在强化学期的每一轮训练epoch中进行良好封装。
2. 创建新的代码文件train_model_rl.py，对强化学习进行了封装，支持在参数空间内搜索最优参数
3. 设置搜索轮数epoch为100，对实验过程进行搜索

实验中设定的初始参数和实验一相同，并没有做额外的修改

## 实验结果（输入、输出、数据记录和计算截图）

 实验结果为：

![result](https://pigkiller-011955-1319328397.cos.ap-beijing.myqcloud.com/img/202412131818210.png)

 在实验过程中进行了epoch=100轮参数搜索，最后搜索的参数为：

```
max_depth: 10
min_child_weight: 1
gamma: 0.055447838467443264
eta: 0.1645789471134122
```

搜索得到的最小RMSE为0.08301699592638039

## 实验总结

本实验提出的强化学习方法能够有效地自动优化XGBoost模型的超参数，能对参数空间进行有效搜索，可以可以找到使RMSE最小的一组参数设置，具有良好的效果。
