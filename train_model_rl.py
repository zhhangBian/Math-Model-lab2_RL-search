import numpy as np
from train_model import train_zero_shot
import pickle
import random
import logging
import argparse
import torch
import os 
import multiprocessing
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from common import load_and_register_tasks, str2bool
import tvm
from tvm import auto_scheduler

from tvm.auto_scheduler.utils import to_str_round
from tvm.auto_scheduler.cost_model import RandomModelInternal

from common import load_and_register_tasks, str2bool

from tvm.auto_scheduler.measure import recover_measure_input

from tvm.auto_scheduler.dataset import Dataset, LearningTask
from tvm.auto_scheduler.cost_model.xgb_model import XGBModelInternal
from tvm.auto_scheduler.cost_model.mlp_model import MLPModelInternal
from tvm.auto_scheduler.cost_model.lgbm_model import LGBModelInternal
from tvm.auto_scheduler.cost_model.tabnet_model import TabNetModelInternal
from tvm.auto_scheduler.cost_model.metric import (
    metric_rmse,
    metric_r_squared,
    metric_pairwise_comp_accuracy,
    metric_top_k_recall,
    metric_peak_score,
    metric_mape,
    random_mix,
)

class XGBParamsOptimizer:
    def __init__(self, prog_inputs, prog_results):
        # 定义初始参数搜索范围
        self.param_ranges = {
            'max_depth': (1, 10),           
            'min_child_weight': (1, 4),     
            'gamma': (0.001, 0.1),          
            'eta': (0.01, 0.2)              
        }
        
        # 动态搜索范围
        self.dynamic_ranges = self.param_ranges.copy()
        
        # 初始化最优结果记录
        self.best_params = None
        self.best_rmse = float('inf')
        
        # 历史记录
        self.history_params = []
        self.history_scores = []
        self.best_history = []  # 记录历史最优解的演变
        
        # 参数重要性
        self.param_importance = {
            'max_depth': 1.0,
            'min_child_weight': 1.0,
            'gamma': 1.0,
            'eta': 1.0
        }
        
        # 贝叶斯优化器
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=10,
            random_state=42
        )
        
        # 早停参数
        self.patience = 20  # 增加耐心值
        self.min_delta = 1e-5
        self.patience_counter = 0
        self.best_rmse_patience = float('inf')
        
        self.prog_inputs = prog_inputs
        self.prog_results = prog_results
        
        # 局部搜索参数
        self.local_search_prob = 0.3  # 局部搜索概率
        self.local_search_radius = 0.2  # 初始局部搜索半径
        
    def update_param_importance(self):
        """更新参数重要性"""
        if len(self.history_params) < 10:
            return
            
        X = np.vstack([self._params_to_vector(p) for p in self.history_params])
        y = np.array(self.history_scores)
        
        # 计算每个参数与RMSE的相关性
        correlations = np.abs(np.corrcoef(X.T, y)[-1, :-1])
        total = np.sum(correlations)
        
        # 更新参数重要性
        for i, param in enumerate(['max_depth', 'min_child_weight', 'gamma', 'eta']):
            self.param_importance[param] = correlations[i] / total if total > 0 else 0.25
            
    def update_dynamic_ranges(self):
        """根据历史表现动态调整搜索范围"""
        if len(self.history_params) < 10:
            return
            
        # 获取表现最好的前30%的参数
        top_k = max(3, len(self.history_params) // 3)
        indices = np.argsort(self.history_scores)[:top_k]
        top_params = [self.history_params[i] for i in indices]
        
        # 对每个参数调整范围
        for param in self.param_ranges.keys():
            values = [p[param] for p in top_params]
            min_val = min(values)
            max_val = max(values)
            
            # 扩展范围，但不超过原始范围
            range_width = max_val - min_val
            new_min = max(self.param_ranges[param][0], 
                         min_val - range_width * 0.2)
            new_max = min(self.param_ranges[param][1], 
                         max_val + range_width * 0.2)
            
            self.dynamic_ranges[param] = (new_min, new_max)
    
    def local_search(self, base_params):
        """在最优参数附近进行局部搜索"""
        new_params = {}
        radius = self.local_search_radius * (1 - len(self.history_params) / 1000)  # 随着训练进行逐渐减小搜索半径
        
        for param, value in base_params.items():
            if param in ['max_depth', 'min_child_weight']:
                delta = random.choice([-1, 0, 1])
                new_value = max(min(value + delta, self.dynamic_ranges[param][1]), 
                              self.dynamic_ranges[param][0])
                new_params[param] = int(new_value)
            else:
                # 在参数重要性的影响下进行局部搜索
                scale = self.param_importance[param]
                delta = random.uniform(-radius, radius) * scale * value
                new_value = max(min(value + delta, self.dynamic_ranges[param][1]), 
                              self.dynamic_ranges[param][0])
                new_params[param] = new_value
                
        return new_params

    def _params_to_vector(self, params):
        """将参数字典转换为向量"""
        return np.array([
            (params['max_depth'] - self.param_ranges['max_depth'][0]) / 
            (self.param_ranges['max_depth'][1] - self.param_ranges['max_depth'][0]),
            (params['min_child_weight'] - self.param_ranges['min_child_weight'][0]) / 
            (self.param_ranges['min_child_weight'][1] - self.param_ranges['min_child_weight'][0]),
            (params['gamma'] - self.param_ranges['gamma'][0]) / 
            (self.param_ranges['gamma'][1] - self.param_ranges['gamma'][0]),
            (params['eta'] - self.param_ranges['eta'][0]) / 
            (self.param_ranges['eta'][1] - self.param_ranges['eta'][0])
        ])

    def _vector_to_params(self, vector):
        """将向量转换回参数字典"""
        vector = np.clip(vector, 0, 1)
        return {
            'max_depth': int(round(vector[0] * (self.param_ranges['max_depth'][1] - self.param_ranges['max_depth'][0]) + 
                                 self.param_ranges['max_depth'][0])),
            'min_child_weight': int(round(vector[1] * (self.param_ranges['min_child_weight'][1] - self.param_ranges['min_child_weight'][0]) + 
                                        self.param_ranges['min_child_weight'][0])),
            'gamma': vector[2] * (self.param_ranges['gamma'][1] - self.param_ranges['gamma'][0]) + 
                    self.param_ranges['gamma'][0],
            'eta': vector[3] * (self.param_ranges['eta'][1] - self.param_ranges['eta'][0]) + 
                   self.param_ranges['eta'][0]
        }

    def _acquisition_function(self, X):
        """计算采集函数值（使用期望改进）"""
        mu, sigma = self.gp.predict(X, return_std=True)
        
        with np.errstate(divide='warn'):
            imp = self.best_rmse - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return ei

    def get_next_params(self):
        """获取下一组要尝试的参数"""
        # 更新参数重要性和动态范围
        self.update_param_importance()
        self.update_dynamic_ranges()
        
        # 随机决定是否进行局部搜索
        if self.best_params is not None and random.random() < self.local_search_prob:
            return self.local_search(self.best_params)
            
        if len(self.history_params) < 10:
            return self._get_random_params()
        
        # 使用贝叶斯优化
        X = np.vstack([self._params_to_vector(p) for p in self.history_params])
        y = np.array(self.history_scores)
        
        self.gp.fit(X, y)
        
        # 随机采样候选点，考虑参数重要性
        n_candidates = 1000
        X_candidates = np.random.uniform(0, 1, size=(n_candidates, 4))
        
        # 计算采集函数值
        ei = self._acquisition_function(X_candidates)
        
        # 选择最佳候选点
        best_candidate_idx = np.argmax(ei)
        best_candidate = X_candidates[best_candidate_idx]
        
        return self._vector_to_params(best_candidate)

    def _get_random_params(self):
        """随机生成参数"""
        params = {
            'max_depth': random.randint(*self.param_ranges['max_depth']),
            'min_child_weight': random.randint(*self.param_ranges['min_child_weight']),
            'gamma': random.uniform(*self.param_ranges['gamma']),
            'eta': random.uniform(*self.param_ranges['eta'])
        }
        return params

    def train(self, dataset, max_episodes=1000):
        """使用优化后的训练过程"""
        for episode in range(max_episodes):
            current_params = self.get_next_params()
            
            # 检查是否已经尝试过这组参数
            current_vector = self._params_to_vector(current_params)
            if any(np.allclose(current_vector, self._params_to_vector(p)) 
                   for p in self.history_params):
                continue
            
            eval_results = train_zero_shot(
                dataset=dataset,
                train_ratio=0.9,
                model_names='xgb',
                split_scheme='within_task',
                use_gpu=True,
                number=0,
                max_depth=current_params['max_depth'],
                min_child_weight=current_params['min_child_weight'],
                gamma=current_params['gamma'],
                eta=current_params['eta'],
                prog_inputs=self.prog_inputs,
                prog_results=self.prog_results
            )
            
            current_rmse = eval_results[0]['RMSE']
            
            # 更新历史记录
            self.history_params.append(current_params)
            self.history_scores.append(current_rmse)
            
            # 更新最优结果
            if current_rmse < self.best_rmse:
                self.best_rmse = current_rmse
                self.best_params = current_params
                self.best_history.append((episode, current_params, current_rmse))
                self.patience_counter = 0
                
                # 保存最优模型的参数
                with open(f'best_params_episode_{episode}.pkl', 'wb') as f:
                    pickle.dump(current_params, f)
            else:
                if abs(current_rmse - self.best_rmse_patience) < self.min_delta:
                    self.patience_counter += 1
                else:
                    self.patience_counter = 0
                self.best_rmse_patience = min(current_rmse, self.best_rmse_patience)
            
            print(f"Episode {episode + 1}/{max_episodes}")
            print(f"Current params: {current_params}")
            print(f"Current RMSE: {current_rmse}")
            print(f"Best RMSE so far: {self.best_rmse}")
            print(f"Best params so far: {self.best_params}")
            print(f"Parameter importance: {self.param_importance}")
            print(f"Patience counter: {self.patience_counter}/{self.patience}")
            print("-" * 60)
            
            # 保存训练历史
            history = {
                'history_params': self.history_params,
                'history_scores': self.history_scores,
                'best_history': self.best_history,
                'param_importance': self.param_importance
            }
            with open('training_history.pkl', 'wb') as f:
                pickle.dump(history, f)
            
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {episode + 1} episodes")
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", type=str, default=["dataset.pkl"])
    parser.add_argument("--models", type=str, default="xgb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--number", type=int, default=0)
    parser.add_argument(
        "--split-scheme",
        type=str,
        choices=["by_task", "within_task", "by_target"],
        default="within_task",
    )
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_child_weight", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)

    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--use-gpu", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Whether to use GPU for xgb.")
    args = parser.parse_args()
    print("Arguments: %s" % str(args))
    
    #Allocate GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

    prog_inputs, prog_results = auto_scheduler.RecordReader("dataset/measure_records/e5-2673/ff/([0c9a5ba46ffc5e1a9e5641018527117f,4,14,14,112,1,1,112,672,1,1,1,672,4,14,14,672],llvm).json").read_lines()

    # Setup random seed and logging
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    logging.basicConfig()
    logging.getLogger("auto_scheduler").setLevel(logging.DEBUG)
    
    # 加载数据集
    print("Load all tasks...")
    load_and_register_tasks()

    print("Load dataset...")
    dataset = pickle.load(open(args.dataset[0], "rb"))
    for i in range(1, len(args.dataset)):
        tmp_dataset = pickle.load(open(args.dataset[i], "rb"))
        dataset.update_from_dataset(tmp_dataset)
    
    # 创建优化器并开始训练
    optimizer = XGBParamsOptimizer(prog_inputs, prog_results)
    optimizer.train(dataset, max_episodes=1000)
    
    print("\nTraining completed!")
    print(f"Best parameters found: {optimizer.best_params}")
    print(f"Best RMSE achieved: {optimizer.best_rmse}")

if __name__ == "__main__":
    main() 