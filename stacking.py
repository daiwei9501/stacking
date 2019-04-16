from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator,RegressorMixin,TransformerMixin,clone
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
#分类问题用ClassifierMixin
#data
X,y = make_regression(n_samples=10000, n_features =27, noise =10)
#model
model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
model_lr = LinearRegression()  # 建立普通线性回归模型对象
model_etc = ElasticNet()  # 建立弹性网络回归模型对象
model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
base_model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR']  # 不同模型的名称列表
base_model = [model_br, model_lr, model_etc, model_svr, model_gbr]  # 不同回归模型对象的集合

meta_model = XGBRegressor()

class StackingAverageModels(BaseEstimator,RegressorMixin,TransformerMixin):
    def __init__(self,base_models,meta_model,n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    #将原来的模型clone出来，并fit
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=328)

        #对每个模型，使用交叉验证来训练初级学习器，得到次级训练集
        out_of_fold_pre = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X[train_index], y[train_index])
                self.base_models_[i].append(instance)
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_pre[holdout_index,i] = y_pred
        #使用次级训练集来训练次级学习器
        self.meta_model_.fit(out_of_fold_pre,y)
        return self
    #再fit方法中，已经保存了初级学习器和次级学习器，需利用predict
    def predict(self,X):
        meta_featues = np.column_stack([np.column_stack([model.predict(X) for model in base_models]).mean(axis =1) for base_models in self.base_models_])
        return self.meta_model_.predict(meta_featues)


STACK = StackingAverageModels(base_model,meta_model)
features = STACK.fit(X,y).predict(X)
print(features)

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
tmp_list = []  # 每个内循环的临时结果列表
for m in model_metrics_name:  # 循环每个指标对象
    tmp_score = m(y, features)  # 计算每个回归指标结果
    tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
import pandas as pd
df = pd.DataFrame(model_metrics_list, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框

import matplotlib.pylab as plt
plt.scatter(x=features,y=y)
plt.show()
print(df)