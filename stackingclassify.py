from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator,ClassifierMixin,TransformerMixin,clone
import numpy as np
from sklearn import datasets
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
#分类问题用ClassifierMixin
#data
import matplotlib,pylab as plt
iris = datasets.load_iris()
x = iris.data
y = iris.target
plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)
plt.show()
#model
model_gnb = GaussianNB()  # 建立朴素贝叶斯分类
model_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 5)) # 建立MLP
model_rfc = RandomForestClassifier(n_estimators=10)
model_svc = SVC()  # 建立支持向量机模型
model_gbc = GradientBoostingClassifier()  # 建立梯度增强分类模型对象
base_model_names = ['GaussianNB', 'MLPClassifier', 'RandomForestClassifier', 'SVC', 'GradientBoostingClassifier']  # 不同模型的名称列表
base_model = [model_gnb, model_mlp, model_rfc, model_svc, model_gbc]  # 不同回归模型对象的集合

meta_model = XGBClassifier()

class StackingAverageModels(BaseEstimator,ClassifierMixin,TransformerMixin):
    def __init__(self,base_models, meta_model, n_folds=5):
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
features = STACK.fit(x,y).predict(x)
#print(features)

from sklearn.metrics import accuracy_score,auc,confusion_matrix,classification_report
#roc_auc_score
model_metrics_name = [accuracy_score, auc]  # 分类评估指标对象集
model_metrics_list = []  # 分类估指标列表
tmp_list = []  # 每个内循环的临时结果列表
for m in model_metrics_name:  # 循环每个指标对象
    tmp_score = m(y, features)  # 计算每个分类指标结果
    tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
model_metrics_list.append(tmp_list)  # 将结果存入分类评估指标列表
import pandas as pd
df = pd.DataFrame(model_metrics_list, columns=['acc','auc'])  # 建立回归指标的数据框

import matplotlib.pylab as plt
plt.scatter(x[:, 0], x[:, 1], marker='o', c=features)
plt.show()
print(df)
print(confusion_matrix(features,y))
print(classification_report(y,features))