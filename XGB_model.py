#coding:utf8
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.simplefilter('ignore')


# 导入数据集
data = load_breast_cancer()
X = pd.DataFrame(data.data)
y = pd.DataFrame(data.target)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2019)
len(X),len(x_train),len(x_test),len(y),len(y_train),len(y_test)


#xgb网格搜索
clf = xgb.XGBClassifier()
xgb_params = {   'n_estimators':[10,20,30], 'max_depth':[2,3,4,5]  }



xgb_grid = GridSearchCV(clf,xgb_params,scoring = 'roc_auc',cv = 5)
xgb_grid.fit(x_train,y_train)
print('xgb最佳参数',xgb_grid.best_params_)
print('-' * 60)

n_estimators = xgb_grid.best_params_['n_estimators']
max_depth = xgb_grid.best_params_['max_depth']


# 最优参数xgb
xgb_model = xgb.XGBClassifier(booster = 'gbtree',objective = 'binary:logistic',  n_estimators = n_estimators, max_depth = max_depth)

xgb_model.fit(x_train,y_train)
xgb_pre = xgb_model.predict_proba(x_test)
xgb_auc = roc_auc_score(y_test,xgb_pre[:,1])
print('xgb模型结果:',xgb_auc)