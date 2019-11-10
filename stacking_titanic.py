#coding:utf8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc   
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from mlxtend.classifier import StackingClassifier
%matplotlib inline
import re
import warnings
warnings.simplefilter('ignore')


def get_stacking(clf, x_train, y_train, x_test, n_folds=5):
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)

        second_level_train_set[test_index] = clf.predict_proba(x_tst)[:,1]
        test_nfolds_sets[:,i] = clf.predict_proba(x_test)[:,1]


    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set


# 导入数据
train_set = pd.read_csv('./train.csv')
test_set = pd.read_csv('./test.csv')

# 特征含义
# PassengerId 乘客编号
# Survived 是否幸存
# Pclass 船票等级
# Name 乘客姓名
# Sex 乘客性别
# Age 乘客年龄
# SibSp 兄弟姐妹/配偶数量
# Parch 父母/子女数量
# Ticket 船票号码
# Fare 船票价格
# Cabin 船舱
# Embarked 登录港口

        
# print(train_set.info())
# print('*'*40)
# print(test_set.info())
# print('\n')
# 训练集测试集合并
train_test = pd.concat([train_set,test_set],axis=0)
# print(train_test.info())


# print(train_test['Survived'].value_counts())

# print(train_test.describe().T)

# train_test_corr = train_test.corr()
# plt.subplots(figsize=(12,7))
# sns.heatmap(train_test_corr,vmin=-1,annot=True,square=True)
# plt.show()
# print('\n')

train_test['Embarked'].value_counts()
train_test['Pclass'].value_counts()
train_test['Embarked'].fillna('S',inplace=True)

#票价与pclass和Embarked有关
train_test.groupby(['Pclass','Embarked']).Fare.mean()
train_test['Fare'].fillna(14.435422,inplace=True)

# 缺失值填充
train_test['Age'].fillna(train_test['Age'].median(),inplace=True)


# 特征工程
train_test['SibSp_Parch'] = train_test['Parch'] + train_test['SibSp']

#从名字中提取出称呼
train_test['Name_new'] = train_test['Name'].str.extract('.+,(.+)',expand=False).str.extract('^(.+?)\.',expand=False).str.strip()
# print(train_test['Name_new'].unique())
# print('\n')

#将姓名分类处理()
train_test['Name_new'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer' , inplace = True)
train_test['Name_new'].replace(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty' , inplace = True)
train_test['Name_new'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs',inplace=True)
train_test['Name_new'].replace(['Mlle', 'Miss'], 'Miss',inplace=True)
train_test['Name_new'].replace(['Mr'], 'Mr' , inplace = True)
train_test['Name_new'].replace(['Master'], 'Master' , inplace = True)
# print(train_test['Name_new'].unique())
# print('\n')

# 分类变量数值化
train_test['Name_new'] = train_test['Name_new'].map({'Mr':0,'Mrs':1,'Miss':2,'Master':3,'Royalty':4,'Officer':5}).astype(int)
train_test['Sex'] = train_test['Sex'].map({'female':1,'male':0}).astype(int)
train_test['Embarked'] = train_test['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)


#将年龄划分阶段
train_test['Age']=pd.cut(train_test['Age'],bins=[0,18,30,40,50,100],labels=[1,2,3,4,5])
train_test['Age'] = train_test['Age'].astype('float64')

# 剔除不需要的特征
train_test.drop(['PassengerId','Ticket','Name','Cabin'],axis=1,inplace=True)
# print(train_test.info())

#特征工程完成，划分数据集
train_data=train_test[:891]
test_data=train_test[891:]
train_data_X=train_data.drop(['Survived'],axis=1)
train_data_Y=train_data['Survived']
test_data_X=test_data.drop(['Survived'],axis=1)
test_data_Y=test_data['Survived']


# 特征选择
# 随机森林评估特征重要性
feat_labels=train_data_X.columns
forest=RandomForestClassifier(n_estimators=10,n_jobs=-1,random_state=2019) 
forest.fit(train_data_X,train_data_Y)
importances=forest.feature_importances_
indices=np.argsort(importances)[::-1]  #排序取反
var_list = []
for f in range(6):
    print ("%2d) %-*s %f" % (f+1,6,feat_labels[f],importances[indices[f]]) )
    var_list.append(feat_labels[f])
print('\n')
print(var_list)
print('\n')
    
    
    
# 基分类器
lr_model = LogisticRegression(random_state=2019)       #逻辑回归
dt_model = DecisionTreeClassifier(random_state=2019)   #决策树
rf_model = RandomForestClassifier(random_state=2019)    #随机森林

lr_model.fit(train_data_X[var_list],train_data_Y)
dt_model.fit(train_data_X[var_list],train_data_Y)
rf_model.fit(train_data_X[var_list],train_data_Y)


#模型融合
x_train_arr = np.array(train_data_X[var_list])
y_train_arr = np.array(train_data_Y)
x_test_arr = np.array(test_data_X[var_list])

train_sets = []
test_sets = []
for clf in [lr_model,dt_model,rf_model]:
    train_set, test_set = get_stacking(clf, x_train_arr, y_train_arr, x_test_arr)
    train_sets.append(train_set)
    test_sets.append(test_set)
#     print(clf)


meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis = 1)  #np.concatenate,axis=1列连接
meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis = 1)

# #各模型预测结果相关性
# m_train=pd.DataFrame(meta_train,columns=['lr','dt','rf'])
# m_train.astype(float).corr() 


#使用lr作为我们的次级分类器
meta_model = LogisticRegression(random_state = 2019)
meta_model.fit(meta_train, y_train_arr)
prediction = meta_model.predict_proba(meta_test)
#print(prediction)


false_positive_rate, recall, thresholds = roc_curve(test_data_Y, prediction[:, 1])
final_auc = auc(false_positive_rate,recall)
print(final_auc)