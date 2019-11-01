#coding:utf8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
%matplotlib inline
import re
import warnings
warnings.simplefilter('ignore')

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


print(train_set.info())
print('*'*40)
print(test_set.info())
print('\n')
# 训练集测试集合并
train_test = pd.concat([train_set,test_set],axis=0)
print(train_test.info())


print(train_test['Survived'].value_counts())

print(train_test.describe().T)

train_test_corr = train_test.corr()
plt.subplots(figsize=(12,7))
sns.heatmap(train_test_corr,vmin=-1,annot=True,square=True)
plt.show()
print('\n')

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
print(train_test['Name_new'].unique())
print('\n')

#将姓名分类处理()
train_test['Name_new'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer' , inplace = True)
train_test['Name_new'].replace(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty' , inplace = True)
train_test['Name_new'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs',inplace=True)
train_test['Name_new'].replace(['Mlle', 'Miss'], 'Miss',inplace=True)
train_test['Name_new'].replace(['Mr'], 'Mr' , inplace = True)
train_test['Name_new'].replace(['Master'], 'Master' , inplace = True)
print(train_test['Name_new'].unique())
print('\n')

# 分类变量数值化
train_test['Name_new'] = train_test['Name_new'].map({'Mr':0,'Mrs':1,'Miss':2,'Master':3,'Royalty':4,'Officer':5}).astype(int)
train_test['Sex'] = train_test['Sex'].map({'female':1,'male':0}).astype(int)
train_test['Embarked'] = train_test['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)


#将年龄划分阶段
train_test['Age']=pd.cut(train_test['Age'],bins=[0,18,30,40,50,100],labels=[1,2,3,4,5])
train_test['Age'] = train_test['Age'].astype('float64')

# 剔除不需要的特征
train_test.drop(['PassengerId','Ticket','Name','Cabin'],axis=1,inplace=True)
print(train_test.info())

#特征工程完成，划分数据集
train_data=train_test[:891]
test_data=train_test[891:]
train_data_X=train_data.drop(['Survived'],axis=1)
train_data_Y=train_data['Survived']
test_data_X=test_data.drop(['Survived'],axis=1)
test_data_Y=test_data['Survived']


LR=LogisticRegression()
LR.fit(train_data_X,train_data_Y)
y_pred = LR.predict(test_data_X)
auc = roc_auc_score(test_data_Y,y_pred)
print(auc)