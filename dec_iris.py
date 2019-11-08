# coding:utf8
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
# 导入数据集
iris = datasets.load_iris()
iris_feature = iris.data
iris_target = iris.target
iris_data = pd.DataFrame(iris_feature,columns=iris.feature_names)
iris_data['target'] = iris_target

# 数据分布
print(iris_data.info())
print('\n')
print(iris_data['target'].unique())
# 特征相关性
iris_data_corr = iris_data.corr()
plt.subplots(figsize=(10,7))
sns.heatmap(iris_data_corr,vmin=1,annot=True,square=True)
plt.show()

# 划分数据集
x_train,x_test,y_train,y_test = train_test_split(iris_data.iloc[:,:-1],iris_data.iloc[:,-1:],test_size=0.3,random_state=2019)

clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_pred,y_test)
print(accuracy)
