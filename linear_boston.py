import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

# 数据导入
boston = load_boston()
x = boston.data     #自变量
y = boston.target   #因变量
print(boston.feature_names)
print('\n')


# 字段含义
# { 'CRIM': 城镇人均犯罪率，
#   'ZN' : 占地面积超过2.5万平方英尺的住宅用地比例,
#  'INDUS' : 城镇非零售业务地区比例,
#  'CHAS' : 查尔斯河虚拟变量,
#  'NOX' : 一氧化氮浓度,
#  'RM' : 平均每居民房数,   
#  'AGE' : 在1940年之前建成的所有者占用单位的比例, 
#  'DIS' : 与5个波士顿就业中心的加权距离,
#  'RAD' : 辐射性公路可达性指数,
#  'TAX' : 每10美元全额物业税率,
#  'PTRATIO' : 城镇师生比例,
#  'B' : 城镇黑人比例,
#  'LSTAT': 人群中地位较低人群的百分数}

boston_df = pd.DataFrame(boston.data,columns=boston.feature_names)
print(boston_df.head(10))
print('\n')

target = list(boston.target)
boston_df['target'] = target
print(boston_df.head())
print('\n')
print(boston_df.info())
print('\n')
print(boston_df.describe().T)
print('\n')
print(boston_df.corr())
print('\n')

for i in range(0,len(boston_df.columns)-1):
    x = boston_df.iloc[:,i]
    y = boston_df.target
    plt.scatter(x,y)
    plt.title(boston.feature_names[i],fontsize=15)
    plt.show()
    
    
# 剔除房价等于50的异常值
del_index = []
for i in range(len(boston_df)):
    if boston_df['target'][i]==50.0:
        del_index.append(i)
boston_df.drop(del_index,inplace=True)


# 根据相关性可以知道，与房价强相关的变量分别是PTRATIO、RM、LSTAT这三个变量
# 注：相关性绝对值大于0.5表示强相关，正负号表示相关性的正负
# 故其他相关性较弱的变量本次建模暂不考虑，忽略处理
stay_list = ['LSTAT','RM','PTRATIO','target']
del_col = list(set(boston_df.columns)-set(stay_list))
boston_df.drop(del_col,axis=1,inplace=True)
print(boston_df.head(10))

# 划分数据集
x_train,x_test,y_train,y_test = train_test_split(boston_df.iloc[:,:-1],boston_df['target'],test_size=0.3,random_state=2018)

# # 数据归一化
# x_train = np.array(x_train)
# x_test = np.array(x_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
# min_max_scaler = MinMaxScaler()
# x_train_scaler = min_max_scaler.fit_transform(x_train)
# x_test_scaler = min_max_scaler.fit_transform(x_test)
# y_train_scaler = min_max_scaler.fit_transform(y_train.reshape(-1,1))
# y_test_scaler = min_max_scaler.fit_transform(y_test.reshape(-1,1))

# lr = LinearRegression()
# lr.fit(x_train_scaler,y_train_scaler)
# y_pred = lr.predict(x_test_scaler)
# r2_score(y_test_scaler,y_pred)

lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
r2_score(y_test,y_pred)
