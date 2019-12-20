#coding:utf8
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import  tensorflow.keras as keras
import warnings
warnings.filterwarnings("ignore")

# 导入数据
faces_data = datasets.fetch_olivetti_faces()


# 显示原始图片
i = 0
plt.figure(figsize=(20, 20))
for img in faces_data.images:
    #总共400张图片，每个人10个头像，共40个人
    plt.subplot(20, 20, i+1)
    plt.imshow(img, cmap="gray")
    #关闭x，y轴显示
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(faces_data.target[i])
    i = i + 1
plt.show()


#人脸数据
X = faces_data.images
#人脸对应的标签
y = faces_data.target
# 400张图片，每张图片64x64，灰色图片通道数为1
X = X.reshape(400, 64, 64, 1)  
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



model = keras.Sequential()
# 第一层卷积，卷积核数为128，卷积核3x3，激活函数使用relu，输入是每张原始图片64x64*1
model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(64, 64, 1)))
# 第一池化层
model.add(keras.layers.MaxPool2D((2, 2), strides=2))

# 第二层卷积，卷积核数为64，卷积核3x3，激活函数使用relu
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
# 第二池化层
model.add(keras.layers.MaxPool2D((2, 2), strides=2))

#把多维数组压缩成一维，里面的操作可以简单理解为reshape，方便后面全连接层使用
model.add(keras.layers.Flatten())

#对应cnn的全连接层，40个人对应40种分类，激活函数使用softmax，进行分类
model.add(keras.layers.Dense(40, activation='softmax'))


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 进行训练和预测
model.fit(X_train, y_train, epochs=10)
y_predict = model.predict(X_test)
# 打印实际标签与预测结果
print(y_test[0], np.argmax(y_predict[0]))