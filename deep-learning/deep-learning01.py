# coding:utf8
import numpy as np


# 定义激活函数
def sigmoid(x):
    return 1.0/(1+np.exp(-x))


#定义网络
def layer_sizes(X, Y):  
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)


#初始化参数w/b
def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x)* 0.01
    b1 = np.zeros(n_h ,1)
    w2 = np.random.randn(n_y, n_h)* 0.01
    b2 = np.zeros((n_y, 1))

    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {
        'w1' : w1,
        'b1' : b1,
        'w2' : w2,
        'b2' : b2
    }
    return parameters


#前向传播
def forward_propagation(x, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    
    Z1 = np.dot(w1, x) + b1  #Z1 激活函数之前的线性表达式结果
    A1 = np.tanh(Z1)         #A1 Z1经过激活函数之后的结果
    Z2 = np.dot(w2, Z1) + b2 
    A2 = sigmoid(Z2)
    assert(A2.shape == (1,X.shape[1]))
    
    cache = {'Z1': Z1,
            'A1' : A1,
            'Z2' : Z2,
            'A2' : A2}
    return A2, cache


#定义损失函数
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    
    log_probs = np.multiply(np.log(A2) ,Y) + np.multiply(np.log(1 - A2),1 - Y)

    cost = -1/m * np.sum(log_probs) #平均损失
    
    cost = np.squeeze(cost) #从数组的形状中删除单维条目，即把shape中为1的维度去掉
    
    assert(isinstance(cost,float))
    
    return cost


# 反向传播
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    w1 = parameters['w1']
    w2 = parameters['w2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(w2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {'dW1': dW1,
             'db1': db1,
             'dW2': dW2,
             'db2': db2}
    return grads


# 更新参数
def update_parameters(parameters, grads, learning_rate=1.2):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    parameters = {'w1': w1,
                  'b1': b1,
                  'w2': w2,
                  'b2': b2}
    return parameters


# 模型训练
def my_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=1.2)
        if print_cost and i % 1000 == 0:
            print('Cost after iteration %i: %f' % (i, cost))

    return parameters

