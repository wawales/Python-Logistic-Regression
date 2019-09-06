#!/usr/bin/env python
# coding: utf-8

# In[20]:



import sklearn.datasets
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


###################init#################
np.random.seed(0)
m = 1000
X, y = sklearn.datasets.make_moons(m, noise=0.20)
# plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
# y = y.reshape((200, 1))
w = np.random.randn(1,2) * 0.01
b = 0
lr = 0.05




for i in range(1000):
    ##############forward################
    z = np.dot(w, X.T) + b
    a = 1/(1 + np.exp(-z))
    J = -(np.dot(y.T, np.log(a).T) + np.dot((1-y.T),np.log(1-a).T))/m
    if i % 100 == 0:
        print('epoch %d loss:%f'%(i,J))
#         plt.plot(X.T[0], (-w[0][0] * X.T[0] - b)/w[0][1])
    #############backward#################
    dz = -y.T + a
    dw =  np.dot(X.T, dz.T)/m
    db = np.sum(dz)/m
    w -= lr * dw.T
    b -= lr * db
def model(X):
    z = np.dot(w, X.T) + b
    a = 1/(1 + np.exp(-z))
    a[a <= 0.5] = 0
    a[a > 0.5] = 1
    return a
    
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plot_decision_boundary(lambda x: model(x))

