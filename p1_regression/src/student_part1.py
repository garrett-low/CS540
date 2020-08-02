import numpy as np 
import matplotlib.pyplot as plt

def data_loader(file):
    a = np.genfromtxt(file, delimiter=',', skip_header=0)
    x = a[:, 1:] / 255.0
    y = a[:, 0]
    return (x, y)

x_train, y_train = data_loader('mnist_train.csv')
x_test, y_test = data_loader('mnist_test.csv')

print('data loading done')

test_labels = [7,4]
indices = np.where(np.isin(y_train,test_labels))[0]
indices_t = np.where(np.isin(y_test, test_labels))[0]

x = x_train[indices]
y = y_train[indices]
x_t = x_test[indices_t]
y_t = y_test[indices_t]


y[y == test_labels[0]] = 0
y[y == test_labels[1]] = 1
y_t[y_t==test_labels[0]] = 0
y_t[y_t == test_labels[1]] = 1



# Todo: you may need to change some hyper-paramter like num_epochs and alpha, etc
num_epochs = 10
m = x.shape[1]
n = x.shape[0]
alpha = 0.1

large_num = 1e8
epsilon = 1e-6
thresh = 1e-4

w = np.random.rand(m)
b = np.random.rand()

c = np.zeros(num_epochs)


for epoch in range(num_epochs):
    a = 1 / (1 + np.exp(-(np.matmul(w, np.transpose(x)) + b)))
    
    w -= alpha * np.matmul(a-y, x)
    
    b -= alpha * (a-y).sum()
    
    cost = np.zeros(len(y))
    idx = (y==0) & (a > 1 - thresh) | (y == 1) & (a < thresh)
    cost[idx] = large_num
    
    a[a<thresh] = thresh
    a[a> 1-thresh] = thresh
    
    inv_idx = np.invert(idx)
    cost[inv_idx] = - y[inv_idx] * np.log(a[inv_idx]) - (1-y[inv_idx]) * np.log(1-a[inv_idx])
    c[epoch] = cost.sum()
    
    if epoch % 3 == 0:
        print('epoch = ', epoch + 1, 'cost = ', c[epoch])
    
    if epoch > 0 and abs(c[epoch -1] - c[epoch]) < epsilon:
        break
    


# Todo: new test
new_test = np.loadtxt('test.txt', delimiter=',')
new_x = new_test / 255.0


    