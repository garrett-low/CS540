import numpy as np 
import matplotlib.pyplot as plt


# Todo: you need to change the activation function from relu (current) version to logistic, remember, not only the activation function, but the weight update part as well.

def data_loader(file):
    a = np.genfromtxt(file, delimiter=',', skip_header=0)
    x = a[:, 1:] / 255.0
    y = a[:, 0]
    return (x, y)


x_train, y_train = data_loader('mnist_train.csv')
x_test, y_test = data_loader('mnist_test.csv')

test_labels = [7,4]
indices = np.where(np.isin(y_train,test_labels))[0]
indices_t = np.where(np.isin(y_test, test_labels))[0]

x = x_train[indices]
y = y_train[indices]
x_t = x_test[indices_t]
y_t = y_test[indices_t]


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
num_hidden_uints = 392


def relu(x):
    y = x
    y[y<0] = 0
    return y

def diff_relu(x):
    y = x
    y[x>0] = 1
    y[x<=0] = 0
    return y


def nnet(train_x, train_y, test_x, test_y, lr, num_epochs):
    num_train = len(train_y)
    num_test = len(test_y)
    
    train_x = np.hstack((train_x, np.ones(num_train).reshape(-1,1)))
    test_x = np.hstack((test_x, np.ones(num_test).reshape(-1,1)))
    
    num_input_uints = train_x.shape[1]  # 785 
    
    
    wih = np.random.uniform(low=-1, high=1, size=(num_hidden_uints, num_input_uints)) #392*785
    who = np.random.uniform(low=-1, high=1, size=(1, num_hidden_uints+1)) # 1 * 393
    
    for epoch in range(1, num_epochs+1):
        out_o = np.zeros(num_train)
        out_h = np.zeros((num_train, num_hidden_uints+1))  #num_train * 393
        out_h[:,-1] = 1
        for ind in range(num_train):
            row = train_x[ind]  # len = 785 
            out_h[ind, :-1] = relu(np.matmul(wih, row))
            out_o[ind] = 1 / (1 + np.exp(-sum(out_h[ind] @ who.T)))

            delta = np.multiply(diff_relu(out_h[ind]), (train_y[ind] - out_o[ind]) * np.squeeze(who))
            wih += lr * np.matmul(np.expand_dims(delta[:-1], axis=1), np.expand_dims(row,axis=0))
            who += np.expand_dims(lr * (train_y[ind] - out_o[ind]) * out_h[ind,:], axis=0)
        error = sum(- train_y * np.log(out_o) - (1-train_y) * np.log(1-out_o))
        num_correct = sum((out_o > 0.5).astype(int) == train_y)
        
        print('epoch = ', epoch, ' error = {:.7}'.format(error), 'correctly classified = {:.4%}'.format(num_correct / num_train))
    
    return wih.T, who




# Todo: change these hyper parameters
lr = 0.1
num_epochs = 3

W1, W2 = nnet(x, y, x_t, y_t, lr, num_epochs)



# Todo: new test

new_test = np.loadtxt('test.txt', delimiter=',')
new_x = new_test / 255.0

    
