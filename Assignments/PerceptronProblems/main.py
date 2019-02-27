import numpy as np
import matplotlib.pyplot as plt

def perceptron(X, Y):
    m, k = X.shape
    w = np.zeros((k, 1))
    X[:, 0] = np.ones(m)
    idx = list(range(m))
    while len(idx) > 0:
        i = idx.pop()
        f = 1 if np.matmul(X[i, :], w) > 0 else -1
        #print(i, Y[i], f)
        if f != Y[i]:
            w = w + (Y[i] * X[i]).reshape(w.shape)
            #print(w)
            idx.append(i)
    print(w)
    return w

def get_data(m, k, epsilon, load = False):
    if load == True:
        X, Y = load_arr()
        X = X.reshape((m, k + 1))
        Y = Y.reshape((m, 1))
    else:
        X = np.zeros((m, k + 1))
        Y = np.zeros((m, 1))
        for i in range(m):
            X[i, 1:k] = np.random.normal(0, 1, k - 1)
            D = np.random.exponential(1)
            X[i, k] = (epsilon + D) if np.random.rand() >= 0.5 else -(epsilon + D)
            Y[i] = 1 if X[i, k] > 0 else -1
        save_arr(np.concatenate((X, Y), axis = 1))
    #print(X, Y)
    return X, Y

def plot_data(X):
    plt.xlabel('X')

def save_arr(arr, filename = 'data.npy'):
    np.save(filename, arr)
    print('Saving arr to ' + filename)

def load_arr(filename = 'data.npy'):
    arr = np.load(filename)
    return arr[:, :-1], arr[:, -1]

def question2():
    X, Y = get_data(100, 20, 1, load = True)
    perceptron(X, Y)

if __name__ == '__main__':
    question2()
