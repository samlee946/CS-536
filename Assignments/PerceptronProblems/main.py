import numpy as np
import matplotlib.pyplot as plt

def perceptron(X, Y, max_epoch = 1000):
    m, k = X.shape
    w = np.zeros((k, 1))
    X[:, 0] = np.ones(m)
    steps = 0
    for epoch in range(max_epoch):
        flag = False
        for i in range(m):
            f = 1 if np.matmul(X[i, :], w) > 0 else -1
            #print(i, Y[i], f)
            if f != Y[i]:
                steps += 1
                flag = True
                w = w + (Y[i] * X[i]).reshape(w.shape)
                #print(w)
        if flag == False:
            break
    gamma = 1e9
    w_norm = np.linalg.norm(w)
    #gamma_t = 1e9
    #tw = np.zeros((k, 1))
    #tw[k - 1] = 1
    #tw_norm = np.linalg.norm(tw)
    for i in range(m):
        #print(i, Y[i], np.matmul(X[i, :], tw))
        gamma = min(gamma, np.abs(np.matmul(X[i, :], w)) / w_norm)
        #gamma_t = min(gamma_t, np.abs(np.matmul(X[i, :], tw)) / tw_norm)
    #print(tw, gamma, gamma_t)
    return w, steps

def get_data(m, k, epsilon, load = False, save = False):
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
    if save == True:
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

def question3(save = False, save_file = 'q3.png'):
    eps = np.arange(0, 1.04, 0.05)
    repeat_times = 100
    step_history = np.zeros(len(eps))
    for i in range(len(eps)):
        for j in range(repeat_times):
            X, Y = get_data(100, 20, eps[i])
            w, steps = perceptron(X, Y)
            step_history[i] += steps
    for i in range(len(eps)):
        step_history[i] /= 1.0 * repeat_times
    plt.xlabel('epsilon')
    plt.ylabel('avg. steps')
    plt.plot(eps, step_history, '--bo')
    if save == False:
        plt.show()
    else:
        plt.savefig(save_file)

def question4(m, save = False, save_file = 'q4.png'):
    ks = np.arange(2, 41, 1)
    repeat_times = 100
    step_history = np.zeros(len(ks))
    for i in range(len(ks)):
        print(i)
        for j in range(repeat_times):
            X, Y = get_data(m, ks[i], 1)
            w, steps = perceptron(X, Y)
            step_history[i] += steps
    for i in range(len(ks)):
        step_history[i] /= 1.0 * repeat_times
    plt.xlabel('k')
    plt.ylabel('avg. steps')
    plt.plot(ks, step_history, '--bo')
    if save == False:
        plt.show()
    else:
        plt.savefig('q4-' + repr(m) + '.png')

if __name__ == '__main__':
    #question3(save = True)
    question4(1000, save = True)
