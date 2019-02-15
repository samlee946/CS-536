import numpy as np

eps = 1e-5
def cross_entropy(P, P_y0):
    return -P * np.log(P) - P_y0 * np.log(P_y0)

def IC(X, Y):
    m = X.shape[0]
    c_00 = 0
    c_01 = 0
    c_10 = 0
    c_11 = 0
    for x, y in zip(X, Y):
        if x < eps:
            if y < eps:
                c_00 += 1
            else:
                c_01 += 1
        else:
            if y < eps:
                c_10 += 1
            else:
                c_11 += 1
    P_x = np.count_nonzero(X) / m
    P_00 = c_00 / (c_00 + c_01)
    P_01 = c_01 / (c_00 + c_01)
    P_10 = c_10 / (c_10 + c_11)
    P_11 = c_11 / (c_10 + c_11)
    print(X, P_x)
    print(c_00, c_01, c_10, c_11)
    print(P_00, P_01, P_10, P_11)
    return P_x * cross_entropy(P_11, P_10) + (1 - P_x) * cross_entropy(P_01, P_00)

def fit_decision_tree(X, Y):
    m, k = X.shape
    P_y = np.count_nonzero(Y) / m
    H_y = cross_entropy(P_y, 1 - P_y) #-P_y * np.log(P_y) - (1 - P_y) * np.log((1 - P_y))
    #print(X, Y)
    print(P_y)
    print(H_y)
    for i in range(k):
        print(IC(X[:, i], Y))

def get_data(k, m, save = False):
    # input args:
    # k: number of features
    # m: number of data points
    X = np.zeros((m, k))
    Y = np.zeros((m, 1))
    w = np.zeros(k)
    denom = 10 * (1 - 0.9 ** k)
    s = 0
    for j in range(m):
        if np.random.rand() >= 0.5:
            X[j][0] = 1
        for i in range(1, k):
            if np.random.rand() >= 0.75:
                X[j][i] = X[j][i - 1]
            else:
                X[j][i] = 1 - X[j][i - 1]
            w[i] = 0.9 ** i / denom
            s += w[i] * X[j][i]
        if s >= 0.5:
            Y[j] = X[j][0]
        else:
            Y[j] = 1 - X[j][0]
    if save == True:
        save_arr(np.concatenate((X, Y), axis = 1))
    return X, Y

def save_arr(arr, filename = 'arr.npy'):
    np.save(filename, arr)
    print('Saving arr to ' + filename)

def load_arr(filename = 'arr.npy'):
    arr = np.load(filename)
    return arr[:, :-1], arr[:, -1]

if __name__ == '__main__':
    k = 4
    m = 30
    #X, Y = get_data(k, m)
    X, Y = load_arr()
    X_train = np.copy(X)
    Y_train = np.copy(Y)
    fit_decision_tree(X_train, Y_train)
