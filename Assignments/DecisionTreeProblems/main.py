import numpy as np

eps = 1e-5
vis = None

class Node:
    x_id = None
    y = None
    equals_zero = None
    equals_one = None
    def __init__(self, x_id):
        self.x_id = x_id

class DecisionTree:
    root = None
    def __init__(self):
        self.root = Node(None)

def entropy(P, P_y0):
    if P < eps or P > 1 - eps or P_y0 < eps or P_y0 > 1 - eps:
        return 0
    return -P * np.log(P) - P_y0 * np.log(P_y0)

def IC(X, Y):
    m = X.shape[0]
    x_1 = np.count_nonzero(X)
    if x_1 == 0 or x_1 == m:
        return 0
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
    P_x = x_1 / m
    P_00 = c_00 / (c_00 + c_01)
    P_01 = c_01 / (c_00 + c_01)
    P_10 = c_10 / (c_10 + c_11)
    P_11 = c_11 / (c_10 + c_11)
    #print(X, P_x)
    print(c_00, c_01, c_10, c_11)
    #print(P_00, P_01, P_10, P_11)
    return P_x * entropy(P_11, P_10) + (1 - P_x) * entropy(P_01, P_00)

def fit_decision_tree(X, Y, node):
    m, k = X.shape
    P_y = np.count_nonzero(Y) / m
    H_y = entropy(P_y, 1 - P_y) #-P_y * np.log(P_y) - (1 - P_y) * np.log((1 - P_y))
    #print(X, Y)
    print(P_y)
    print(H_y)
    maxx = 0
    max_id = -1
    for i in range(k):
        if vis[i] == 1:
            continue
        IG = H_y - IC(X[:, i], Y)
        if IG > maxx:
            maxx = IG
            max_id = i
    #print(X, Y)
    if max_id != -1:
        vis[max_id] = 1
        new_X_0 = np.copy(X)
        new_X_1 = np.copy(X)
        new_Y_0 = np.copy(Y)
        new_Y_1 = np.copy(Y)
        for i in range(m - 1, -1, -1):
            if new_X_0[i][max_id] > 1 - eps:
                new_X_0 = np.delete(new_X_0, i, axis = 0)
                new_Y_0 = np.delete(new_Y_0, i, axis = 0)
            if new_X_1[i][max_id] < eps:
                new_X_1 = np.delete(new_X_1, i, axis = 0)
                new_Y_1 = np.delete(new_Y_1, i, axis = 0)
        #new_X_0 = np.concatenate((new_X_0[:, :max_id], new_X_0[:, max_id+1:]), axis = 1)
        #new_X_1 = np.concatenate((new_X_1[:, :max_id], new_X_1[:, max_id+1:]), axis = 1)
        #print(max_id, new_X_0, new_Y_0)
        #print(new_X_1, new_Y_1)
        node.x_id = max_id
        if new_X_0.shape[0] > 0:
            node.equals_zero = Node(None)
            fit_decision_tree(new_X_0, new_Y_0, node.equals_zero)
        if new_X_1.shape[0] > 0:
            node.equals_one = Node(None)
            fit_decision_tree(new_X_1, new_Y_1, node.equals_one)
    else:
        if P_y >= 0.5:
            node.y = 1
        else:
            node.y = 0

def print_decision_tree(node, depth = 0):
    if node.y != None:
        print(' ' * 2 * depth + 'Y = %d' % node.y)
    if node.x_id == None:
        return
    if node.equals_zero != None:
        print(' ' * 2 * depth + 'If X[%d] == 0:' % node.x_id)
        print_decision_tree(node.equals_zero, depth + 1)
    if node.equals_one != None:
        print(' ' * 2 * depth + 'If X[%d] == 1:' % node.x_id)
        print_decision_tree(node.equals_one, depth + 1)

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
    tree = DecisionTree()
    k = 4
    m = 30
    #X, Y = get_data(k, m)
    X, Y = load_arr()
    X_train = np.copy(X)
    Y_train = np.copy(Y)
    vis = np.zeros(k)
    fit_decision_tree(X_train, Y_train, tree.root)
    print(tree.root)
    print_decision_tree(tree.root)
