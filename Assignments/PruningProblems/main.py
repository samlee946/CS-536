import os.path
import numpy as np
import collections
import matplotlib.pyplot as plt

eps = 1e-5
max_datapoints = 500000
reserved_datapoints_for_testing = 50000
data_X = np.zeros((max_datapoints, 21))
data_Y = np.zeros((max_datapoints, 1))

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
    #print(c_00, c_01, c_10, c_11)
    #print(P_00, P_01, P_10, P_11)
    return P_x * entropy(P_11, P_10) + (1 - P_x) * entropy(P_01, P_00)

def fit_decision_tree(X, Y, node, p_vis):
    m, k = X.shape
    P_y = np.count_nonzero(Y) / m
    H_y = entropy(P_y, 1 - P_y) #-P_y * np.log(P_y) - (1 - P_y) * np.log((1 - P_y))
    #print(X, Y)
    #print(P_y)
    #print(H_y)
    maxx = 0
    max_id = -1
    vis = np.copy(p_vis)
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
            fit_decision_tree(new_X_0, new_Y_0, node.equals_zero, vis)
        if new_X_1.shape[0] > 0:
            node.equals_one = Node(None)
            fit_decision_tree(new_X_1, new_Y_1, node.equals_one, vis)
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
    print(' ' * 2 * depth + 'If X[%d] == 0:' % node.x_id)
    if node.equals_zero != None:
        print_decision_tree(node.equals_zero, depth + 1)
    else:
        print(' ' * 2 * (depth + 1) + 'No data')
    print(' ' * 2 * depth + 'Else:')
    if node.equals_one != None:
        #print(' ' * 2 * depth + 'If X[%d] == 1:' % node.x_id)
        print_decision_tree(node.equals_one, depth + 1)
    else:
        print(' ' * 2 * (depth + 1) + 'No data')

def generate_unique_data(load = False, save = False):
    global data_X
    global data_Y
    if load == True:
        data_X, data_Y = load_arr()
        return data_X, data_Y
    exist = set()
    for j in range(max_datapoints):
        print('Generating data point %d' %j)
        while(True):
            num = 0
            for i in range(21):
                if i == 0 or i >= 15:
                    if np.random.rand() >= 0.5:
                        data_X[j][i] = 1
                    else:
                        data_X[j][i] = 0
                else:
                    if np.random.rand() >= 0.75:
                        data_X[j][i] = data_X[j][i - 1]
                    else:
                        data_X[j][i] = 1 - data_X[j][i - 1]
                num *= 2
                num += data_X[j][i]
            if num not in exist:
                exist |= {num}
                if data_X[j][0] == 0:
                    data_Y[j] = collections.Counter(data_X[j][1:8]).most_common()[0][0]
                else:
                    data_Y[j] = collections.Counter(data_X[j][8:15]).most_common()[0][0]
                #print(data_X[j], data_Y[j])
                break
    if save == True:
        save_arr(np.concatenate((data_X, data_Y), axis = 1))
    return data_X, data_Y

def get_data(m, for_testing = False):
    # input args:
    # k: number of features
    # m: number of data points
    if for_testing == True:
        return data_X[-m:], data_Y[-m:]
    else:
        idx = np.random.randint(max_datapoints - reserved_datapoints_for_testing, size = m)
        return data_X[idx, :], data_Y[idx]

def predict(node, x):
    if node == None:
        return 1 # No data captured
    if node.y != None:
        return node.y
    if node.x_id == None:
        print('?????')
    if x[node.x_id] < eps:
        return predict(node.equals_zero, x)
    else:
        return predict(node.equals_one, x)

def get_err(tree, X, Y):
    s = 0
    m, k = X.shape
    for i in range(m):
        prediction = predict(tree.root, X[i])
        #if X[i][0] == 1 and X[i][2] == 0:
        #print(X[i], prediction, Y[i])
        if prediction != Y[i]:
            s += 1
    #print('The error is: %f' % (1.0 * s / m))
    return 1.0 * s / m

def get_num_of_vars(node):
    if node.x_id == None:
        return set()
    set_num_of_vars = set()
    if node.x_id > 14:
        set_num_of_vars |= {node.x_id}
    if node.equals_zero != None:
        set_num_of_vars |= get_num_of_vars(node.equals_zero)
    if node.equals_one != None:
        set_num_of_vars |= get_num_of_vars(node.equals_one)
    #print(set_num_of_vars)
    return set_num_of_vars

def save_arr(arr, filename = 'data.npy'):
    np.save(filename, arr)
    print('Saving arr to ' + filename)

def load_arr(filename = 'data.npy'):
    arr = np.load(filename)
    return arr[:, :-1], arr[:, -1]

def test(tree, m):
    X_test, Y_test = get_data(m, save = False)
    err = get_err(tree, X_test, Y_test)
    return err

def uni_test():
    X, Y = get_data(m = 30)
    vis = np.zeros(21)
    tree = DecisionTree()
    fit_decision_tree(X, Y, tree.root, vis)
    print_decision_tree(tree.root)
    print(test(tree, 50))

def question1(show = True, save = False, save_file = 'q1.png'):
    k = 21
    errors = []
    repeat_times_1 = 5
    num_of_test_data_sets = 1
    X_test, Y_test = get_data(50000, for_testing = True)
    ms = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000]
    for m in ms:
        err = 0
        for j in range(repeat_times_1):
            X_train, Y_train = get_data(m)
            vis = np.zeros(k)
            tree = DecisionTree()
            fit_decision_tree(X_train, Y_train, tree.root, vis)
            #print_decision_tree(tree.root)
            #for i in range(num_of_test_data_sets):
            #    #print(m, i)
            #    err += test(tree, 500)
            err += get_err(tree, X_test, Y_test) 
            print('Accumulate error for %d, the %d time is: %f' % (m, j, err))
        print('Error for %d is %f:' % (m, err / repeat_times_1 / num_of_test_data_sets))
        errors.append(err / repeat_times_1 / num_of_test_data_sets)
    plt.xlabel('m')
    plt.ylabel('err')
    plt.plot(ms, errors, '--o')
    if show == True:
        plt.show()
    if save == True:
        plt.savefig(save_file)
    return errors

def question2(show = True, save = False, save_file = 'q2.png'):
    k = 21
    num_of_vars = []
    repeat_times_1 = 10
    num_of_test_data_sets = 1
    #ms = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000]
    ms = [10000]
    for m in ms:
        ir_vars = 0
        for j in range(repeat_times_1):
            X_train, Y_train = get_data(m, save = False)
            vis = np.zeros(k)
            tree = DecisionTree()
            fit_decision_tree(X_train, Y_train, tree.root, vis)
            ir_vars += len(get_num_of_vars(tree.root))
            print(get_num_of_vars(tree.root))
        print('Average irrelevant variables for m=%d is %f:' % (m, ir_vars / repeat_times_1 / num_of_test_data_sets))
        num_of_vars.append(ir_vars / repeat_times_1 / num_of_test_data_sets)
    plt.xlabel('m')
    plt.ylabel('avg. number of irrelevant variables')
    plt.plot(ms, num_of_vars, '--o')
    if show == True:
        plt.show()
    if save == True:
        plt.savefig(save_file)
    return errors

def question3():
    tree = DecisionTree()
    k = 4
    m = 30
    #X, Y = get_data(k, m, save = True)
    X, Y = load_arr()
    X_train = np.copy(X)
    Y_train = np.copy(Y)
    print(X_train)
    print(Y_train)
    vis = np.zeros(k)
    fit_decision_tree(X_train, Y_train, tree.root, vis)
    print_decision_tree(tree.root)

def question4():
    tree = DecisionTree()
    k = 4
    m = 30
    #X, Y = get_data(k, m, save = True)
    X, Y = load_arr()
    X_train = np.copy(X)
    Y_train = np.copy(Y)
    vis = np.zeros(k)
    fit_decision_tree(X_train, Y_train, tree.root, vis)
    print_decision_tree(tree.root)
    get_err(tree, X_train, Y_train)
    err = 0
    for i in range(50):
        err += test(tree, 4, 10000)
    print(err / 50)

def question5():
    k = 10
    errors = []
    ms = [30, 100, 300, 1000, 3000, 10000]
    for m in ms:
        err = 0
        for j in range(10):
            X_train, Y_train = get_data(k, m, save = False)
            vis = np.zeros(k)
            tree = DecisionTree()
            fit_decision_tree(X_train, Y_train, tree.root, vis)
            #print_decision_tree(tree.root)
            for i in range(10):
                #print(m, i)
                err += test(tree, k, 10000)
        print(err / 100)
        errors.append(err / 100)
    plt.xlabel('m')
    plt.ylabel('err')
    plt.plot(ms, errors, '--o')
    plt.show()
    return errors

def question6():
    k = 10
    errors5 = question5()
    errors = []
    ms = [30, 100, 300, 1000, 3000, 10000]
    for m in ms:
        err = 0
        for j in range(10):
            X_train, Y_train = get_data(k, m, save = False)
            vis = np.zeros(k)
            tree = DecisionTree()
            fit_decision_tree_my(X_train, Y_train, tree.root, vis)
            #print_decision_tree(tree.root)
            for i in range(10):
                #print(m, i)
                err += test(tree, k, 10000)
        print(err / 100)
        errors.append(err / 100)
    fig, ax = plt.subplots()
    plt.xlabel('m')
    plt.ylabel('err')
    line6, = ax.plot(ms, errors, '--ro', label='My approach')
    line5, = ax.plot(ms, errors5, '--bo', label='Information gain')
    ax.legend()
    plt.show()
    return errors

def count():
    global data_X
    global data_Y
    exist = set()
    for j in range(max_datapoints):
        while(True):
            num = 0
            for i in range(21):
                num *= 2
                num += data_X[j][i]
            exist |= {num}
            break
    print(sorted(list(exist)))
 
if __name__ == '__main__':
    if os.path.isfile('data.npy'):
        generate_unique_data(load = True, save = False)
    else:
        generate_unique_data(load = False, save = True)
    #count()
    #question2(show = True, save = True)
    question1(show = False, save = True)
