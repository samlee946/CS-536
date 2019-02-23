import time
import os.path
import numpy as np
import collections
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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

def chi_square_test(X, Y):
    m = X.shape[0]
    x = np.zeros((2, 1))
    y = np.zeros((2, 1))
    O = np.zeros((2, 2))
    P_X = np.zeros((2, 1))
    P_Y = np.zeros((2, 1))
    #x[1] = np.count_nonzero(X)
    #x[0] = m - x[1]
    #y[1] = np.count_nonzero(Y)
    #y[0] = m - y[1]
    for x_, y_ in zip(X, Y):
        if x_ < eps:
            if y_ < eps:
                O[0][0] += 1
            else:
                O[0][1] += 1
        else:
            if y_ < eps:
                O[1][0] += 1
            else:
                O[1][1] += 1
    for i in range(2):
        #P_X[i] = 1.0 * x[i] / m
        #P_Y[i] = 1.0 * y[i] / m
        #print(O[i][0] + O[i][1])
        #print(O[0][i] + O[1][i])
        P_X[i] = 1.0 * (O[i][0] + O[i][1]) / m
        P_Y[i] = 1.0 * (O[0][i] + O[1][i]) / m
    T = 0
    for i in range(2):
        for j in range(2):
            E = P_X[i] * P_Y[j] * m
            fh = (O[i][0] + O[i][1])
            sh = (O[0][j] + O[1][j])
            E2 = 1.0 * fh  * sh / m
            if E == 0:
                continue
            #print(i, j, fh, sh, O[i][j] , E, E2)
            #print(O[i][j] - E)
            T += ((O[i][j] - E) ** 2) / E
    return T

def fit_decision_tree(X, Y, node, p_vis, pruning = 0, extra_info = None):
    # args
    # pruning = 0, 1, 2, 3 -- no pruning, by depth, by size, by significance
    if pruning != 0 and extra_info is None:
        print('ERROR, EXTRA_INFO IS NEEDED')
        return
    m, k = X.shape
    P_y = np.count_nonzero(Y) / m
    # Pruning
    if pruning == 1 and extra_info == 0: # depth = 0
        if P_y >= 0.5:
            node.y = 1
        else:
            node.y = 0
        return
    elif pruning == 2 and X.shape[0] <= extra_info: # sample size <= threshold
        if P_y >= 0.5:
            node.y = 1
        else:
            node.y = 0
        return
    if pruning == 1:
        extra_info -= 1

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
        if pruning == 3 and chi_square_test(X[:, max_id], Y) < extra_info:
            #print(max_id)
            #print(maxx)
            #print(chi_square_test(X[:, max_id], Y))
            if P_y >= 0.5:
                node.y = 1
            else:
                node.y = 0
            return
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
            fit_decision_tree(new_X_0, new_Y_0, node.equals_zero, vis, pruning, extra_info)
        if new_X_1.shape[0] > 0:
            node.equals_one = Node(None)
            fit_decision_tree(new_X_1, new_Y_1, node.equals_one, vis, pruning, extra_info)
    else:
        if P_y >= 0.5:
            node.y = 1
        else:
            node.y = 0

def print_decision_tree(node, depth = 0):
    if node.y != None:
        print(' ' * 2 * depth + 'Y = %d' % node.y)
    if node.x_id == None:
        return depth
    print(' ' * 2 * depth + 'If X[%d] == 0:' % node.x_id)
    max_depth = depth
    if node.equals_zero != None:
        max_depth = max(max_depth, print_decision_tree(node.equals_zero, depth + 1))
    else:
        print(' ' * 2 * (depth + 1) + 'No data')
    print(' ' * 2 * depth + 'Else:')
    if node.equals_one != None:
        #print(' ' * 2 * depth + 'If X[%d] == 1:' % node.x_id)
        max_depth = max(max_depth, print_decision_tree(node.equals_one, depth + 1))
    else:
        print(' ' * 2 * (depth + 1) + 'No data')
    return max_depth

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
    ms = [100, 300, 1000, 3000, 10000, 30000, 100000]
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

def question2(show = True, save = False, save_file = 'q5.png'):
    k = 21
    num_of_vars = []
    repeat_times_1 = 50
    ms = [100, 300, 1000, 3000, 10000, 30000, 100000]
    #t_start = time.time()
    for m in ms:
        ir_vars = 0
        for j in range(repeat_times_1):
            X_train, Y_train = get_data(m)
            vis = np.zeros(k)
            tree = DecisionTree()
            fit_decision_tree(X_train, Y_train, tree.root, vis, pruning = 1, extra_info = 10)
            ir_vars += len(get_num_of_vars(tree.root))
            #print(get_num_of_vars(tree.root))
            #print(print_decision_tree(tree.root))
        print('Average irrelevant variables for m=%d is %f:' % (m, ir_vars / repeat_times_1))
        num_of_vars.append(ir_vars / repeat_times_1)
    #t_stop = time.time()
    #print((t_stop - t_start))
    plt.xlabel('m')
    plt.ylabel('avg. number of irrelevant variables')
    plt.plot(ms, num_of_vars, '--o')
    if show == True:
        plt.show()
    if save == True:
        plt.savefig(save_file)
    return num_of_vars

def question3(show = True, save = False, save_file = 'q3-2.png'):
    m = 10000
    X, Y = get_data(m, for_testing = True)
    X_train = X[:8000,:]
    Y_train = Y[:8000]
    X_test = X[-2000:,:]
    Y_test = Y[-2000:]

    D = np.arange(1, 17)
    #S = np.arange(50, 4, -5) + np.arange(4, 1, -1)
    S = np.concatenate((np.arange(50, 4, -5), np.arange(4, 0, -0.5)))
    T0 = [0.102, 0.455, 1.323, 2.706, 3.841, 5.024, 6.635, 7.879, 9.550, 10.828]

    errs_train = []
    err_D = []
    err_S = []
    err_T = []

#    print('Pruning by depth')
#    for d in D:
#        vis = np.zeros(21)
#        tree = DecisionTree()
#        fit_decision_tree(X_train, Y_train, tree.root, vis, pruning = 1, extra_info = d)
#        err_train = get_err(tree, X_train, Y_train)
#        err_test = get_err(tree, X_test, Y_test)
#        #print_decision_tree(tree.root)
#        print(err_test)
#        err_D.append(err_test)
#        errs_train.append(err_train)
#
    print('Pruning by sample size')
    for s in S:
        vis = np.zeros(21)
        tree = DecisionTree()
        fit_decision_tree(X_train, Y_train, tree.root, vis, pruning = 2, extra_info = s)
        err_train = get_err(tree, X_train, Y_train)
        err_test = get_err(tree, X_test, Y_test)
        #print_decision_tree(tree.root)
        print(err_test)
        err_S.append(err_test)
        errs_train.append(err_train)

#    print('Pruning by significance')
#    for t in T0:
#        vis = np.zeros(21)
#        tree = DecisionTree()
#        fit_decision_tree(X_train, Y_train, tree.root, vis, pruning = 3, extra_info = t)
#        err_train = get_err(tree, X_train, Y_train)
#        err_test = get_err(tree, X_test, Y_test)
#        #print_decision_tree(tree.root)
#        print(err_test)
#        err_T.append(err_test)
#        errs_train.append(err_train)

    fig, ax = plt.subplots()
    plt.xlabel('sample size (%)')
    plt.ylabel('err')
    #line_d, = ax.plot(list(D), err_D, '--ro', label='Error_test')
    line_train, = ax.plot(list(S), errs_train, '--bo', label='Error_train')
    line_s, = ax.plot(list(S), err_S, '--ro', label='Error_test')
    plt.gca().invert_xaxis()
    #line_t, = ax.plot(list(T0), err_T, '--ro', label='Error_test')
    ax.legend()
    if show == True:
        plt.show()
    if save == True:
        plt.savefig(save_file)

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
    #question1(show = False, save = True)
    question2(show = False, save = True)
    #question3(show = False, save = True)
