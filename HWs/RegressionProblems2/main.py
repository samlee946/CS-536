import numpy as np
import matplotlib.pyplot as plt

def naive_linear_regression(X, Y):
    Xt = X.transpose()
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(Xt, X)), Xt), Y)
    return w

def ridge_regression(X, Y, Lambda):
    Xt = X.transpose()
    I = np.identity(X.shape[1])
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(Xt, X)+ Lambda * I), Xt), Y)
    return w

def lasso_regression(X, Y, Lambda):
    num_iters = 100
    m = Y.shape[0]
    w = np.zeros((num_iters + 1, 21))
    w = [np.zeros(21) for _ in range(num_iters + 1)]

    for i in range(1, num_iters + 1):
        w[i] = w[i - 1]
        error = 0
        predictions = np.matmul(X, w[i]).reshape(Y.shape)
        diff = Y - predictions
        w[i][0] = w[i - 1][0] + np.sum(diff) / m

        for j in range(1, 21):
            predictions = np.matmul(X, w[i]).reshape(Y.shape)
            diff = Y - predictions
            Xt = X[:, j].transpose()
            dinom = np.matmul(Xt, X[:, j])
            num1 = (-np.matmul(Xt, diff) + Lambda/2)
            num2 = (-np.matmul(Xt, diff) - Lambda/2)
            threshold1 = num1 / dinom
            threshold2 = num2 / dinom
            if w[i - 1][j] > threshold1:
                w[i][j] = w[i - 1][j] - threshold1
            elif w[i - 1][j] < threshold2:
                w[i][j] = w[i - 1][j] - threshold2
            else:
                w[i][j] = 0
    #print w[num_iters - 1]
    return w[num_iters - 1]

def get_data(m, load = False, save = False):
    bias = 0
    coef = np.zeros(21)
    for i in range(21):
        coef[i] = 0.6**i
    if load == True:
        X, Y = load_arr()
        X = X.reshape((m, 21))
        Y = Y.reshape((m, 1))
    else:
        X = np.zeros((m, 21))
        Y = np.zeros((m, 1))
        for i in range(m):
            X[i, 0] = 1
            X[i, 1:21] = np.random.standard_normal(20)
            X[i, 11] = X[i, 1] + X[i, 2] + np.random.normal(0, np.sqrt(0.1))
            X[i, 12] = X[i, 3] + X[i, 4] + np.random.normal(0, np.sqrt(0.1))
            X[i, 13] = X[i, 4] + X[i, 5] + np.random.normal(0, np.sqrt(0.1))
            X[i, 14] = 0.1 * X[i, 7] + np.random.normal(0, np.sqrt(0.1))
            X[i, 15] = 2 * X[i, 2] - 10 + np.random.normal(0, np.sqrt(0.1))
            b = 10 + np.random.normal(0, np.sqrt(0.1))
            Y[i] = np.sum(coef[1:11] * X[i, 1:11]) + b
            bias += b
    if save == True:
        save_arr(np.concatenate((X, Y), axis = 1))
    #print(X.shape, Y.shape)
    true_weight = np.zeros(21)
    true_weight[1:11] = coef[1:11]
    return X, Y, true_weight, bias / m

def plot_data(X):
    plt.xlabel('X')

def save_arr(arr, filename = 'data.npy'):
    np.save(filename, arr)
    print('Saving arr to ' + filename)

def load_arr(filename = 'data.npy'):
    arr = np.load(filename)
    return arr[:, :-1], arr[:, -1]

def question1(save = False):
    X, Y, true_weight, true_bias = get_data(1000, save = save, load = not save)
    w = naive_linear_regression(X, Y)

    plot_comparison_true_weights(w, true_weight, true_bias, 'q1-1.png',
                                 'q1-2.png')

def plot_comparison_true_weights(w, true_weight, true_bias, filename1,
                                 filename2):
    #plot weight
    index = np.arange(1, 21)
    fig, ax = plt.subplots()
    bar_width = 0.3
    rects1 = plt.bar(index, true_weight[1:21], bar_width, label = 'True Weights')
    rects2 = plt.bar(index + bar_width, w[1:21], bar_width, color = 'g', label
                     = 'Model Weights')
    plt.legend()
    #plt.show()
    plt.savefig(filename1)
    plt.cla()

    #plot bias
    index = np.arange(1, 2)
    rects1 = plt.bar(index, true_bias, bar_width, label = 'True Bias')
    rects2 = plt.bar(index + bar_width, w[0], bar_width, color = 'g', label =
                     'Model Bias')
    plt.legend()
    #plt.show()
    plt.savefig(filename2)

    get_true_error(100000, w)

def get_true_error(n_data, w, w_l = None):
    X, Y = get_data(n_data)[:2]
    if w_l is not None:
        for i in range(20, 0, -1):
            if w_l[i] == 0:
                #X = np.delete(X, i, 1)
                X[:, i] = 0
    true_error = 0
    for i in range(n_data):
        prediction = np.matmul(X[i], w)
        true_error += (Y[i] - prediction)**2
    true_error /= 1.0 * n_data
    print('True error is: {}'.format(true_error))
    return true_error

def question2(save = False):
    repeat_times = 5
    X, Y, true_weight, true_bias = get_data(1000)

    min_error = float('inf')
    optimalLambda = None
    optimal_w = None

    lambdas = np.arange(0, 5, 0.05)
    errors = []
    for Lambda in lambdas:
        error = 0
        w = ridge_regression(X, Y, Lambda)
        for i in range(repeat_times):
            error += get_true_error(10000, w)
        error /= 1.0 * repeat_times
        errors.append(error)
        if error < min_error:
            min_error = error
            optimalLambda = Lambda
            optimal_w = w

    plt.plot(lambdas, errors)
    plt.xlabel('Lamdba')
    plt.ylabel('True Error')
    #plt.show()
    plt.savefig('q2-1.png')
    print('optimalLambda = {}, error = {}'.format(optimalLambda, min_error))
    #plot_comparison_true_weights(optimal_w, true_weight, true_bias, 'q2-2.png',
    #                             'q2-3.png')

def question3(save = False):
    X, Y = get_data(1000)[:2]

    lambdas = np.arange(0, 100, 0.1)
    cnt = []
    for Lambda in lambdas:
        t = 0
        w = lasso_regression(X, Y, Lambda)
        for weight in w:
            if weight == 0:
                t += 1
        print(Lambda, t)
        cnt.append(t)

    plt.plot(lambdas, cnt)
    plt.xlabel('Lamdba')
    plt.ylabel('Number of eliminated features')
    #plt.show()
    plt.savefig('q3-1.png')
    #print('optimalLambda = {}, error = {}'.format(optimalLambda, min_error))
    #plot_comparison_true_weights(optimal_w, true_weight, true_bias, 'q2-2.png',
    #                             'q2-3.png')

def question4(save = False):
    repeat_times = 1
    X, Y, true_weight, true_bias = get_data(1000)

    min_error = float('inf')
    optimalLambda = None
    optimal_w = None

    lambdas = np.arange(0, 50, 0.05)
    errors = []
    for Lambda in lambdas:
        error = 0
        w = lasso_regression(X, Y, Lambda)
        for i in range(repeat_times):
            error += get_true_error(10000, w)
        error /= 1.0 * repeat_times
        errors.append(error)
        if error < min_error:
            min_error = error
            optimalLambda = Lambda
            optimal_w = w

    plt.plot(lambdas, errors)
    plt.xlabel('Lamdba')
    plt.ylabel('True Error')
    #plt.show()
    plt.savefig('q4-1.png')
    print('optimalLambda = {}, error = {}'.format(optimalLambda, min_error))

    plot_comparison_true_weights(optimal_w, true_weight, true_bias, 'q4-2.png',
                                 'q4-3.png')

def question5(save = False):
    X_raw, Y, true_weight, true_bias = get_data(1000)
    w_l = lasso_regression(X_raw, Y, 17.4)
    X = X_raw
    for i in range(20, 0, -1):
        if w_l[i] == 0:
            #X = np.delete(X, i, 1)
            X[:, i] = 0
    #print(X.shape)

    repeat_times = 10
    min_error = float('inf')
    optimalLambda = None
    optimal_w = None

    lambdas = np.arange(0.05, 5, 0.05)
    errors = []
    for Lambda in lambdas:
        error = 0
        w = ridge_regression(X, Y, Lambda)
        for i in range(repeat_times):
            error += get_true_error(10000, w, w_l)
        error /= 1.0 * repeat_times
        errors.append(error)
        if error < min_error:
            min_error = error
            optimalLambda = Lambda
            optimal_w = w

    plt.plot(lambdas, errors)
    plt.xlabel('Lamdba')
    plt.ylabel('True Error')
    #plt.show()
    plt.savefig('q5-1.png')
    print('optimalLambda = {}, error = {}'.format(optimalLambda, min_error))

    plot_comparison_true_weights(optimal_w, true_weight, true_bias, 'q5-2.png',
                                 'q5-3.png')

if __name__ == '__main__':
    #question1(save = True)
    #question2(save = False)
    #question3(save = False)
    #question4(save = False)
    question5()
