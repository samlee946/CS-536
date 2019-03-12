import numpy as np

def compute(x, y):
    w = np.cov(x, y)[0][1] / np.var(x)
    b = np.mean(y) - w * np.mean(x)
    return w, b

def question5(m = 200, w = 1, b = 5, sig_2 = 0.1):
    repeat_times = 1000
    w_hat_list = []
    b_hat_list = []
    w_prime_hat_list = []
    b_prime_hat_list = []
    w_the_hat_list = []
    b_the_hat_list = []
    w_prime_hat_the_list = []
    b_prime_hat_the_list = []
    for i in range(repeat_times):
        sz = (1, m)
        x = np.random.uniform(100, 102, sz)
        eps = np.random.normal(0, np.sqrt(sig_2), sz)
        y = w * x + b + eps
        x_prime = x - 101
        w_hat, b_hat = compute(x, y)
        w_prime_hat, b_prime_hat = compute(x_prime, y)
        #print(w_hat, b_hat)
        #print(w_prime_hat, b_prime_hat)
        w_hat_list.append(w_hat)
        b_hat_list.append(b_hat)
        w_prime_hat_list.append(w_prime_hat)
        b_prime_hat_list.append(b_prime_hat)

        w_the = w + np.sum((x - np.mean(x)) * eps) / m / np.var(x)
        b_the = np.mean(y) - w_the * np.mean(x)
        w_prime_the = w + np.sum((x_prime - np.mean(x_prime)) * eps) / m / np.var(x_prime)
        b_prime_the = np.mean(y) - w_the * np.mean(x_prime)
        w_the_hat_list.append(w_the)
        w_prime_hat_the_list.append(w_prime_the)
        b_the_hat_list.append(b_the)
        b_prime_hat_the_list.append(b_prime_the)
    print('Expected values:')
    print('w_hat: {}'.format(np.mean(w_hat_list)))
    print('b_hat: {}'.format(np.mean(b_hat_list)))
    print('w_prime_hat: {}'.format(np.mean(w_prime_hat_list)))
    print('b_prime_hat: {}'.format(np.mean(b_prime_hat_list)))
    print('Variances:')
    print('w_hat: {}'.format(np.var(w_hat_list)))
    print('b_hat: {}'.format(np.var(b_hat_list)))
    print('w_prime_hat: {}'.format(np.var(w_prime_hat_list)))
    print('b_prime_hat: {}'.format(np.var(b_prime_hat_list)))
    print('Theoretical values:')
    print('w_hat: {}'.format(np.mean(w_the_hat_list)))
    print('b_hat: {}'.format(np.mean(b_the_hat_list)))
    print('w_prime_hat: {}'.format(np.mean(w_prime_hat_the_list)))
    print('b_prime_hat: {}'.format(np.mean(b_prime_hat_the_list)))
    print('Theoretical variances:')
    print('w_hat: {}'.format(np.var(w_the_hat_list)))
    print('b_hat: {}'.format(np.var(b_the_hat_list)))
    print('w_prime_hat: {}'.format(np.var(w_prime_hat_the_list)))
    print('b_prime_hat: {}'.format(np.var(b_prime_hat_the_list)))
    return

if __name__ == '__main__':
    question5()
