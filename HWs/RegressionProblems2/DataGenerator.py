import numpy as np
import math


class DG:
    def __init__(self):
        pass

    def one_data(self):
        x = np.random.standard_normal(21)
        x[0] = 1
        x[11] = x[1] + x[2] + np.random.normal(0, math.sqrt(0.1))
        x[12] = x[3] + x[4] + np.random.normal(0, math.sqrt(0.1))
        x[13] = x[4] + x[5] + np.random.normal(0, math.sqrt(0.1))
        x[14] = x[7] * 0.1 + np.random.normal(0, math.sqrt(0.1))
        x[15] = 2 * x[2] - 10 + np.random.normal(0, math.sqrt(0.1))

        y = 10 + np.random.normal(0, math.sqrt(0.1))
        for i in range(10):
            y += (0.6**(i+1))*x[i+1]

        return x, y

    def generate(self, size):
        X = []
        Y = []
        for i in range(size):
            x, y = self.one_data()
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)


if __name__ == '__main__':
    dg = DG()
    x, Y = dg.generate(100)
    print(np.shape(x))
    print(np.shape(Y))
    print(x, Y)
