import numpy as np

n = 100
L = 10

LJMOM_list = []
LJMLE_list = []
for j in range(0, 1000):
    X = np.random.uniform(0, L, n)
    mu = np.average(X)
    LJMOM = 2 * mu
    LJMLE = np.max(X)
    LJMOM_list.append(LJMOM)
    LJMLE_list.append(LJMLE)
mu_MOM = np.average(LJMOM_list)
mu_MLE = np.average(LJMLE_list)
var_MOM = np.average(np.array(LJMOM_list) ** 2) - mu_MOM ** 2
var_MLE = np.average(np.array(LJMLE_list) ** 2) - mu_MLE ** 2
biasMOM = L - mu_MOM
biasMLE = L - mu_MLE
MSEMOM = biasMOM ** 2 + var_MOM
MSEMLE = biasMLE ** 2 + var_MLE

X = np.random.uniform(0, L, n)
mu = np.average(X)
LMOM = 2 * mu
LMLE = np.max(X)
print('{:<20s} {:<20s}'.format('Estimated MSEs', 'Theoretival MSEs'))
print('{:<20f} {:<20f}'.format(MSEMOM, 1.0 * L * L / 3 / n))
print('{:<20f} {:<20f}'.format(MSEMLE, 2.0 * L * L / (n + 2) / (n + 1)))
print('L^MOM:' + repr(LMOM))
print('L^MLE:' + repr(LMLE))
