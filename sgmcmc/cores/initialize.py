import numpy as np
from scipy.sparse import diags


def init_zero(data, funcs, tau, reg, info, **kwargs):
    func, grad, hess = funcs
    n, d, C = data['n'], data['d'], data['C']
    beta = np.zeros(d)
    variable = {'beta': beta, 'gamma': beta, 'tau': tau}
    if info in ['SGHMC']:
        N = kwargs['batch']
        # samples = np.random.randint(n, size=N)
        samples = np.array(range(n))
        # delta1,delta2 = grad(beta, data, params, samples)
        # F = fisher(beta, data, samples)
        # I = (1. / (N - 1)) * (F - N * np.outer(delta1, delta1))
        V = fisher(beta, data, samples)
        variable['I'] = V
        variable['B'] = V * n * .1
    return variable


def init_ones(data,params, algo, funcs):
    func, grad, hess = funcs
    n, d, C = data['n'], data['d'], data['C']
    beta = np.ones(d)*params['alpha']
    variable = {'beta': beta, 'gamma': beta, 'tau': params['tau']}
    if algo['method'] == "lbfgs":
        variable['s_mem'] = []
        variable['y_mem'] = []
        variable['H'] = diags(np.ones(d)) * 1e-3
        variable['r'] = np.sum(grad(beta, data, params, np.array(range(n))), axis=0)
    if algo['method'] in ['sgfs', 'sghmc']:
        N = algo['batch']
        # samples = np.random.randint(n, size=N)
        samples = np.array(range(n))
        V = fisher(beta, data, samples)
        variable['I'] = V
        variable['B'] = V * n * .1
    return variable


def init_rand_normal(data, funcs, tau, info, **kwargs):
    func, grad, hess = funcs
    n, d, C = data['n'], data['d'], data['C']
    alpha = kwargs['alpha']
    beta = np.random.normal(0,1,d)*alpha
    gamma = np.random.normal(0,1,d) * alpha
    variable = {'beta': beta, 'gamma': gamma}
    if info in ['SGHMC']:
        N = kwargs['batch']
        samples = np.random.randint(n, size=N)
        # delta1, delta2 = grad(beta, data, params, samples)
        # F = fisher(beta, data, samples)
        # samples = np.array(range(n))
        V = fisher(beta, data, samples)
        # I = (1. / (N - 1)) * (F - N * np.outer(delta1, delta1))
        variable['I'] = V
        variable['B'] = V * n * .1
    return variable


def fisher(beta, data, samples):
    X = data['X'][samples, :]
    Y = data['Y'][samples]
    n = Y.shape[0]
    tmp = np.exp(-X.dot(beta) * Y)
    weight = tmp / (1 + tmp)
    G = X.T * (-weight * Y)
    G_bar = (1./n) * np.outer((X.T.dot(-Y*weight)), np.ones(n))
    F = G - G_bar
    V = 1./(n-1) * F.dot(F.T)
    return V