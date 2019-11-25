from time import *
import numpy as np
import numpy.linalg as la
from tqdm import tqdm
from tqdm import trange
from abc import ABC, abstractmethod
from sgmcmc.cores.loss import *
from sgmcmc.cores.utils import *
from sgmcmc.cores.initialize import *
from scipy.sparse import diags
from sklearn.preprocessing import OneHotEncoder


class MCMC(ABC):
    def __init__(self, linesearch=btls,max_iter=100, stepsize=1e-2, reg=0., tau=0, seed=200,
                 btls_alpha=.1, btls_beta=.9, m=10, batch=1000, alpha=1., info='SGLD'):
        self.linesearch=linesearch
        self.btls_alpha = btls_alpha
        self.btls_beta = btls_beta
        self.max_iter=max_iter
        self.stepsize=stepsize
        self.reg = reg
        self.tau = tau
        self.seed = seed
        self.m = m
        self.batch = batch
        self.alpha = alpha
        self.info = info

    def solve(self, data, funcs, init_method=init_zero):
        # params
        linesearch = {"method": self.linesearch, "alpha": self.btls_alpha, "beta": self.btls_beta,
                      "stepsize": self.stepsize}
        info = self.info
        tau = self.tau
        reg = self.reg
        m = self.m
        batch = self.batch
        max_iter = self.max_iter
        train = data['train']
        test = data['test']
        alpha = self.alpha
        # # params
        # linesearch = self.algo['linesearch']
        # train = self.data['train']
        # test = self.data['test']
        # funcs = self.funcs
        # params = self.params
        # algo = self.algo

        # initialization
        betas, tr_errors, regularizers, te_errors, elapsed_times = [], [], [], [], []
        variable = init_method(train, funcs, tau, reg, info, batch=batch, alpha=alpha)
        func, grad, hess = funcs
        beta = variable['beta']
        # betas.append(beta)
        # loss, reg = func(beta, train, params, np.array(range(train['n'])))
        te_error = func(beta, test, reg, np.array(range(test['n'])))[0]
        # losses.append(loss)
        # regs.append(reg)
        # te_errors.append(te_error)
        # elapsed_times.append(0)

        # optimization
        with trange(max_iter-1) as t:
            for iteration in t:
                start = time()
                variable = self.update(variable, train, funcs, reg, linesearch, batch=batch)
                end = time()
                beta = variable['beta']
                tr_error, regularizer = func(beta, train, reg, np.array(range(train['n'])))
                if iteration >= 10:
                    te_error = func(beta, test, reg, np.array(range(test['n'])))[0]
                    betas.append(beta)
                    tr_errors.append(tr_error)
                    regularizers.append(regularizer)
                    te_errors.append(te_error)
                    elapsed_times.append(end - start)
                t.set_description('tr_error : {:3.6f}, reg : {:.7e}, te_error : {:3.6f} '.format(tr_error, regularizer, te_error))
                if (iteration > 0) and (iteration % int(max_iter/5) == 0):
                    t.write(' ')
        res = {
            'betas': betas, 'beta': beta,
            'tr_errors': tr_errors, 'regs': regularizers, 'te_errors': te_errors,
            'times': list(np.cumsum(elapsed_times))
        }
        return res

    @abstractmethod
    def update(self, variable, data, funcs, reg, linesearch, **kwargs):
        raise NotImplementedError


class SGLD(MCMC):
    def update(self, variable, data, funcs, reg, linesearch, **kwargs):
        func, grad, hess = funcs
        beta = variable['beta']
        n = data['n']
        samples = np.random.randint(n, size=kwargs['batch'])
        delta1, delta2, = grad(beta, data, reg, samples)
        delta = -np.sum((delta1 * n, delta2), axis=0)
        eta = linesearch['method'](beta, func, grad, delta, data, reg, samples, linesearch)
        noise = np.random.normal(0, np.sqrt(eta), data['d'])
        beta = beta + eta/2. * delta + noise
        variable['beta'] = beta
        return variable


class SGHMC(MCMC):
    def update(self, variable, data, funcs, reg, linesearch, **kwargs):
        func, grad, hess = funcs
        beta = variable['beta']
        a, b = .0001, 0
        n, d = data['n'], data['d']
        eta = linesearch['method'](beta, func, grad, None, data, reg, None, linesearch)
        gamma = np.random.normal(0, 1, d)
        # samples = np.random.randint(n, size=algo['batch'])
        for _ in range(20):
            beta = beta + eta * gamma
            samples = np.random.randint(n, size=kwargs['batch'])
            delta1, delta2, = grad(beta, data, reg, samples)
            delta = -np.sum((delta1 * n, delta2), axis=0)
            noise = np.random.normal(0, np.sqrt(2 * (a - b) * eta), data['d'])
            gamma = (1-a*eta)*gamma + eta * delta + noise
        variable['beta'] = beta
        variable['gamma'] = gamma
        return variable


# class SGFS(Optimize):
#     def update(self, variable, funcs, data, linesearch, iteration, algo, params):
#         func, grad, hess = funcs
#         beta = variable['beta']
#         I = variable['I']
#         n = data['n']
#         N = algo['batch']
#         d = data['d']
#         B = variable['B']
#         # lamb = .001
#         prop = (n+N) / N
#         kai = 1./(iteration+1.)
#         samples = np.random.randint(n, size=N)
#         delta1, delta2 = grad(beta, data, params, samples)
#         delta = -np.sum((delta1 * n, delta2), axis=0)
#         V = fisher(beta, data, samples)
#         I = (1-kai)*I + kai*V
#         eta = linesearch(beta, func, grad, None, data, params, samples, algo, iteration)
#         # noise = np.random.normal(0, 4*lamb/eta, d)
#         noise = np.random.normal(np.zeros(d), 4*B / eta)
#         beta = beta + 2 * np.linalg.solve(prop * n * I + 4*B/eta, delta + noise)
#         variable['I'] = I
#         variable['beta'] = beta
#         return variable
#
#     @classmethod
#     def num_grad(cls, train, algorithm):
#         n = train['n']
#         counts = list(np.array(range(algorithm['max_iter'])) * n)
#         return counts
#
#     # def fisher(self,beta, data, params, samples):
#     #     X = data['X'][samples, :]
#     #     Y = data['Y'][samples]
#     #     n = Y.shape[0]
#     #     tmp = np.exp(-X.dot(beta) * Y)
#     #     weight = tmp / (1 + tmp)
#     #     G = X.T.dot(-weight * Y)
#     #     F = G.dot(G.T)
#     #     return F
#
#
# class SGHMC(Optimize):
#     def update(self, variable, funcs, data, linesearch, iteration, algo, params):
#         func, grad, hess = funcs
#         beta = variable['beta']
#         gamma = variable['gamma']
#         d = data['d']
#         I = variable['I']
#         N = algo['batch']
#         n = data['n']
#         m = algo['m']
#         lamb = .5
#         # gamma = np.random.normal(0, 1, d)
#         samples = np.random.randint(n, size=N)
#         kai = 1. / (iteration + 1)
#         eta = linesearch(beta, func, grad, None, data, params, None, algo, iteration)
#         for i in range(m):
#             beta_old = beta
#             beta = beta + eta * gamma
#             delta1, delta2, = grad(beta_old, data, params, samples)
#             delta = -np.sum((delta1 * n, delta2), axis=0)
#             F = fisher(beta, data, samples)
#             V = (1. / (N - 1)) * (F - N * np.outer(delta1, delta1))
#             I = (1 - kai) * I + kai * V
#             # B = .5* eta * I
#             B = .5 * lamb * I
#             C = eta * B/2. + lamb*np.eye(d)/2.
#             noise = np.random.normal(0, lamb * eta, d)
#             gamma = gamma + eta * delta - eta * C.dot(gamma) + noise
#         variable['beta'] = beta
#         variable['gamma'] = beta
#         variable['I']=I
#         return variable
#
#     @classmethod
#     def num_grad(cls, train, algorithm):
#         n = train['n']
#         counts = list(np.array(range(50, algorithm['max_iter'])) * (algorithm['batch']*algorithm['m']))
#         return counts

