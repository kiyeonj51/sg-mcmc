from sgmcmc.cores.data import *
from sgmcmc.cores.utils import *
from sgmcmc.cores.loss import *
from sgmcmc.cores.algo import SGLD, SGHMC
from sgmcmc.cores.initialize import *

# Example Setting
params = {
    "data_name": DataMNIST,
    "seed": 200,
    "reg": 1.,
    "tau": 0,
    "stepsize": 5e-4,
    "max_iter": 2000,
    "a": 0.1,
    "b": 0.9,
    "init_method": init_rand_normal,
    "alpha": 1.,
    "is_mcr": True,
    "is_numgrad": False,
    "is_save": True,
    "batch": 500,
}

plots = [
    {"ylabel": "mcrs", "xlabel": "iteration",
     "yscale":"log", "title": "mis classification rate", "xlim": False},
]

file_name = []
for key, val in params.items():
    if key == "init_method":
        val = val.__name__
        file_name.append(val)
    if key in ["seed", "reg", "stepsize", "max_iter"]:
        file_name.append(key)
        file_name.append(str(val))
file_name = params['data_name'].__name__+'_'+'_'.join(file_name)

funcs = (func_lr_2c,  grad_lr_2c, None)

mcmc_algorithms = [
    ("SGLD_500", "+", SGLD(linesearch=constantstep, max_iter=params['max_iter'], batch=params['batch'],
                       reg=params['reg'], stepsize=params['stepsize'], info='SGLD')),
    ("SGHMC", "*", SGHMC(linesearch=constantstep, max_iter=params['max_iter'], batch=params['batch'],
                         reg=params['reg'],stepsize=params['stepsize'], info='SGHMC')),
]

dataset = params['data_name']
data = dataset.load_data()

results = []
for name, marker, algorithm in mcmc_algorithms:
    print(f'{name}')
    res = algorithm.solve(data=data, funcs=funcs)
    res["marker"] = marker
    res["name"]=name
    if params["is_mcr"]:
        res["mcrs"] = mcr_posterior(data['test'], res['betas'])
    if params["is_numgrad"]:
        res["num_grad"] = algorithm.num_grad(data['train'])
    results.append(res)

# collect results
outputs = {'tr_errors': [res['tr_errors'] for res in results],
           'te_errors': [res['te_errors'] for res in results],
           'times': [res['times'] for res in results],
           'names': [res['name'] for res in results],
           'markers': [res['marker'] for res in results],
           'is_save': params['is_save']}
if params["is_mcr"]:
    outputs['mcrs'] = [res['mcrs'] for res in results]
if params["is_numgrad"]:
    outputs['num_grads'] = [res['num_grad'] for res in results]

# plot results
for plot in plots:
    plotresult(plot, outputs, file_name)