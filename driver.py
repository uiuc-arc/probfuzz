import pystan
import pickle
from hashlib import md5
import json
import subprocess as sp


def StanModel_cache(model_code, rebuild=False):
    code_hash = md5(model_code.encode('ascii')).hexdigest()

    cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    try:
        if not rebuild:
            sm = pickle.load(open(cache_fn, 'rb'))
        else:
            raise FileNotFoundError
    except:
        sm = pystan.StanModel(file=model_code)
        #with open(cache_fn, 'wb') as f:
        #    pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm


import sys

if len(sys.argv) > 1:
    inf_type = sys.argv[1]
else:
    print("Err: Missing inference name")
    exit(-1)
params = False
rebuild = False
if len(sys.argv) > 2:
    if sys.argv[2] == '-r':
        rebuild = True

    if sys.argv[2] == '-p':
        params = True
    elif len(sys.argv) > 3 and sys.argv[3] == '-p':
        params = True
    elif len(sys.argv) > 3 and sys.argv[3] == '-r':
        rebuild = True

sm = StanModel_cache('model.stan', rebuild)
with open('data.json') as dataFile:
    data = json.load(dataFile)

if inf_type == 'sampling':
    fit = sm.sampling(data=data, iter=1000, chains=4)
    print(fit)
    if params:
        acc = fit.sim['samples'][0].sampler_params[0]
        step_sizes = fit.sim['samples'][0].sampler_params[1]
        n_steps = fit.sim['samples'][0].sampler_params[3]
        m_acc = sum(acc) / len(acc)
        m_step_sizes = sum(step_sizes) / len(step_sizes)
        m_n_steps = sum(n_steps) / len(n_steps)
        print("params_avg : " + str(m_acc) + ", " + str(m_step_sizes) + ", " + str(m_n_steps))
        b_acc = [ind for ind, x in enumerate(acc) if x > 0.9][:5]
        b_step_sizes = [step_sizes[n] for n in b_acc]
        b_n_steps = [n_steps[n] for n in b_acc]
        print("params_best: " + str(list(zip(b_step_sizes, b_n_steps))))
        sp.getoutput('echo {0} {1} > params'.format(b_step_sizes[0], int(b_n_steps[0])))

elif inf_type == 'hmc':
    fit = sm.sampling(data=data, iter=1000, chains=4, algorithm='HMC')
    print(fit)
elif inf_type == 'vb':
    fit = sm.vb(data=data, iter=1000)
    print(fit['args']['sample_file'])
