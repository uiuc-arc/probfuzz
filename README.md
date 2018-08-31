Probfuzz
================================================================================

## What is Probfuzz?  

ProbFuzz is a tool for testing Probabilistic Programming Systems(PPS). It encodes a probabilistic language grammar from which it systematically generates probabilistic programs. ProbFuzz has language specific translators to transform the generated program into versions which use the API of individual systems. Then it finds bugs by differential testing of the given systems. The developer writes the templates of probabilistic models in an intermediate language with holes, which represent missing distributions, parameters or data. ProbFuzz generates the probabilistic models by completing the holes in the template with specific choices. Currently, ProbFuzz has backends for 3 languages (Stan, Edward and Pyro), which translate the completed program into a valid program for each system. Finally, it runs them and computes metrics to reveal bugs.

## Project Structure

		.  
		├── language					# Contains the language grammar and templates  
		├── backends					# Contains the model translators for PPS  
		├── utils					# General utility functions  						
		├── probfuzz.py					# Probfuzz tool which controls the tool flow  
		├── README.md					# README for basic info  
		├── models.json					# Contains information for all valid distributions, their arguments and support  
		├── config.json					# Configuration for the tool 

## How to install?  

Install dependencies:  

```
sudo ./install.sh
```
It should print ```Install successful```

Run probfuzz:  
```
./probfuzz.py [#programs]
```
Check program output in ```output``` folder

##  Running the tool

1. Install the dependencies for probfuzz as mentioned in "How to
install"

2.  Run ProbFuzz: `./probfuzz.py 5`. This will take around 5-10
minutes. The argument, '5' tells ProbFuzz to generate 5 programs for
fuzzing in each supported PPS. You can test with some other value as
well. A directory called 'output' will be created to contain the
output of this and all subsequent runs. For each run, a directory
named 'progsXXX' will be created under 'output' where you will see 5
directories ('prob_rand_X' for X in {1, 2, 3, 4, 5}) which contain
programs for each PPS (Stan, Edward and Pyro) together with files
containing the output of running those programs.

### Details of output

Stan programs are split into several files: "model.stan" which
contains the model, "data.json" which contains the datasets and
"driver.py", which is the python script to run the code. The output
from running a generated Stan program is called pplout_*. For Edward,
the generated program is in "edward_prog.py" and the output from
running the program is stored in a file called "edwardout_X". For
Pyro, the generated program is in "pyro_prog.py" and the output is
stored in "pyroout_X". The respective output files contain stderr and
stdout output from running the generated programs, including such
information as the values of the parameters.

The output from each PPS has a specific format. For Stan, the
output format would look like the following, where "mean" is
expected value of the parameters and "lp__" is the
log_probability error:


```
       mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
w      5.14    0.04   0.81   3.67   4.58   5.11   5.66   6.81  448.0   1.01
b      0.34  1.2e-3   0.04   0.28   0.32   0.34   0.37   0.42  854.0    1.0
p      0.74  5.5e-3   0.12   0.53   0.66   0.74   0.82   0.98  451.0   1.01
lp__ -14.73    0.05   1.28 -18.09 -15.32  -14.4 -13.82 -13.32  615.0   1.01
```

For Edward, the output would look like the following, where each
row indicates the means of the parameters 'w', 'b' and 'p'.:

```
...
[9.882316]  
[40768.516]  
[27.659203]
```

Finally for Pyro, the output would look like the following, where *_v
is the variance and *_w is the mean of each parameter in the model.:

```
...
p_v : [8.048228]  
p_w : [-3833.4492]  
w_w : [-0.3105538]  
w_v : [-7.687871]  
b_v : [10.014385]  
b_w : [0.94321126]  
```