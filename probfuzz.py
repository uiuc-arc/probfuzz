#!/usr/bin/python

import subprocess as sp
import time
import os
import sys
import threading
import Queue as queue

from utils.utils import *
from language.templateparser import TemplatePopulator

# region global config

_TOOL = 'tool'
_ENABLED = 'enabled'
_ALGORITHM = 'algorithm'
_TIMEOUT = 'timeout'
_MODELTYPE = 'model_type'
_CURRENT_MODEL_TYPE = 'LR'  # LR
_STAN_GENERATE_QUANTS = False
_COMPARE = True

# choose fuzzer type
set_fuzzer_type('STR')

generate_special_data = False
generate_special_params = False


# endregion global config


class InferenceEngineRunner(threading.Thread):
    def __init__(self, id, queue, runconfigs):
        threading.Thread.__init__(self)
        self.queue = queue
        self.id = id
        self.runconfigs = runconfigs

    @staticmethod
    def _runTool(config, prog_id, target_directory):
        from backends.stan import Stan
        from backends.edward import Edward
        from backends.pyro import Pyro

        if not config[_ENABLED]:
            return
        if config[_TOOL] == 'stan':
            stan = Stan(target_directory, None, None, None, None, None, None)
            stan.run(config[_ALGORITHM], config[_TIMEOUT], prog_id, config['python'])
        elif config[_TOOL] == 'edward':
            edward = Edward(target_directory, None, None, None, None, None, None, config.get("name", "edward"))
            edward.run(config[_ALGORITHM], config[_TIMEOUT], prog_id, config['python'])
        elif config[_TOOL] == 'pyro':
            pyro = Pyro(target_directory, None, None, None, None, None, None)
            pyro.run(config[_ALGORITHM], config[_TIMEOUT], prog_id, config['python'])

    def run(self):
        while not self.queue.empty():
            args = self.queue.get(block=False)
            for config in self.runconfigs:
                self._runTool(config, args[0] + 1, args[1])


def filterCommonDistributions(models, runconfigurations):
    filteredModels = []
    for model in models:
        i = 1
        for config in runconfigurations:
            if config[_ENABLED] and (config[_TOOL] not in model.keys()):
                i = 0
                break
        if i == 1:
            filteredModels.append(model)
    return filteredModels


def generate_programs_from_template():
    from backends.stan import Stan
    from backends.edward import Edward
    from backends.pyro import Pyro

    config = read_config()

    max_thread = config['max_threads']
    if len(sys.argv) > 1:
        progs = int(sys.argv[1])
        if progs is None:
            print("Err: Program count should be integer")
            exit(-1)
    else:
        print("Err: Missing program count")
        exit(-1)

    # flag to indicate whether to use domain info for filtering
    structured = config['structured']
    current_template = config['templates'][config['current_template']]

    # directory for output
    dirname = config['output_dir'] + "/progs" + str(time.strftime("%Y%m%d-%H%M%S"))

    # initialize queues
    queues = [queue.Queue() for _ in range(0, max_thread)]

    # select models
    models = parse_models()
    all_models = filterCommonDistributions(models, config['runConfigurations'])

    # generate programs using template
    for i in range(0, progs):
        # create program directory
        file_dir_name = dirname + '/prob_rand_' + str(i + 1)
        if not os.path.exists(file_dir_name):
            os.makedirs(file_dir_name)

        template_populator = TemplatePopulator(file_dir_name, current_template, all_models)
        data, priors, distmap, constmap = template_populator.populate(structured)

        # check if model is valid if structured check is enabled
        if structured:
            while not template_populator.validModel:
                data, priors, distmap, constmap = template_populator.populate()


        for toolconfig in config['runConfigurations']:
            if toolconfig[_ENABLED]:
                if toolconfig[_TOOL] == 'stan':
                    stan = Stan(file_dir_name, data, priors, distmap, constmap, current_template, config)
                    stan.create_program()
                    # add the driver
                    sp.Popen(['cp', 'driver.py', file_dir_name], stdout=sp.PIPE).communicate()
                elif toolconfig[_TOOL] == 'edward':
                    edward = Edward(file_dir_name, data, priors, distmap, constmap, current_template, config,
                                    toolconfig.get('name', 'edward'))
                    edward.create_program()
                elif toolconfig[_TOOL] == 'pyro':
                    pyro = Pyro(file_dir_name, data, priors, distmap, constmap, current_template, config)
                    pyro.create_program()

        # enqueue threads
        queues[i % max_thread].put((i, file_dir_name))

    print("Output directory : " + dirname)

    for i in range(0, max_thread):
        print("Starting thread .. " + str(i + 1))
        thread = InferenceEngineRunner(i, queues[i], config['runConfigurations'])
        thread.start()


if __name__ == "__main__":
    generate_programs_from_template()
