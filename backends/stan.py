import subprocess as sp
import six

from backend import Backend
from utils.utils import *
from language.antlr.TemplateParser import TemplateParser
import antlr4
from language.antlr.TemplateLexer import TemplateLexer

import numpy as np


class Stan(Backend):
    def __init__(self, file_dir, data_dict, prior_dict, distmap, constmap, templatefile, config):
        super(Stan, self).__init__(file_dir)
        self.output_program = " "
        self.data_dict = data_dict
        self.prior_dict = prior_dict
        self.distmap = distmap
        self.constmap = constmap
        self.templatefile = templatefile
        self.data_json = {}
        self.index_map = {}
        self.data_init = False
        self.parameter_init = False
        self.model_init = False
        self.model_string = ""
        self.quants_string = ""
        self.quants_init = False
        self.mode = []
        self.config = config

    def visitData(self, ctx=TemplateParser.DataContext):
        self.mode.append('data')
        if self.data_init is False:
            self.output_program += "data {\n}"
            self.data_init = True
        self.output_program = self.output_program[:-1]  # removing braces
        name = ctx.ID().getText()
        data_item = self.data_dict[name]
        if isinstance(data_item, np.ndarray):
            indexmap = []
            indexstr = "["
            for dim in data_item.shape:
                dimname = get_new_var_name(name)
                self.data_json[dimname] = dim
                self.output_program += "int " + dimname + ";\n"
                indexmap.append(dimname)
                indexstr += dimname + ","
            self.index_map[name] = indexmap
            self.data_json[name] = data_item
            if len(data_item.shape) == 1:
                self.output_program += "vector" + indexstr[:-1] + "] " + name + ";\n"
            else:
                self.output_program += "matrix" + indexstr[:-1] + "] " + name + ";\n"
        else:
            self.data_json[name] = data_item
            if isinteger(data_item):
                self.output_program += "int " + name + ";\n"
            else:
                self.output_program += "real " + name + ";\n"
        self.output_program += "}"

    def visitTemplate(self, ctx=TemplateParser.TemplateContext):
        for child in ctx.children:
            self.visit(child)

    def visitPrior(self, ctx=TemplateParser.PriorContext):
        self.mode.append('prior')
        if self.parameter_init is False:
            self.output_program += "\nparameters {\n}"
            self.parameter_init = True
        self.output_program = self.output_program[:-1]
        name = ctx.ID().getText()
        prior = self.prior_dict[name]
        if prior['prior']['stan'] == 'dirichlet':
            assert ('dims' in prior)
            self.output_program += 'simplex[' + str(prior['dims']) + "] " + name + ";\n"
        else:
            if prior.has_key('dims'):
                self.output_program += 'vector<lower=0>[' + str(prior['dims']) + "] " + name + ";\n"
            else:
                self.output_program += 'real ' + name + ";\n"
        self.output_program += "}"

        if self.model_init is False:
            self.model_string = "\nmodel {\n}"
            self.model_init = True
        self.model_string = self.model_string[:-1]
        self.model_string += name + " ~ " + prior['prior']['stan'] + '('

        for arg in prior['args']:
            self.model_string += str(arg) + ','

        self.model_string = self.model_string[:-1]
        self.model_string += ");\n}"

    def visitLrmodel(self, ctx=TemplateParser.LrmodelContext):
        self.mode.append('model')
        self.model_string = self.model_string[:-1]
        self.model_string += ctx.ID().getText() + " ~ " + self.visit(ctx.distexpr()) + ";"
        self.model_string += "\n}"
        self.mode.pop(-1)

    def visitAssign(self, ctx=TemplateParser.AssignContext):
        name = ctx.ID().getText()
        distexpr = ctx.distexpr()
        self.model_string = self.model_string[:-1]

        if distexpr.DISTHOLE() is not None:
            dist = self.distmap[distexpr.DISTHOLE().getSymbol().tokenIndex]
        else:
            dist = self.distmap[distexpr.DISTRIBUTION().getSymbol().tokenIndex]

        if isintegertype(dist['support']):
            self.model_string = self.model_string.replace("{\n", '{\nint ' + name + '=0;\n')

        else:
            self.model_string = self.model_string.replace("{\n", '{\nreal ' + name + '=0.0;\n')
        self.model_string += name + '~' + self.visit(distexpr) + ';\n'
        self.model_string += '}'

    def visitDistexpr(self, ctx):
        expr = ""
        if ctx.DISTHOLE() is not None:
            dist = self.distmap[ctx.DISTHOLE().getSymbol().tokenIndex]
        else:
            dist = self.distmap[ctx.DISTRIBUTION().getSymbol().tokenIndex]
        expr += dist['stan'] + '(' + self.visit(ctx.params()) + ')'
        return expr

    def visitParams(self, ctx):
        params = ""
        for child in ctx.param():
            params += self.visit(child) + " ,"
        return params[:-1]

    def visitParam(self, ctx=TemplateParser.ParamContext):
        if ctx.expr() is not None:
            return self.visit(ctx.expr())
        else:
            return str(self.constmap[ctx.CONSTHOLE().getSymbol().tokenIndex])

    def visitRef(self, ctx):
        return ctx.getText()

    def visitVal(self, ctx):
        return ctx.getText()

    def visitArith(self, ctx=TemplateParser.ArithContext):
        return self.visit(ctx.getChild(0)) + ctx.AOP().getText() + self.visit(ctx.getChild(2))

    def visitCondmodel(self, ctx=TemplateParser.CondmodelContext):
        self.model_string = self.model_string[:-1]
        self.model_string += 'if (' + self.visit(ctx.expr()) + ') {\n}'
        self.visit(ctx.model(0))
        self.model_string += 'else{\n}'
        self.visit(ctx.model(1))
        self.model_string += "\n}"

    def currentMode(self):
        if len(self.mode) > 0:
            return self.mode[-1]
        else:
            return ""

    def add_quants(self):
        quants = "generated quantities {\n"
        decl = ""
        rng_calls = ""
        # choose rng

        for p in self.prior_dict:
            prior = self.prior_dict[p]
            if prior.has_key('dims'):
                rng = [m for m in self._rngs if m['args'][0]['type'] == 'vector'][0]
                decl += rng["return"] + '[' + str(prior['dims']) + "] " + p + '_q' + ";\n"
                rng_calls += p + '_q <- ' + rng["name"] + "({0});\n".format(p)

            else:
                rng = [m for m in self._rngs if m['args'][0]['type'] == 'f'][0]
                decl += 'real ' + p + '_q' + ";\n"
                rng_calls += p + '_q <- ' + rng["name"] + "({0},".format(p)
                for i in range(1, len(rng["args"])):
                    rng_calls += str(generate_primitives(rng["args"][i]["type"], 1, False)[0]) + ","
                rng_calls = rng_calls[:-1] + ");\n"
        quants += decl + rng_calls + "}\n"
        return quants

    def create_program(self):
        template = antlr4.FileStream(self.templatefile)
        lexer = TemplateLexer(template)
        stream = antlr4.CommonTokenStream(lexer)
        parser = TemplateParser(stream)
        self.visit(parser.template())

        # write data
        with open(self._directory + '/data.json', 'w') as datafile:
            for k in six.iterkeys(self.data_json):
                if isinstance(self.data_json[k], np.ndarray):
                    self.data_json[k] = self.data_json[k].tolist()
            json.dump(self.data_json, datafile)

        with open(self._directory + '/model.stan', 'w') as modelfile:
            modelfile.write(self.output_program)
            modelfile.write(self.model_string)
            try:
                if self.config["stan"]["quants"] is True:
                    modelfile.write(self.add_quants())
            except:
                pass

    def run(self, algorithm, timeout, prog_id, python_cmd):

        print("Running Stan program " + str(prog_id) + " >>>>")
        if algorithm == 'nuts':
            process = sp.Popen(
                "cd " + self._directory + "; timeout {0} {1} driver.py sampling -p".format(timeout, python_cmd),
                stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
            dataout, dataerr = process.communicate()

            with open(self._directory + '/pplout_nuts_' + str(prog_id), 'w') as outputfile:
                outputfile.write(dataerr)
                outputfile.write(dataout)

        elif algorithm == 'hmc':
            process = sp.Popen(
                "cd " + self._directory + "; timeout {0} {1} driver.py hmc -p".format(timeout, python_cmd),
                stdout=sp.PIPE, stderr=sp.PIPE, shell=True)

            dataout, dataerr = process.communicate()
            with open(self._directory + '/pplout_nuts_' + str(prog_id), 'w') as outputfile:
                outputfile.write(dataerr)
                outputfile.write(dataout)

        print("Done Stan program " + str(prog_id) + " >>>>")

    _rngs = [
        {
            "name": "dirichlet_rng",
            "args": [
                {
                    "name": "alpha",
                    "type": "vector"
                }
            ],
            "return": "simplex"
        },
        {
            "name": "normal_rng",
            "args": [
                {
                    "name": "mu",
                    "type": "f"
                },
                {
                    "name": "sigma",
                    "type": "f+"
                },
            ],
            "return": "f"
        },

    ]
