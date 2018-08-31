from backend import Backend
import antlr4
from language.antlr.TemplateLexer import TemplateLexer
from language.antlr.TemplateParser import TemplateParser
import subprocess as sp
import six
from utils.utils import *


class Pyro(Backend):
    def __init__(self, file_dir, data_dict, prior_dict, distmap, constmap, templatefile, config):
        super(Pyro, self).__init__(file_dir)
        self.output_program = ""
        self.data_dict = data_dict
        self.prior_dict = prior_dict
        self.distmap = distmap
        self.constmap = constmap
        self.templatefile = templatefile
        self.output_program=""
        self.model = ""
        self.guide = ""
        self.module = ""
        self.prior_post_map = dict()
        self.tab = ""
        self.forward=""
        self.forward_ref_string = ""
        self.data_string = ""
        self.config = config
        self.posteriors = {}
        self.queries = ""


    def create_program(self):
        template = antlr4.FileStream(self.templatefile)
        lexer = TemplateLexer(template)
        stream = antlr4.CommonTokenStream(lexer)
        parser = TemplateParser(stream)
        self.visit(parser.template())
        self.output_program = self.output_program.replace("\t", "    ")
        with open(self._directory + '/pyro_prog.py', 'w') as pyrofile:
            pyrofile.write(self.output_program)

    def run(self, algorithm, timeout, prog_id, python_cmd):
        print("Running Pyro program " + str(prog_id) + " >>>>")
        process = sp.Popen("cd " + self._directory + "; timeout {0} {1} pyro_prog.py".format(timeout, python_cmd),
                           stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
        dataout, dataerr = process.communicate()

        with open(self._directory + '/pyroout_' + str(prog_id), 'w') as outputfile:
            outputfile.write(dataerr)
            outputfile.write(dataout)

        print("Done Pyro program " + str(prog_id) + " >>>>")

    def visitTemplate(self, ctx):
        # imports
        self.output_program += "import pyro, numpy as np, torch, pyro.distributions as dist, torch.nn as nn\n" \
                               "from pyro.optim import Adam\n"\
                               "from pyro.infer import SVI\n"\
                                "if pyro.__version__=='0.2.1': from pyro.infer import Trace_ELBO\n"\
                                "from torch.autograd import Variable\n"\
                                "from torch.nn.parameter import Parameter\n"
        # fix seeds
        self.output_program += "torch.manual_seed(77349895)\n"\
                              "np.random.seed(77349895)\n"

        for child in ctx.children:
            code = self.visit(child)


        self.output_program += self.createModule()
        self.output_program += "def model(" + self.data_string[:-1] + "):\n"+"\t"+self.model.replace("\n", "\n\t")
        self.output_program = self.output_program[:-1]

        # create map for guide and wrap up
        map = "priors = {"
        for p in self.prior_post_map:
            if p == 'p':
                continue
            map += "'{0}' : {1},".format(p, self.prior_post_map[p])
        self.guide += map[:-1] + "}\n"
        self.guide += "lifted_module = pyro.random_module('module', module, priors)\nreturn lifted_module()\n"

        self.output_program += "def guide(" + self.data_string[:-1] + "):\n"+"\t"+self.guide.replace("\n", "\n\t")

        self.output_program = self.output_program[:-1]
        self.output_program += self.writeInference()
        self.output_program += self.queries

    def visitData(self, ctx=TemplateParser.DataContext):
        name = ctx.ID().getText()
        data_item = self.data_dict[name]
        code=""
        if isinstance(data_item, np.ndarray):
            if data_item.ndim == 1:
                # just extending the dimension to 2d
                code += name + "= np.array(" + str(data_item.tolist()) + \
                                       ", dtype=np.float32).reshape({0},1)\n".format(str(data_item.shape[0]))
            else:
                code += name + "= np.array(" + str(data_item.tolist()) + ", dtype=np.float32)\n"
            code += name + "=Variable(torch.Tensor(" + name + "))\n"
        else:
            code += name + "=" + str(data_item)+"\n"
            code += name + "=Variable(torch.Tensor([" + name + "]))\n"

        self.output_program += code
        return code

    def visitPrior(self, ctx):
        name = ctx.ID().getText()
        prior = self.prior_dict[name]
        # add prior to model
        if prior.has_key('dims'):
            dims = prior['dims']
            if dims.count(',') == 0:
                dims+=",1"
        else:
            dims ="1"
        argslist = ''
        for arg in prior['args']:
            argslist+="Variable("+str(arg)+"*torch.ones(("+dims+"))),"

        # hack for sample
        if name == 'p':
            self.model += "{0} = pyro.sample('{1}', {2}({3}))\n".format(name, name, prior['prior']['pyro'], argslist[:-1])
        else:
            self.model += "{0} = {1}({2})\n".format(name, prior['prior']['pyro'], argslist[:-1])

        # update guide
        # get posterior dist with matching support
        posterior_candidates = getSupportedDistributions(prior["prior"]["support"], pps="pyro")
        posterior = np.random.choice([p for p in posterior_candidates if p['type'] == 'C' and p["name"] in ["normal", "gamma", "beta"]])
        arglist = ''
        args = []
        for arg in posterior["args"]:
            argname = get_new_var_name("arg_")
            self.guide += "{0} = Variable(torch.randn(({1})), requires_grad=True)\n".format(argname, dims)
            if is_positive(arg["type"]):
                arglist+="torch.nn.Softplus()(pyro.param('{0}', {0})),".format(argname)
            else:
                arglist += "pyro.param('{0}', {0}),".format(argname)
            args.append(argname)

        param_var = name + "_dist"
        if name == 'p':
            self.guide += "{0}_dist = pyro.sample('{1}', {2}({3}))\n".format(name,
                                                                             name,
                                                                             posterior["pyro"],
                                                                             arglist[:-1])
        else:
            self.guide += param_var + "={0}({1})\n".format(posterior["pyro"], arglist[:-1])
        self.posteriors[name] = {"dist": posterior, "args": args}
        self.prior_post_map[name] = param_var

    def visitLrmodel(self, ctx):
        name = ctx.ID().getText()
        # fill the forward function
        model = self.visit(ctx.distexpr())
        paramlist = "self,"+self.forward_ref_string
        self.forward = "\tdef forward("+paramlist[:-1]+"):\n"
        if ctx.distexpr().dims() is not None:
            dims = ctx.distexpr().dims().getText()
        else:
            dims = "1"

        for priorname in self.prior_dict:
            self.forward += "\t\t{0} = self.{0}\n".format(priorname)

        self.forward += "\t\toutput = " + self.visit(ctx.distexpr().params().param(0)) + "*Variable(torch.ones((" + dims +")))\n"
        self.forward += "\t\treturn output\n"

        # get the priors map
        self.model += "priors = {"
        for priorname in self.prior_dict:
            if priorname == 'p':
                continue
            self.model += "'{0}' : {0},".format(priorname)
        self.model += "}\n"

        # lift module
        lifted_module_name = get_new_var_name("lifted_module")
        self.model += lifted_module_name+"=pyro.random_module('module', module, priors)\n"
        lifted_reg_name = get_new_var_name("lifted_reg")
        self.model += lifted_reg_name + "="+lifted_module_name+"()\n"
        self.data_string = name + "," + self.forward_ref_string

        self.model += "prediction="+lifted_reg_name+"(" + self.forward_ref_string[:-1]+").squeeze(-1)\n"

        if len(model[1]["args"]) == 1:
            self.model += "pyro.sample('obs', " + model[1]["pyro"] + "(prediction), obs=" + name + \
                              ".squeeze(-1))\n"
        else:
            self.model += "pyro.sample('obs', " + model[1]["pyro"] + "(prediction, " + self.visit(ctx.distexpr().params().param(1)) + "*Variable(torch.ones("+name+".squeeze(-1).size()))), obs=" + name + ".squeeze(-1))\n"
        return ""

    def visitCondmodel(self, ctx):
        code = "if torch.equal({0}, Variable(torch.Tensor([1.0]))):\n".format(self.visit(ctx.expr()))
        truebranch = ctx.model(0).lrmodel()
        falsebranch = ctx.model(1).lrmodel()
        model = self.visit(truebranch.distexpr())
        paramlist = "self," + self.forward_ref_string

        self.forward = "\tdef forward(" + paramlist[:-1] + "):\n"+self.forward
        if truebranch.distexpr().dims() is not None:
            dims = truebranch.distexpr().dims().getText()
        else:
            dims = "1"

        for priorname in self.prior_dict:
            self.forward += "\t\t{0} = self.{0}\n".format(priorname)

        self.forward += "\t\tif torch.equal({0}, Variable(torch.Tensor([1.0]))):\n".format(self.visit(ctx.expr()))

        # true branch
        self.forward += "\t\t\toutput = " + self.visit(truebranch.distexpr().params().param(0)) + "*Variable(torch.ones((" + dims + ")))\n"
        self.forward+= "\t\telse:\n"

        # false branch
        self.forward += "\t\t\toutput = " + self.visit(
            falsebranch.distexpr().params().param(0)) + "*Variable(torch.ones((" + dims + ")))\n"
        self.forward += "\t\treturn output\n"

        # get the priors map
        self.model += "priors = {"
        for priorname in self.prior_dict:
            if priorname == 'p':
                continue
            self.model += "'{0}' : {0},".format(priorname)
        self.model += "}\n"

        # lift module
        lifted_module_name = get_new_var_name("lifted_module")
        self.model += lifted_module_name + "=pyro.random_module('module', module, priors)\n"
        lifted_reg_name = get_new_var_name("lifted_reg")
        self.model += lifted_reg_name + "=" + lifted_module_name + "()\n"

        name = truebranch.ID().getText()
        self.data_string = name + "," + self.forward_ref_string
        self.model += "prediction=" + lifted_reg_name + "(" + self.forward_ref_string[:-1] + ").squeeze(-1)\n"

        self.model += "pyro.sample('obs', " + model[1]["pyro"] + "(prediction, " + self.visit(
            truebranch.distexpr().params().param(1)) + "*Variable(torch.ones(" + name + ".squeeze(-1).size()))), obs=" + name + \
                      ".squeeze(-1))\n"

        return ""

    def visitAssign(self, ctx):
        name = ctx.ID().getText()
        distexpr = ctx.distexpr()
        if distexpr.DISTHOLE() is not None:
            dist = self.distmap[distexpr.DISTHOLE().getSymbol().tokenIndex]
        else:
            dist = self.distmap[distexpr.DISTRIBUTION().getSymbol().tokenIndex]

        self.forward += "\t\t"+name+"="+self.visit(distexpr)[0]+".sample()\n"
        return ""

    def visitQuery(self, ctx):
        name = ctx.ID().getText()
        output = ""
        try:
            posterior = self.posteriors[name]
            if len(posterior["args"]) == 2:
                output += "print('{0}_mean', {1}({2}, {3}).mean if pyro.__version__=='0.2.1' else {1}({2}, {3}).analytic_mean())\n".\
                    format(name, posterior["dist"]["pyro"],
                           self._cast(posterior["dist"]["args"][0], posterior["args"][0]),
                           self._cast(posterior["dist"]["args"][1], posterior["args"][1]))
            else:
                output += "print('{0}_mean', {1}({2}).mean if pyro.__version__=='0.2.1' else {1}({2}).analytic_mean())\n".\
                    format(name,
                           posterior["dist"]["pyro"],
                           self._cast(posterior["dist"]["args"][0], posterior["args"][0]))
        except:
            print(name + " not found")
        self.queries += output

    @staticmethod
    def _cast(arg, val):
        if is_positive(arg["type"]):
            return "torch.nn.Softplus()(pyro.param('{0}'))".format(val)
        else:
            return "pyro.param('{0}')".format(val)

    def visitDistexpr(self, ctx):
        if ctx.DISTHOLE() is not None:
            dist = self.distmap[ctx.DISTHOLE().getSymbol().tokenIndex]
        else:
            dist = self.distmap[ctx.DISTRIBUTION().getSymbol().tokenIndex]
        params = self.visit(ctx.params())
        expr = dist['pyro'] + '(' + params + ')'
        return expr, dist, params

    def visitParams(self, ctx):
        params = ""
        dims = ctx.parentCtx.dims()
        if dims is not None:
            dim_string = 'Variable(torch.ones((' + dims.getText() + ')))'
        else:
            dim_string = 'Variable(torch.ones(1))'
        for child in ctx.param():
            params += self.visit(child) + "*" + dim_string + ","

        return params[:-1]

    def visitParam(self, ctx):
        if ctx.expr() is not None:
            return self.visit(ctx.expr())
        else:
            return str(self.constmap[ctx.CONSTHOLE().getSymbol().tokenIndex])

    def visitRef(self, ctx):
        ref = ctx.getText()
        if ref in self.data_dict and ref+"," not in self.forward_ref_string:
            self.forward_ref_string += ref + ","

        return ref

    def visitVal(self, ctx):
        if ctx.number().CONSTHOLE() is not None:
            return str(self.constmap[ctx.number().CONSTHOLE().getSymbol().tokenIndex])
        else:
            return ctx.getText()

    def visitArith(self, ctx=TemplateParser.ArithContext):
        operation = ctx.AOP().getText()
        lop = ctx.getChild(0)
        rop = ctx.getChild(2)
        if operation == '*':
            # check if one of them is scalar
            if self.isScalar(lop) or self.isScalar(rop):
                return self.visit(lop) + '*' + self.visit(rop)
            else:
                return self.visit(ctx.getChild(0)) + '.matmul(' + self.visit(ctx.getChild(2)) + ')'
        else:
            return self.visit(ctx.getChild(0)) + ctx.AOP().getText() + self.visit(ctx.getChild(2))

    def isScalar(self, ctx):
        if isinstance(ctx, TemplateParser.RefContext):
            ref = ctx.getText()
            if (ref in self.data_dict and not isinstance(self.data_dict[ref], np.ndarray)) or (ref in self.prior_dict and not 'dims' in self.prior_dict[ref]):
                return True
        elif isinstance(ctx, TemplateParser.ValContext):
            return True

        return False

    def createModule(self):
        initfunc="\t\tsuper(Module, self).__init__()\n"
        for priorname in six.iterkeys(self.prior_dict):
            prior = self.prior_dict[priorname]
            if prior.has_key('dims'):
                dims = prior['dims']
                if dims.count(',') == 0:
                    dims += ",1"
            else:
                dims = "1"

            initfunc+="\t\tself."+priorname+" = Parameter(torch.Tensor(({0})))\n".format(dims)

        module = "class Module(nn.Module):\n"+"\tdef __init__(self):\n"+initfunc+self.forward

        module += "module = Module()\n"
        return module

    def writeInference(self):

        try:
            optim = self.config["pyro"]["optimizers"][0]
            params = ""
            assert len(optim["params"]) > 0
            for param in optim["params"]:
                if "size" not in param:
                    params += "\"{0}\" : {1},".format(param["name"], generate_primitives(param["type"], 1, param["special"])[0])
                else:
                    params += "\"{0}\" : {1},".format(param["name"], tuple(generate_primitives(param["type"], param["size"], param["special"])))

            code = "optim = {0}({{ {1} }})\n".format(optim["name"], params[:-1])
        except Exception:
            code = "optim = Adam({'lr': 0.05})\n"
        code += "svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__=='0.2.1' else 'ELBO')\n"
        code += "for i in range(1000):\n"
        code += "\tloss = svi.step("+self.data_string[:-1]+")\n"
        code += "\tif ((i % 1000) == 0):\n"
        code += "\t\tprint(loss)\n"
        code += "for name in pyro.get_param_store().get_all_param_names():\n"
        code += "\tprint(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))\n"
        return code
