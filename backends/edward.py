from backend import Backend
import antlr4
from language.antlr.TemplateLexer import TemplateLexer
from language.antlr.TemplateParser import TemplateParser
import numpy as np
import subprocess as sp
import six

class Edward(Backend):

    def __init__(self, file_dir, data_dict, prior_dict, distmap, constmap, templatefile, config, name):
        super(Edward, self).__init__(file_dir)
        self.output_program = ""
        self.data_dict = data_dict
        self.prior_dict = prior_dict
        self.distmap = distmap
        self.constmap = constmap
        self.templatefile = templatefile
        self.placeholders = dict()
        self.posteriors = list()
        self.tab = ''
        self.name = name
        # choose an inference method
        if config is not None:
            inferences = config['edward']['inferences']
            self.inference = np.random.choice([i for i in inferences if not i['ig']])
            if self.inference.get('params', None) is not None:
                self.param_name = self.inference['params'][0]['name']
                self.param_type = self.inference['params'][0]['type']
            else:
                self.param_name = None
            

    def create_program(self):
        template = antlr4.FileStream(self.templatefile)
        lexer = TemplateLexer(template)
        stream = antlr4.CommonTokenStream(lexer)
        parser = TemplateParser(stream)
        self.visit(parser.template())

        with open(self._directory + '/{0}_prog.py'.format(self.name), 'w') as edwardfile:
            edwardfile.write(self.output_program)

    def visitData(self, ctx=TemplateParser.DataContext):
        name = ctx.ID().getText()
        data_item = self.data_dict[name]
        code=""
        if isinstance(data_item, np.ndarray):
            if data_item.ndim == 1:
                # just extending the dimension to 2d
                code += name +"= np.array(" + str(data_item.tolist()) + \
                                       ", dtype=np.float32).reshape({0},1)\n".format(str(data_item.shape[0]))
            else:
                code += name + "= np.array(" + str(data_item.tolist()) + ", dtype=np.float32)\n"
        else:
            code += name + "=" + str(data_item)+"\n"
        return code

    def visitLrmodel(self, ctx=TemplateParser.LrmodelContext):
        data_placeholder = ctx.ID().getText()+"_ph"
        model = data_placeholder + " = " + self.visit(ctx.distexpr())
        # add placeholders if needed

        for ph in six.iterkeys(self.placeholders):
            if self.placeholders[ph] != '':
                self.output_program += self.placeholders[ph]
                self.placeholders[ph] = ''

        # just storing a placeholder for observe
        self.placeholders[data_placeholder] = ''

        return self.tab+model+"\n"

    def visitDistexpr(self, ctx=TemplateParser.DistexprContext):
        if ctx.DISTHOLE() is not None:
            dist = self.distmap[ctx.DISTHOLE().getSymbol().tokenIndex]
        else:
            dist = self.distmap[ctx.DISTRIBUTION().getSymbol().tokenIndex]
        if ctx.dims() is not None:
            dims = ctx.dims().getText()
            expr = dist['edward'] + '(' + self.visit(ctx.params()) + ', sample_shape=[' +dims+'])'
        else:
            expr = dist['edward'] + '(' + self.visit(ctx.params()) +')'
        return expr

    def visitParams(self, ctx):
        params = ""
        for child in ctx.param():
            params += self.visit(child) + " ,"
        return params[:-1]

    def visitParam(self, ctx):
        if ctx.expr() is not None:
            return self.visit(ctx.expr())
        else:
            return str(self.constmap[ctx.CONSTHOLE().getSymbol().tokenIndex])

    def visitRef(self, ctx):
        name = ctx.getText()
        if name in self.data_dict:
            data_item = self.data_dict[name]
            if isinstance(data_item, np.ndarray):
                shape = data_item.shape
                phname = name+"_ph"
                if phname in self.placeholders:
                    return phname
                if data_item.ndim == 1:
                    # broadcasting
                    self.placeholders[phname] = phname + "=tf.placeholder(tf.float32, [" + str(shape[0])+ ",1])\n"
                else:
                    self.placeholders[phname] = phname +"=tf.placeholder(tf.float32, "+str(shape).replace('(', '[').replace(')', ']') + ")\n"
                return phname
        return ctx.getText()

    def visitVal(self, ctx):
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
                return 'tf.matmul('+self.visit(ctx.getChild(0))+','+self.visit(ctx.getChild(2))+')'
        else:
            return self.visit(ctx.getChild(0)) + ctx.AOP().getText() + self.visit(ctx.getChild(2))

    def visitPrior(self, ctx):
        name = ctx.ID().getText()
        prior = self.prior_dict[name]
        code = ""
        if prior.has_key('dims'):
            code += name + "=" + prior['prior']['edward']+"("
            dims = prior['dims']

            for arg in prior['args']:
                # adjusting for dimension
                if dims.count(',') == 0:
                    code+= "tf.ones(("+str(dims)+", 1))*" +str(arg) +" ,"
                else:
                    code += "tf.ones((" + str(dims) + "))*" + str(arg) + " ,"

            code = code[:-1] + ")\n"

        else:
            code+= name + "=" + prior['prior']['edward'] + "("
            for arg in prior['args']:
                code += "tf.ones(1)*" + str(arg) + " ,"
            code = code[:-1] + ")\n"
        return code

    def visitTemplate(self, ctx):
        # add imports
        self.output_program+= "import edward as ed, tensorflow as tf, numpy as np\n"

        for child in ctx.children:
            code = self.visit(child)
            self.output_program+=code

        # run inference code
        posterior_map = ",".join(['{0} : q_{0}'.format(p) for p in self.posteriors])
        placeholder_map = ",".join(['{0}_ph : {0}'.format(ph.split('_')[0]) for ph in six.iterkeys(self.placeholders)])

        inference_str = 'inference = {0}({1}, data={2})'.format(self.inference['name'],
                                                                                         "{" + posterior_map + "}", "{" + placeholder_map + "}")
        from utils.utils import *
        self.output_program+=inference_str+"\n"
        if self.param_name is not None:
            param = generate_primitives(self.param_type, 1, False)[0]
            self.output_program += "inference.run({0})\n".format(self.param_name + "=" + str(param))
        else:
            self.output_program+="inference.run()\n"

        # fetch the posteriors
        supported_posterior = self.inference['supported_posterior']
        for posterior in self.posteriors:
            if supported_posterior == 'ed.models.Empirical':
                eval_string = "print({0}.params.eval()[100:1000:10].mean(0))".format("q_"+posterior)
                self.output_program+=eval_string+"\n"
            else:
                eval_string = "print({0}.eval())".format("q_" + posterior)
                self.output_program+=eval_string+"\n"

    def visitQuery(self, ctx=TemplateParser.QueryContext):
        supported_posterior = self.inference['supported_posterior']
        param = ctx.ID().getText()
        if 'dims' in self.prior_dict[param]:
            dim = self.prior_dict[param]['dims']
            if dim.count(',') == 0:
                dim += ', 1'
        else:
            dim = '1'
        if supported_posterior == 'ed.models.Empirical':
            post = _posterior_string_empirical.format(param, dim)
        elif supported_posterior == 'ed.models.Normal':
            post = _posterior_string_normal.format(param, dim)
        elif supported_posterior == 'ed.models.PointMass':
            post = _posterior_string_pointmass.format(param, dim)
        self.posteriors.append(param)

        return post+"\n"

    def visitAssign(self, ctx=TemplateParser.AssignContext):
        name = ctx.ID().getText()
        distexpr = ctx.distexpr()
        if distexpr.DISTHOLE() is not None:
            dist = self.distmap[distexpr.DISTHOLE().getSymbol().tokenIndex]
        else:
            dist = self.distmap[distexpr.DISTRIBUTION().getSymbol().tokenIndex]
        return self.tab+name + "=" +self.visit(distexpr)+"\n"

    def visitCondmodel(self, ctx=TemplateParser.CondmodelContext):
        code = ""
        code+=self.tab+"if "+self.visit(ctx.expr())+" is True:\n"
        self.tab+='\t'
        code +=self.visit(ctx.model(0))
        self.tab = self.tab[:-1]
        code+=self.tab+"else:\n"
        self.tab+='\t'
        code+=self.visit(ctx.model(1))
        self.tab = self.tab[:-1]
        return code

    def isScalar(self, ctx):
        if isinstance(ctx, TemplateParser.RefContext):
            ref = ctx.getText()
            if (ref in self.data_dict and not isinstance(self.data_dict[ref], np.ndarray)) or (ref in self.prior_dict and not 'dims' in self.prior_dict[ref]):
                return True
        elif isinstance(ctx, TemplateParser.ValContext):
            return True

        return False

    def run(self, algorithm, timeout, prog_id, python_cmd):
        print("Running Edward program " + str(prog_id) + " >>>>")
        process = sp.Popen("cd " + self._directory + "; timeout {0} {1} edward_prog.py".format(timeout, python_cmd),
            stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
        dataout, dataerr = process.communicate()

        with open(self._directory + '/{0}out_'.format(self.name) + str(prog_id), 'w') as outputfile:
            outputfile.write(dataerr)
            outputfile.write(dataout)

        print("Done Edward program " + str(prog_id) + " >>>>")

_inferences =  [
        {
            "name": "ed.HMC",
            "supported_posterior": "ed.models.Empirical",
            "datasets": 2,
            "ig": False,
            "lr": 0.05,
            "thin": True,
            "mean": True,
            "ig": True
        },
        {
            "name": "ed.MAP",
            "supported_posterior": "ed.models.PointMass",
            "datasets": 1,
            "ig": False
        },
        {
            "name": "ed.KLqp",
            "supported_posterior": "ed.models.Normal",
            "ig": False,
            "supported_models": ["ed.models.Normal", "ed.models.Exponential", "ed.models.InverseGamma"]
        },
        {
            "name": "ed.KLpq",
            "supported_posterior": "ed.models.Normal",
            "ig": False,
            "supported_models": ["ed.models.Normal", "ed.models.Exponential", "ed.models.InverseGamma"]
        },
        {
            "name": "ed.Gibbs",
            "supported_posterior": "ed.models.Empirical",
            "ig": True,
            "supported_models": ["ed.models.Normal"]
        },
        {
            "name": "ed.MetropolisHastings",
            "supported_posterior": "ed.models.Empirical",
            "ig": True,
            "has_proposal": True
        },
        {
            "name": "ed.SGHMC",
            "supported_posterior": "ed.models.Empirical",
            "ig": False,
            "has_proposal": False,
            "lr": 0.01,
            "thin": True,
            "mean" : True
        },
        {
            "name": "ed.SGLD",
            "supported_posterior": "ed.models.Empirical",
            "ig": False,
            "has_proposal": False,
            "lr": 0.05,
            "thin": True,
            "mean": True
        }
    ]

_posterior_string_empirical = 'q_{0} = ed.models.Empirical(params=tf.Variable(tf.random_normal([1000, {1}])))'
_posterior_string_normal = 'q_{0} = ed.models.Normal(tf.Variable(tf.random_normal([{1}])), ' \
                          'tf.nn.softplus(tf.Variable(tf.random_normal([{1}]))))'
_posterior_string_pointmass = 'q_{0} = ed.models.PointMass(params=tf.Variable(tf.random_normal([{1}])))'

