from antlr.TemplateListener import TemplateListener
from antlr4 import *
from antlr4.TokenStreamRewriter import TokenStreamRewriter
from antlr.TemplateParser import *
import numpy as np
from myvisitor import MyVisitor
from utils.utils import *


class MyListener(TemplateListener):

    def __init__(self, models, parser, structured=True):
        self.program = ""
        self.data = dict()
        self.priors = dict()
        self.distMap = dict()
        self.constMap = dict()
        self.models = models
        self.mode = []
        self.structured = structured

    def enterData(self, ctx=TemplateParser.DataContext):

        if not ctx.dtype() is None:
            # create random data
            if ctx.dtype().primitive().getText() == 'float':

                name = ctx.ID().getText()

                if ctx.dtype().dims() is not None:
                    # array
                    size = [int(x.getText()) for x in ctx.dtype().dims().INT()]
                    xdata = np.random.uniform(0, 100, size)
                    self.data[name] = xdata
                else:
                    # scalar
                    value = np.random.uniform(0, 10)
                    self.data[name] = value
        else:
            # evaluate expression
            myvisitor = MyVisitor(self.data)
            tempid = myvisitor.visit(ctx.expr())
            self.data[ctx.ID().getText()] = self.data[tempid]

    def enterPrior(self, ctx=TemplateParser.PriorContext):
        name = ctx.ID().getText()

        if ctx.distexpr().children[0].getText() == 'DIST':
            # check if all arguments are scalars or vectors
            params = ctx.distexpr().params()
            isscalar = True
            for i in range(0, len(params.param())):
                param = params.param(i)
                if param.expr() is not None:
                    # check what is the reference for and whether its an array
                    ref = param.expr().ID().getText()
                    if ref in self.data and isinstance(self.data[ref], np.ndarray):
                        # break if atleast one is vector
                        isscalar = False
                        break

            # choose a distribution
            if isscalar:
                prior = np.random.choice([m for m in self.models if m['type'] == 'C' and not self._dist_has_vector(m)])
            else:
                prior = np.random.choice([m for m in self.models if m['type'] == 'C' and self._dist_has_vector(m)])

            arglist = []
            args = len(prior['args'])
            for i in range(0, args):
                arg = prior['args'][i]
                if ctx.distexpr().params().param(i).getText() == "CONST":
                    v = generate_primitives(arg['type'] if self.structured else 'f', 1, False)[0]
                    arglist.append(v)
                elif ctx.distexpr().params().param(i).expr() is not None:
                    ref = ctx.distexpr().params().param(i).expr().ID().getText()
                    arglist.append(ref)

            self.priors[name] = {'prior': prior, 'args': arglist}
            if ctx.distexpr().dims() is not None:
                self.priors[name]['dims'] = ctx.distexpr().dims().getText()

        elif ctx.distexpr().children[0].getText() == 'DISTX':
            prior = np.random.choice([m for m in self.models if m['type'] == 'C' and not self._dist_has_vector(m)])
            arglist = []
            args = len(prior['args'])
            for i in range(0, args):
                arg = prior['args'][i]
                v = generate_primitives(arg['type'] if self.structured else 'f', 1, False)[0]
                arglist.append(v)
            self.priors[name] = {'prior': prior, 'args': arglist}
            if ctx.distexpr().dims() is not None:
                self.priors[name]['dims'] = ctx.distexpr().dims().getText()
        else:
            dist = ctx.distexpr().children[0].getText()
            prior = [m for m in self.models if m['name'] == dist][0]
            arglist = []
            args = len(prior['args'])
            for i in range(0, args):
                arg = prior['args'][i]
                v = generate_primitives(arg['type'] if self.structured else 'f', 1, False)[0]
                arglist.append(v)
            self.priors[name] = {'prior': prior, 'args': arglist}

    def enterModel(self, ctx):
        self.mode.append('model')

    def exitModel(self, ctx):
        self.mode.pop(-1)

    def enterDistexpr(self, ctx=TemplateParser.DistexprContext):

        if len(self.mode) > 0 and self.mode[-1] == 'model':
            if ctx.DISTHOLE() is not None:
                # need a distribution
                params = len(ctx.params().param())
                filteredmodels = [m for m in self.models if len(m['args']) == params and not self._dist_has_vector(m)]
                model = np.random.choice(filteredmodels)
                self.distMap[ctx.DISTHOLE().getSymbol().tokenIndex] = model
            elif ctx.DISTXHOLE() is not None:
                filteredmodels = [m for m in self.models if len(m['args']) == 2 and not self._dist_has_vector(m)]
                model = np.random.choice(filteredmodels)
                self.distMap[ctx.DISTXHOLE().getSymbol().tokenIndex] = model
            else:
                distname = ctx.DISTRIBUTION().getText()
                model = [m for m in self.models if m['name'] == distname]
                self.distMap[ctx.DISTRIBUTION().getSymbol().tokenIndex] = model[0]
                args = model[0]['args']
                # adding constraints
                for i in range(0, len(args)):
                    param = ctx.params().param(i)
                    if param.CONSTHOLE() is not None:
                        self.constMap[param.CONSTHOLE().getSymbol().tokenIndex] = \
                            generate_primitives(args[i]['type'] if self.structured else 'f', 1, False)[0]

    def enterVal(self, ctx=TemplateParser.ValContext):
        if ctx.number().CONSTHOLE() is not None:
            self.constMap[ctx.number().CONSTHOLE().getSymbol().tokenIndex] = generate_primitives('f', 1, False)[0]

    def current_mode(self):
        if len(self.mode) > 1:
            return  self.mode[-1]
        return ''

    @staticmethod
    def _dist_has_vector(model):
        args = model['args']
        has_vector = False
        for arg in args:
            if arg.get('dim', None) is not None:
                has_vector =True
                break
        return has_vector

