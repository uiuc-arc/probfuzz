from antlr4 import *
from antlr.TemplateVisitor import TemplateVisitor
from antlr.TemplateParser import *
import numpy as np
from utils.utils import *


class Checker(TemplateVisitor):
    def __init__(self, distmap, priormap, constmap):
        self.checked = True
        self.distMap = distmap
        self.priorMap = priormap
        self.constMap = constmap
        self.valid = True

    def visitTemplate(self, ctx):
        for child in ctx.children:
            if isinstance(child, TemplateParser.ModelContext):
                self.visit(child)

    def visitDistexpr(self, ctx):
        if ctx.DISTHOLE() is not None:
            model = self.distMap[ctx.DISTHOLE().getSymbol().tokenIndex]
        elif ctx.DISTXHOLE() is not None:
            return
        else:
            distname = ctx.DISTRIBUTION().getText()
            model = self.distMap[ctx.DISTRIBUTION().getSymbol().tokenIndex]
        modelsupport = model['support']

        params = ctx.params()
        supports = self.visit(params)

        for i in range(0, len(model['args'])):
            arg = model['args'][i]
            type = arg['type']
            paramsupport = supports[i]
            for s in paramsupport:
                if not includes(s, type):
                    self.valid = False
                    break

    def visitParams(self, ctx):
        supports = []
        for param in ctx.param():
            supports.append(self.visit(param))
        return supports

    def visitParam(self, ctx):
        if ctx.CONSTHOLE() is not None:
            return ['x']
        else:
            return self.visit(ctx.expr())

    def visitRef(self, ctx):
        name = ctx.ID().getText()
        if name in self.priorMap:
            return [self.priorMap[name]['prior']['support']]
        return []

    def visitVal(self, ctx):
        return ['x']

    def visitArith(self, ctx):
        lhs = self.visit(ctx.expr(0))
        rhs = self.visit(ctx.expr(1))
        return lhs + rhs