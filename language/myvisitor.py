from antlr4 import *
from antlr.TemplateVisitor import TemplateVisitor
from antlr.TemplateParser import *
import numpy as np
from utils.utils import *

class MyVisitor(TemplateVisitor):

    def __init__(self, data):
        self.data = data

    def visitArith(self, ctx=TemplateParser.ArithContext):
        if ctx.AOP().getText() == '+':
            lhs = self.visit(ctx.expr(0))
            rhs = self.visit(ctx.expr(1))
            if isinstance(self.data[lhs], np.ndarray):
                if isinstance(self.data[rhs], np.ndarray):
                    sum = self.data[lhs] + self.data[rhs]
                else:
                    sum = self.data[lhs] + np.repeat(self.data[rhs], len(self.data[lhs]))
            elif isinstance(self.data[rhs], np.ndarray):
                if isinstance(self.data[lhs], np.ndarray):
                    sum = self.data[lhs] + self.data[rhs]
                else:
                    sum = self.data[rhs] + np.repeat(self.data[lhs], len(self.data[lhs]))
            else:
                sum = self.data[lhs] + self.data[rhs]

            tempname = '_sum'
            self.data[tempname] = sum
            return tempname

        elif ctx.AOP().getText() == '*':
            lhs = self.visit(ctx.expr(0))
            rhs = self.visit(ctx.expr(1))
            if isinstance(self.data[lhs], np.ndarray):
                if isinstance(self.data[rhs], np.ndarray):
                    mul = np.matmul(self.data[lhs], self.data[rhs])
                else:
                    mul = self.data[lhs] * self.data[rhs]
            elif isinstance(self.data[rhs], np.ndarray):
                if isinstance(self.data[lhs], np.ndarray):
                    mul = np.matmul(self.data[lhs], self.data[rhs])
                else:
                    mul = self.data[rhs] * self.data[lhs]
            else:
                mul = self.data[lhs] * self.data[rhs]

            tempname = '_mul'
            self.data[tempname] = mul
            return tempname
        else:
            print("unsupported")

    def visitVal(self, ctx):
        newname = get_new_var_name('_')
        self.data[newname] = float(ctx.getText())
        return newname

    def visitRef(self, ctx):
        return ctx.getText()
