#!/usr/bin/python
try: 
    import antlr4, six, astunparse, ast, pystan, edward, pyro, tensorflow, pandas, torch
    print("Install successful")
except ImportError as err:
    print("Install failed: {0}".format(err))
