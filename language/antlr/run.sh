#!/bin/bash
antlr4='java -Xmx500M -cp ".:./antlr-4.7.1-complete.jar:$CLASSPATH" org.antlr.v4.Tool'
grun='java org.antlr.v4.gui.TestRig'
$antlr4 -package "tool.parser"  -Dlanguage=Python2 -visitor Template.g4
touch __init__.py
