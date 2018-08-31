from language.antlr.TemplateListener import TemplateListener
from language.antlr.TemplateVisitor import TemplateVisitor

class Backend(TemplateVisitor):
    def __init__(self, target_directory):
        self._directory = target_directory

    @staticmethod
    def run(self):
        return NotImplementedError

    def create_program(self):
        return NotImplementedError