"""

Collection of all mystery functions the model has to learn...
Currently supports: add, sub, mul, div (4)

"""

class MysteryTools(): 

    def __init__(self):
        self._functions = [self._add, self._sub, self._mul, self._div]
        
    def _add(self, a, b):
        return a + b

    def _sub(self, a, b):
        return a - b

    def _mul(self, a, b):
        return a * b

    def _div(self, a, b):
        if b != 0:
            return round(a / b, 4)
        else:
            return "Error: Division by zero"

    def get_func(self, index):
        if 0 <= index < len(self._functions):
            return self._functions[index]
        else:
            return "Error: Invalid index"