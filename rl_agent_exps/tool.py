# Example tool classes (simplified)
class Calculator:
    def calculate(self):
        return "Result of calculation"

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        if b != 0:
            return round(a / b, 4)
        else:
            return "Error: Division by zero"
