class Polynomial:
    def __init__(self, coefficients=[0]):
        self.coefficients = coefficients

    def evaluate(self, x):
        sum = 0
        for power_ndx in range(len(self.coefficients)):
            sum += self.coefficients[power_ndx] * x**power_ndx
        return sum