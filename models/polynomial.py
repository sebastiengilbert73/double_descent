import torch

class Polynomial:
    def __init__(self, coefficients=[0]):
        self.coefficients = coefficients

    def evaluate(self, x):
        sum = 0
        for power_ndx in range(len(self.coefficients)):
            sum += self.coefficients[power_ndx] * x**power_ndx
        return sum

class PolynomialWithGD(torch.nn.Module):
    def __init__(self, coefficients=[0]):
        super(PolynomialWithGD, self).__init__()
        self.coefficients = torch.nn.Parameter(torch.tensor(coefficients))  # The coefficients will be registered for optimization

    def forward(self, x_tsr):  # x_tsr.shape = (N)
        sum = torch.zeros_like(x_tsr)
        for power_ndx in range(len(self.coefficients)):
            sum += self.coefficients[power_ndx] * torch.pow(x_tsr, power_ndx)#**power_ndx
        return sum