import sys
sys.path.append('../')  # Include the project base directory
import logging
import argparse
import ast
import os
from models.polynomial import Polynomial
import random
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
        outputDirectory,
        trainingDataset,
        validationDataset,
        maximumPolynomialDegree
):
    logging.info("polynomial_solution.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Load the datasets
    training_df = pd.read_csv(trainingDataset)
    validation_df = pd.read_csv(validationDataset)

    # Loop through the polynomial degrees
    with open(os.path.join(outputDirectory, "coefficients.csv"), 'w') as coefficients_file:
        coefficients_file.write("degree,")
        for coef in range(maximumPolynomialDegree):
            coefficients_file.write(f"c{coef},")
        coefficients_file.write(f"c{maximumPolynomialDegree}\n")
        with open(os.path.join(outputDirectory, "degree_loss.csv"), 'w') as loss_file:
            loss_file.write("degree,training_loss,validation_loss\n")
            for degree in range(0, maximumPolynomialDegree + 1):
                logging.info(f"Degree {degree}")

                # Build the system of linear equations
                # y = c0 + c1 * x + c2 * x^2 + c3 * x^3 + ...
                #
                # | 1   x0   x0^2   x0^3 ... | | c0 |     | y0 |
                # | 1   x1   x1^2   x1^3 ... | | c1 |  =  | y1 |
                # | ...                      | | ...|     | ...|
                #           A z = b
                A = np.zeros((len(training_df), degree + 1))
                b = np.zeros(len(training_df))
                for row_ndx in range(len(training_df)):
                    x = training_df.iloc[row_ndx]['x']
                    y = training_df.iloc[row_ndx]['y']
                    for col_ndx in range(degree + 1):
                        A[row_ndx, col_ndx] = x**col_ndx
                    b[row_ndx] = y

                z, residuals, rank, singular_values = np.linalg.lstsq(A, b)
                #AT = A.T
                #z = np.linalg.inv(AT @ A) @ AT @ b
                # Write the coefficients
                coefficients_file.write(f"{degree},")
                for coef in range(maximumPolynomialDegree + 1):
                    if coef < z.shape[0]:
                        coefficients_file.write(f"{z[coef]}")
                        if coef < maximumPolynomialDegree:
                            coefficients_file.write(',')
                        else:
                            coefficients_file.write("\n")
                    else:
                        if coef < maximumPolynomialDegree:
                            coefficients_file.write("0,")
                        else:
                            coefficients_file.write("0\n")

                # Create the polynomial
                polynomial = Polynomial(z)

                # Evaluate the training and validation losses
                training_loss = squared_error_average(training_df, polynomial)
                validation_loss = squared_error_average(validation_df, polynomial)
                logging.info(f"training_loss = {training_loss};\tvalidation_loss = {validation_loss}")
                loss_file.write(f"{degree},{training_loss},{validation_loss}\n")

def squared_error_average(xy_df, polynomial):
    if len(xy_df) < 1:
        raise ValueError(f"polynomial_solution.squared_error_average(): Empty dataframe")
    sum = 0
    for pt_ndx in range(len(xy_df)):
        x = xy_df.iloc[pt_ndx]['x']
        y = xy_df.iloc[pt_ndx]['y']
        polynomial_output = polynomial.evaluate(x)
        squared_error = (polynomial_output - y)**2
        sum += squared_error
    return sum/len(xy_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_polynomial_solution'",
                        default='./output_polynomial_solution')
    parser.add_argument('--trainingDataset', help="The training dataset. Default: '../data_generation/output_generate_polynomial/training.csv'",
                        default='../data_generation/output_generate_polynomial/training.csv')
    parser.add_argument('--validationDataset',
                        help="The validation dataset. Default: '../data_generation/output_generate_polynomial/validation.csv'",
                        default='../data_generation/output_generate_polynomial/validation.csv')
    parser.add_argument('--maximumPolynomialDegree', help="The maximum polynomial degree. Default: 100", type=int, default=100)
    args = parser.parse_args()
    main(
        args.outputDirectory,
        args.trainingDataset,
        args.validationDataset,
        args.maximumPolynomialDegree
    )