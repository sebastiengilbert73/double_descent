import sys
sys.path.append('../')  # Include the project base directory
import logging
import argparse
import ast
import os
from models.polynomial import Polynomial
import random

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
        outputDirectory,
        numberOfTrainingPoints,
        numberOfValidationPoints,
        numberOfTestPoints,
        coefficients,
        xRange,
        noiseSigma
):
    logging.info("generate_polynomial.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    polynomial = Polynomial(coefficients)

    training_points = generate_points(polynomial, xRange, numberOfTrainingPoints, noiseSigma)
    save_points(os.path.join(outputDirectory, "training.csv"), training_points)

    validation_points = generate_points(polynomial, xRange, numberOfValidationPoints, noiseSigma)
    save_points(os.path.join(outputDirectory, "validation.csv"), validation_points)

    test_points = generate_points(polynomial, xRange, numberOfTestPoints, noiseSigma)
    save_points(os.path.join(outputDirectory, "test.csv"), test_points)


def generate_points(polynomial, x_range, number_of_points, noise_sigma):
    points = []
    for p_ndx in range(number_of_points):
        x = x_range[0] + random.random() * (x_range[1] - x_range[0])
        y = polynomial.evaluate(x) + random.gauss(0, noise_sigma)
        points.append((x, y))
    return points

def save_points(filepath, points):
    with open(filepath, 'w') as output_file:
        output_file.write("x,y\n")
        for p in points:
            output_file.write(f"{p[0]},{p[1]}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_generate_polynomial'", default='./output_generate_polynomial')
    parser.add_argument('--numberOfTrainingPoints', help="The number of training points. Default: 400",
                        type=int, default=400)
    parser.add_argument('--numberOfValidationPoints', help="The number of validation points. Default: 100",
                        type=int, default=100)
    parser.add_argument('--numberOfTestPoints', help="The number of test points. Default: 100",
                        type=int, default=100)
    parser.add_argument('--coefficients', help="The polynomial coefficients. Default: '[1, -14, 59, -88, 43]'", default='[1, -14, 59, -88, 43]')
    parser.add_argument('--xRange', help="The range of x values. Default: '[0, 1]'", default='[0, 1]')
    parser.add_argument('--noiseSigma', help="The gaussian noise standard deviation. Default: 0.05", type=float, default=0.05)
    args = parser.parse_args()
    args.coefficients = ast.literal_eval(args.coefficients)
    args.xRange = ast.literal_eval(args.xRange)
    main(
        args.outputDirectory,
        args.numberOfTrainingPoints,
        args.numberOfValidationPoints,
        args.numberOfTestPoints,
        args.coefficients,
        args.xRange,
        args.noiseSigma
    )