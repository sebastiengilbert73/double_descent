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
import matplotlib.pyplot as plt
import imageio

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
        outputDirectory,
        trainingDataset,
        coefficientsFilepath
):
    logging.info("create_gif.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    training_df = pd.read_csv(trainingDataset)
    coefficients_df = pd.read_csv(coefficientsFilepath)

    xs = training_df['x'].tolist()
    ys = training_df['y'].tolist()



    maximum_degree = len(coefficients_df) - 1
    polynomial_xs = np.linspace(0, 1, 300)
    logging.debug(f"polynomial_xs = {polynomial_xs}")
    figure_filepaths = []
    for degree in range(maximum_degree + 1):
        coef_row = coefficients_df[coefficients_df['degree']==degree]
        coefs = []
        for c in range(0, degree + 1):
            coefs.append(coef_row[f"c{c}"].item())
        polynomial = Polynomial(coefs)
        #logging.info(f"polynomial.coefficients = {polynomial.coefficients}")

        polynomial_ys = [polynomial.evaluate(x) for x in polynomial_xs]

        fig, ax = plt.subplots()
        ax.set_title(f'Degree = {degree}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylim(-0.3, 1.2)

        ax.plot(polynomial_xs, polynomial_ys, c='red')
        ax.scatter(xs, ys, c='blue')

        figure_filepath = os.path.join(outputDirectory, f"fig_{degree}.png")
        plt.savefig(figure_filepath)
        figure_filepaths.append(figure_filepath)

    logging.info(f"Creating a gif...")
    images = []
    for figure_filepath in figure_filepaths:
        images.append(imageio.v2.imread(figure_filepath))
    imageio.mimsave(os.path.join(outputDirectory, "polynomials.gif"), images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_create_gif'",
                        default='./output_create_gif')
    parser.add_argument('--trainingDataset',
                        help="The training dataset. Default: '../data_generation/output_generate_polynomial/training.csv'",
                        default='../data_generation/output_generate_polynomial/training.csv')
    parser.add_argument('--coefficientsFilepath', help="The filepath to the coefficients csv file. Default: '../least_square_solving/output_polynomial_solution/coefficients.csv'",
                        default='../least_square_solving/output_polynomial_solution/coefficients.csv')
    args = parser.parse_args()
    main(
        args.outputDirectory,
        args.trainingDataset,
        args.coefficientsFilepath
    )