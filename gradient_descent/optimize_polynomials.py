import sys
sys.path.append('../')  # Include the project base directory
import logging
import argparse
import ast
import os
from models.polynomial import PolynomialWithGD
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

class OneDDataset(Dataset):
    def __init__(self, dataframe):
        super(OneDDataset, self).__init__()
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data_row = self.dataframe.iloc[idx]
        x = data_row['x']
        y = data_row['y']
        return torch.tensor(x), torch.tensor(y)

def main(
        outputDirectory,
        randomSeed,
        trainingDataset,
        validationDataset,
        maximumPolynomialDegree,
        batchSize,
        learningRate,
        weightDecay,
        initializationSigma,
        numberOfEpochs
):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    logging.info(f"optimize_polynomials.main() device = {device}")

    random.seed(randomSeed)
    torch.manual_seed(randomSeed)

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Load the datasets
    training_df = pd.read_csv(trainingDataset)
    validation_df = pd.read_csv(validationDataset)
    xs = training_df['x'].tolist()
    ys = training_df['y'].tolist()

    training_dataset = OneDDataset(training_df)
    validation_dataset = OneDDataset(validation_df)

    training_dataloader = DataLoader(training_dataset, batch_size=batchSize, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batchSize)



    # Training parameters
    criterion = torch.nn.MSELoss()

    # Loop through the polynomial degree
    for degree in range(4, 5):#maximumPolynomialDegree + 1):
        # Create the polynomial
        polynomial = PolynomialWithGD( (initializationSigma * np.random.randn(degree + 1)).tolist())
        optimizer = torch.optim.Adam(polynomial.parameters(), lr=learningRate, weight_decay=weightDecay)

        with open(os.path.join(outputDirectory, f"epochLoss_degree{degree}.csv"), 'w') as epoch_loss_file:
            epoch_loss_file.write("epoch,training_loss,validation_loss,is_champion\n")
            minimum_validation_loss = float('inf')
            for epoch in range(1, numberOfEpochs + 1):
                # Set the polynomial to training mode
                polynomial.train()
                running_loss = 0.0
                number_of_batches = 0
                for input_tsr, target_tsr in training_dataloader:
                    input_tsr = input_tsr.to(device)
                    target_tsr = target_tsr.to(device)
                    polynomial.zero_grad()
                    output_tsr = polynomial(input_tsr)
                    loss = criterion(output_tsr, target_tsr)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    number_of_batches += 1
                    if number_of_batches %1 == 0:
                        print('.', end='', flush=True)
                average_training_loss = running_loss/number_of_batches
                logging.debug(f"average_training_loss = {average_training_loss}")

                # Evaluate with the validation dataset
                polynomial.eval()
                validation_running_loss = 0.0
                number_of_batches = 0
                for validation_input_tsr, validation_target_output_tsr in validation_dataloader:
                    validation_input_tsr = validation_input_tsr.to(device)
                    validation_target_output_tsr = validation_target_output_tsr.to(device)
                    validation_output_tsr = polynomial(validation_input_tsr)
                    validation_loss = criterion(validation_output_tsr, validation_target_output_tsr)
                    validation_running_loss += validation_loss.item()
                    number_of_batches += 1
                average_validation_loss = validation_running_loss/number_of_batches

                is_champion = False
                if average_validation_loss < minimum_validation_loss:
                    minimum_validation_loss = average_validation_loss
                    is_champion = True
                    champion_filepath = os.path.join(outputDirectory, f"champion_degree{degree}.pth")
                    torch.save(polynomial.state_dict(), champion_filepath)
                logging.info(f" **** Epoch {epoch}: average_training_loss = {average_training_loss}\taverage_validation_loss = {average_validation_loss}")
                if is_champion:
                    logging.info(f" ++++ Champion for validation loss ({average_validation_loss}) ++++")
                epoch_loss_file.write(f"{epoch},{average_training_loss},{average_validation_loss},{is_champion}\n")
        # Plot the polynomial
        polynomial_xs = np.linspace(0, 1, 300)
        polynomial_ys = [polynomial(torch.tensor([x]).unsqueeze(0)).item() for x in polynomial_xs]
        fig, ax = plt.subplots()
        ax.set_title(f'Degree = {degree}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylim(-0.3, 1.2)

        ax.plot(polynomial_xs, polynomial_ys, c='red')
        ax.scatter(xs, ys, c='blue')

        figure_filepath = os.path.join(outputDirectory, f"fig_{degree}.png")
        plt.savefig(figure_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_optimize_polynomials'",
                        default='./output_optimize_polynomials')
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    parser.add_argument('--trainingDataset',
                        help="The training dataset. Default: '../data_generation/output_generate_polynomial/training.csv'",
                        default='../data_generation/output_generate_polynomial/training.csv')
    parser.add_argument('--validationDataset',
                        help="The validation dataset. Default: '../data_generation/output_generate_polynomial/validation.csv'",
                        default='../data_generation/output_generate_polynomial/validation.csv')
    parser.add_argument('--maximumPolynomialDegree', help="The maximum polynomial degree. Default: 100", type=int,
                        default=100)
    parser.add_argument('--batchSize', help="The batch size. Default: 32", type=int, default=32)
    parser.add_argument('--learningRate', help="The learning rate. Default: 0.001", type=float, default=0.001)
    parser.add_argument('--weightDecay', help="The weight decay. Default: 0.00001", type=float, default=0.00001)
    parser.add_argument('--initializationSigma', help="The standard deviation of the parameter initialization. Default: 0.001", type=float, default=0.001)
    parser.add_argument('--numberOfEpochs', help="The number of epochs. Default: 100", type=int, default=100)
    args = parser.parse_args()
    main(
        args.outputDirectory,
        args.randomSeed,
        args.trainingDataset,
        args.validationDataset,
        args.maximumPolynomialDegree,
        args.batchSize,
        args.learningRate,
        args.weightDecay,
        args.initializationSigma,
        args.numberOfEpochs
    )
