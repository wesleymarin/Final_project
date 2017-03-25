# You can do all of this in the `__main__.py` file, but this file exists
# to shows how to do relative import functions from another python file in
# the same directory as this one.
import numpy as np
from algs import *
from support import *
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import csv

def run_stuff():
    """
    This function is called in `__main__.py`

    Runs the autoencoder, nn trainer and unknown predictions.
    Writes predictions to the predictions.txt file
    """
    auto_nn = auto_encoder(1000)
    nn, auc = auc_for_all_optimized_params()
    unknown_dict = predict_unknowns(nn)

    f = open('predictions.txt', 'w')
    writer = csv.writer(f, delimiter='\t')
    for key, value in unknown_dict.items():
        writer.writerow([key] + [value])

def optimize_stuff():
    """
    This makes a bunch of graphs
    """
    determine_optimal_hidden_layer()
    determine_optimal_training_iterations()
    determine_optimal_subset_proportion()
    determine_optimal_neg_multiplier()
    determine_optimal_lambda()
    determine_optimal_alpha()

run_stuff()
