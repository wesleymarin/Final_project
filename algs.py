import numpy as np
import time
from support import *
import math
import itertools as it
from random import sample
from sklearn import metrics
import matplotlib.pyplot as plt

def auto_encoder(train_iterations):
    """
    This function trains the neural network on 8-bit binary vectors, then
    returns those same vectors as output.

    Input:  Int for training iterations
    Output: Boolean, if all inputs were correctly output
    """
    ## Get the Neural network set up, 8 input layers, 3 mid layers and 8 output
    auto_nn = NeuralNetwork(8,3,8)

    ## Displaying the pre-training weights
    print("Pre-training weights:")
    print(auto_nn.W1)
    print(auto_nn.W2)
    print(" ")

    ## Generating a training set and test set
    training_set = np.zeros((8,8),dtype=np.int)
    test_set = np.zeros_like(training_set)

    ## Put a single 1 in every 0's vector
    for i in range(0,8):
        training_set[i][i] = 1
        test_set[i][i] = 1

    ## Train on the training set
    auto_nn.train(training_set, training_set, train_iterations)

    ## Display the post training weights
    print("Post-training weights:")
    print(auto_nn.W1)
    print(auto_nn.W2)
    print(" ")

    ## Display the nn predictions of the test set
    predictions = auto_nn.predict(test_set)

    ## Round predictions to nearest int
    rounded_predictions = np.zeros_like(test_set)
    for i in range(0, predictions.shape[0]):
        rounded_predictions[i] = np.rint(predictions[i])

    ## Print everything for visual inspection
    print(rounded_predictions)
    print(test_set)
    print("")
    print("Correctly predicted:")
    print((rounded_predictions == test_set))

    ## Return if all predictions were correct
    return auto_nn

def dna_read_in(filepath):
    """
    This function reads in DNA sequences from the given filepath

    Input: Filepath of file with DNA sequences
    Output: List of sequences
    """
    sequence_list = []

    ## Open file and treat every line as a new sequence
    f = open(filepath, 'r')
    for line in f:

        ## Strip all the pesky new line characters
        line = line.rstrip('\r\n')
        sequence_list.append(line)

    return sequence_list

def fasta_read_in(filepath):
    """
    This function reads in sequences from fasta files

    Input: Filepath of fasta file with DNA sequences
    Output: List of DNA sequences
    """
    f = open(filepath, 'r')

    ## Initialize first pass list of sequences
    raw_sequence_list = []

    ## Since the seunces are spread over multiple lines, this counter helps
    ## keep track of what sequence we are on
    counter = -1
    for line in f:
        line = line.rstrip('\r\n')

        ## If the line begins with >, it means it is the start of a new sequence
        ## so we want to iterate counter by 1 and add a new element to
        ## raw_sequence_list
        if '>' in line:
            counter += 1
            raw_sequence_list.append('')
        ## If there is no >, the current sequence is continued
        else:
            raw_sequence_list[counter] += line

    ## Final sequence list, sequences that are smaller than 1000bp mess up
    ## my random sampling, so I want to remove them at this step.
    ## There are only 5 of them, so I figured it was fine.
    sequence_list = []
    for sequence in raw_sequence_list:
        if len(sequence) == 1000:
            sequence_list.append(sequence)

    ## Get rid of the raw list to make sure we are not taking up more space
    ## than needed.
    del raw_sequence_list
    return(sequence_list)

def dna_to_num(dna_sequences, region_size):
    """
    This function takes dna sequences as input, then converts them to binary
    vectors that are based on region size.
    http://file.scirp.org/Html/3-9102277_65923.htm this document is what this
    method was based on.

    Input: List of dna sequences
    Output: Binary array representing each sequence.
    """
    array_rows = 0

    ## Getting the needed parameters for initializing the output array
    for sequence in dna_sequences:
        array_rows += len(sequence)-2-(region_size-1)

    ## Initializing the output array to an array of 1's, since there should
    ## only be as many ones per line as region_size, I can validate that
    ## this array is correctly filled
    num_array = np.ones((array_rows,64*region_size))
    for i in range(0,len(dna_sequences)):
        ## Convert sequence to a DNAsequence object.
        ## This object converts it into numbers
        current_sequence = DNAsequence(dna_sequences[i], region_size)

        ## Save the numbers
        current_array = current_sequence.number_array

        ## Figure out how many rows this sequence will take up
        array_rows = len(dna_sequences[i])-2-(region_size-1)

        ## Write the sequnce to the output array
        num_array[array_rows*i:array_rows*(i+1),:] = current_array

    ## Make sure there are no extra 1's
    assert all([sum(row) == region_size for row in num_array])
    return(num_array)

def nn_training(train_array, true_array, iterations, region_size, hidden_layer_multiplier, lam, alpha):
    """
    This function creates a nerualnetwork object based on the input parameters, then trains it.

    Input: Numpy array of training vectors
           Numpy array of true values
           Int for training iterations
           Int for region size (always 15)
           Int for how large to make the hidden layer
           Float for the NN lambda value
           Float for the NN alpha value
    Output: A trained NeuralNetwork object
    """
    ## Since my vectors are 64*region_size bits long, make the input layer this large
    top_layer_size = region_size*64

    ## Set the middle layer to be log2(top_layer_size)*input
    middle_layer_size = int(math.log(top_layer_size,2))*hidden_layer_multiplier

    ## Create the NeuralNetwork object based on the set parameters
    binding_nn = NeuralNetwork(top_layer_size,middle_layer_size,1)
    binding_nn.lam = lam
    binding_nn.alpha = alpha

    ## Train the network
    binding_nn.train(train_array, true_array, iterations)
    return(binding_nn)

def nn_prediction(nn, test_sequences, region_size):
    """
    This function takes a trained NeuralNetwork object and uses it to
    make predictions

    Input:  Trained NeuralNetwork object
            List of sequences to predict
            Int for region size (always 15)
    Output: List of predicted probabilities
    """
    predictions = []
    for test_sequence in test_sequences:
        ## Convert the sequence to number form
        test_number = dna_to_num([test_sequence], region_size)

        ## Predict, I am normalizing by length of the test_number array
        ## because the way the input method works separates a sequence
        ## into regions. I set the region size at 15 (which takes up all
        ## 17 bp) to make things simpler.
        predictions.append(sum(nn.predict(test_number))/len(test_number))
    return(predictions)

def neg_sample_preparation(neg_sequence_list, pos_selection_list_len, neg_list_proportion):
    """
    This function randomly samples the negative sequences, witholding all the
    non-selected samples as test data.

    Input:  List of sequences
            Int of the length of positive training samples
            Int for how many neg training samples to have compared to pos
    Output: List of negative training samples
            List of negative testing samples
    """
    ## Add the reverse complement of every sequence to the list
    for i in range(0,len(neg_sequence_list)):
        neg_sequence_list.append(reverse_complement(neg_sequence_list[i]))

    ## Create a list of random indices, list is as long as the pos_list_len * neg_proportion
    neg_random_sampling = sample(range(0,len(neg_sequence_list)), math.floor(pos_selection_list_len*neg_list_proportion))

    ## Sort the list so we can pop stuff off
    neg_random_sampling.sort(reverse=True)

    ## Pop every selected index off from the neg_sequence_list, create a list from these
    neg_train_sequences = [neg_sequence_list.pop(x) for x in neg_random_sampling]

    ## Since my method is easier and simpler if everything is the same length,
    ## this part picks a random 17bp sequence from the 1000bp sequence
    ## This section is for the training data
    in_sequence_sampling = np.random.randint(0,high=len(neg_train_sequences[0])-17,size=len(neg_train_sequences))
    neg_train_sequences = [neg_train_sequences[i][in_sequence_sampling[i]:in_sequence_sampling[i]+17] for i in range(0,len(neg_train_sequences))]

    ## This does the same thing for the test data
    neg_test_sequences = neg_sequence_list
    in_sequence_sampling = np.random.randint(0,high=len(neg_test_sequences[0])-17,size=len(neg_test_sequences))
    neg_test_sequences = [neg_test_sequences[i][in_sequence_sampling[i]:in_sequence_sampling[i]+17] for i in range(0,len(neg_test_sequences))]

    return(neg_train_sequences, neg_test_sequences)

def pos_sample_preparation(pos_sequence_list, rand_sample_proportion):
    """
    This function randomly samples the positive sequences, witholding all the
    non-selected samples as test data.

    Input:  List of positive sequences
            Float for proportion of the positive sequence to train on
    Output: List of positive training samples
            List of positive testing samples
    """
    ## Add the reverse complement of every sequence to the end of the list
    for i in range(0,len(pos_sequence_list)):
        pos_sequence_list.append(reverse_complement(pos_sequence_list[i]))

    ## Create a list of random indices, list is as long as the pos_list_len*proportion
    pos_random_sampling = sample(range(0,len(pos_sequence_list)), math.floor(len(pos_sequence_list)*rand_sample_proportion))

    ## Sort the list so we can pop stuff off
    pos_random_sampling.sort(reverse=True)

    ## Pop every selected index off from the positive sequences list to go input_vector
    ## the training data list
    pos_train_sequences = [pos_sequence_list.pop(x) for x in pos_random_sampling]

    ## All remaining sequences are put in the test data list
    pos_test_sequences = pos_sequence_list

    return(pos_train_sequences, pos_test_sequences)

def reverse_complement(sequence):
    """
    This function creates the reverse complement of a DNA sequence

    Input:  DNA sequence
    Output: DNA sequence
    """
    ## Turn the sequence into a list so we can do list comprehension
    sequence = list(sequence)
    ## Pop off the end of the sequence, find the complement of that piece, then
    ## add that to the new sequence. Join everything together at the end.
    new_sequence = ''.join([comp_dict[sequence.pop()] for i in range(0,len(sequence))])
    return(new_sequence)

def nn_test(training_iterations, sampling_proportion, hidden_layer_multiplier, neg_multiplier, lam, alpha):
    """
    This function is the workhorse, it coordinates all the different parameters,
    reads in the sequences from files, creates a NeuralNetwork object, trains
    that object, tests the object on the witheld data, then returns the testing
    results, labels, and the NeuralNetwork object.

    Input:  Int of how many training iterations to go through
            Float of what proportaion of the pos data to samples
            Int for how large to make the hidden layer
            Float for how many neg samples there should be compared to pos
            Float for lambda
            Float for alpha
    Output: List of predicted probabilities
            List of true labels
            Trained NeuralNetwork object
    """
    ## This is a consequence of the method of data input i chose. I fixed it at
    ## 15 to simplify things
    region_size = 15

    ## Gotta have that visual output
    print("Reading in sequences")
    all_pos_sequences = dna_read_in("rap1-lieb-positives.txt")
    all_neg_sequences = fasta_read_in('yeast-upstream-1k-negative.fa')
    print(" ")

    print("Setting up random sampling for " + str(sampling_proportion) +" of pos samples")
    pos_train_sequences, pos_test_sequences = pos_sample_preparation(all_pos_sequences, sampling_proportion)
    neg_train_sequences, neg_test_sequences = neg_sample_preparation(all_neg_sequences, len(all_pos_sequences)*sampling_proportion, neg_multiplier)
    print(" ")

    ## Here we are getting all the training and test data, as well as the true labels
    print("Converting DNA sequence to binary")
    pos_train_numbers = dna_to_num(pos_train_sequences, region_size)
    pos_test_numbers = dna_to_num(pos_test_sequences, region_size)
    pos_true_vector = np.ones((pos_train_numbers.shape[0],1))
    neg_train_numbers = dna_to_num(neg_train_sequences, region_size)
    neg_test_numbers = dna_to_num(neg_test_sequences, region_size)
    neg_true_vector = np.zeros((neg_train_numbers.shape[0],1))
    print(" ")

    ## Combine the pos and neg stuff
    print("Creating full training array")
    train_numbers = np.concatenate((pos_train_numbers, neg_train_numbers), axis=0)
    print(" ")

    print("Creating true vector")
    true_vector = np.concatenate((pos_true_vector, neg_true_vector), axis=0)
    print(" ")

    print("Training neural network")
    binding_nn = nn_training(train_numbers, true_vector, training_iterations, region_size, hidden_layer_multiplier, lam, alpha)
    print(" ")

    print("Testing positives")
    pos_test = np.array([x for x in nn_prediction(binding_nn, pos_test_sequences, region_size)])
    pos_labels = np.ones_like(pos_test)
    print(" ")

    print("Testing negatives")
    neg_test = np.array([x for x in nn_prediction(binding_nn, neg_test_sequences, region_size)])
    neg_labels = np.zeros_like(neg_test)
    print(" ")

    test_results = np.concatenate((pos_test, neg_test), axis=0)
    test_labels = np.concatenate((pos_labels, neg_labels), axis=0)

    print("----------------------------")
    return(test_results, test_labels, binding_nn)

def predict_unknowns(binding_nn):
    """
    This function uses a trained NeuralNetwork to predict the sequences in
    the unknown file.

    Input:  Trained NeuralNetwork object
    Output: A dictionary of sequence keys with prediction values
    """
    ## Set up the dictionary
    unknown_dict = {}

    ## Again with the set region size
    region_size=15

    print("Testing unknowns")
    unknown_sequences = dna_read_in("rap1-lieb-test.txt")

    ## Do the predictions one at a time so i can add them to the dict
    for x in unknown_sequences:
        unknown_dict[x] = nn_prediction(binding_nn, [x], region_size)[0][0]
    print(" ")
    return(unknown_dict)

def determine_optimal_hidden_layer():
    """
    This function iterates over a range of hidden layer sizes, computes
    the area under the ROC curve for each one, and outputs that data as
    a graph.
    """
    ## Initialize the graphing stuff
    auc_history = []
    x_vals = []

    ## Goint to iterate over 10 hidden layer sizes
    for i in range(0,10):
        print("Part " + str(i+1) + " of " + str(10))
        x_vals.append(i)

        ## Set up and test a NeuralNetwork for the given hidden layer size
        ## all other parameters are held constant
        test_results, test_labels, nn = nn_test(750, 0.2, i, 1, 0.001, 2)

        ## Calculate an ROC curve for the tested stuff
        fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_results)

        ## Calculate the area under the ROC curve
        area_under_curve = metrics.auc(fpr, tpr)

        ## Add that to the Y-values list
        auc_history.append(area_under_curve)

    ## Plot it all
    plt.plot(x_vals, auc_history)
    plt.title("AUC for hidden layer size")
    plt.xlabel("log2(input_layer_size)*multiplier")
    plt.ylabel("AUC")
    plt.ylim([0.0, 1.0])
    plt.show()

def determine_optimal_training_iterations():
    """
    This function iterates over a range of training iterations, computes
    the area under the ROC curve for each one, and outputs that data as
    a graph.
    """

    ## Everything here is the same as the previous function, except that
    ## we are iterating over 20 values, and the training iterations is what
    ## is being changed.

    auc_history = []
    x_vals = []
    for i in range(0,20):
        print("Part " + str(i+1) + " of " + str(20))
        x_vals.append(i*100)
        test_results, test_labels, nn = nn_test(100*i, 0.2, 1, 1, 0.001, 2)
        fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_results)
        area_under_curve = metrics.auc(fpr, tpr)
        auc_history.append(area_under_curve)

    plt.plot(x_vals, auc_history)
    plt.title("AUC for training iterations")
    plt.xlabel("Training iterations")
    plt.ylabel("AUC")
    plt.ylim([0.0, 1.0])
    plt.show()

def determine_optimal_subset_proportion():
    """
    This function iterates over a range of training subset sizes, computes
    the area under the ROC curve for each one, and outputs that data as
    a graph.
    """

    ## Everything here is the same as the previous function, except that
    ## we are iterating over 20 values, and the training proportion is what
    ## is being changed.

    auc_history = []
    x_vals = []
    for i in range(1,20):
        print("Part " + str(i) + " of " + str(19))
        x_vals.append(i*0.05)
        test_results, test_labels, nn = nn_test(750, 0.05*i, 1, 1, 0.001, 2)
        fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_results)
        area_under_curve = metrics.auc(fpr, tpr)
        auc_history.append(area_under_curve)

    plt.plot(x_vals, auc_history)
    plt.title("AUC per proportion of samples")
    plt.xlabel("Proportion of samples")
    plt.ylabel("AUC")
    plt.ylim([0.0, 1.0])
    plt.show()

def determine_optimal_neg_multiplier():
    """
    This function iterates over a range of neg to pos ratio sizes, computes
    the area under the ROC curve for each one, and outputs that data as
    a graph.
    """

    ## Everything here is the same as the previous function, except that
    ## we are iterating over 9 values, and the neg to pos ratio is what
    ## is being changed.

    auc_history = []
    x_vals = []
    for i in range(1,10):
        print("Part " + str(i+1) + " of " + str(10))
        x_vals.append(i*0.5)
        test_results, test_labels, nn = nn_test(750, 0.2, 1, i*0.5, 0.001, 2)
        fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_results)
        area_under_curve = metrics.auc(fpr, tpr)
        auc_history.append(area_under_curve)

    plt.plot(x_vals, auc_history)
    plt.title("AUC per proportion of neg to pos samples")
    plt.xlabel("Proportion of neg samples")
    plt.ylabel("AUC")
    plt.ylim([0.0, 1.0])
    plt.show()

def determine_optimal_lambda():
    """
    This function iterates over a range of lambda sizes, computes
    the area under the ROC curve for each one, and outputs that data as
    a graph.
    """

    ## Everything here is the same as the previous function, except that
    ## we are iterating over 10 values, and the lambda value is what
    ## is being changed.

    auc_history = []
    x_vals = []
    for i in range(1,11):
        value = pow(3, i)*0.00001
        print("Part " + str(i) + " of " + str(10))
        x_vals.append(value)
        test_results, test_labels, nn = nn_test(750, 0.2, 1, 1, value, 2)
        fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_results)
        area_under_curve = metrics.auc(fpr, tpr)
        auc_history.append(area_under_curve)

    plt.semilogx(x_vals, auc_history, basex=3)
    plt.title("AUC per lambda value")
    plt.xlabel("lambda value")
    plt.ylabel("AUC")
    plt.ylim([0.0, 1.0])
    plt.show()

def determine_optimal_alpha():
    """
    This function iterates over a range of alpha values, computes
    the area under the ROC curve for each one, and outputs that data as
    a graph.
    """

    ## Everything here is the same as the previous function, except that
    ## we are iterating over 10 values, and the lambda value is what
    ## is being changed.

    auc_history = []
    x_vals = []
    for i in range(1,11):
        value = pow(2,i)*0.01
        print("Part " + str(i) + " of " + str(10))
        x_vals.append(value)
        test_results, test_labels, nn = nn_test(750, 0.2, 1, 1, 0.001, value)
        fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_results)
        area_under_curve = metrics.auc(fpr, tpr)
        auc_history.append(area_under_curve)

    plt.semilogx(x_vals, auc_history, basex=2)
    plt.title("AUC per alpha value")
    plt.xlabel("alpha value")
    plt.ylabel("AUC")
    plt.ylim([0.0, 1.0])
    plt.show()

def auc_for_all_optimized_params():
    """
    This function is hard coded for good values found in the previous
    optimizations. It creates, trains and tests a NeuralNetwork based
    on those values. It then plots the ROC for the held out data, and
    outputs the NeuralNetwork object.

    Output: Trained NeuralNetwork object
    """
    test_results, test_labels, nn = nn_test(1000, 0.75, 2, 5, 0.001, 1)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_results)
    area_under_curve = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=area_under_curve)
    plt.plot([0,1],[0,1], 'k--', color='k')
    plt.title("ROC curve for optimal parameters")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend(loc='lower right')
    #plt.show()

    return(nn, area_under_curve)
