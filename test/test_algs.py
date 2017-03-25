import numpy as np
import math
from algs import *
from support import *

def test_auto_encoder():
    test_input = np.array([0,0,0,1,0,0,0,0])
    auto_nn = auto_encoder(10000)
    prediction = auto_nn.predict(test_input)
    prediction = np.rint(prediction)
    print(prediction)
    assert (test_input == prediction).all()

def test_dna_nn():
    """
    This is to make sure the auc is better than random (so my nn is actually
    doing something)
    """
    nn, auc = auc_for_all_optimized_params()
    assert (auc > 0.99)
