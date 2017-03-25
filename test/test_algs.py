import numpy as np
import math
from algs import *
from support import *

def test_auto_encoder():
    test_input = np.array([0,1,0,0,0,0,0,0])
    test_output = algs.auto_encoder(test_input)
    assert test_input == test_output

print(sum((np.array([1, 2])**2)**0.5))
