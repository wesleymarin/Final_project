import numpy as np
import math
from ... import algs
from support import *

def test_auto_encoder():
    test_input = np.array([0,0,0,1,0,0,0,0])
    auto_nn = algs.auto_encoder(test_input)
    print(auto_nn.predict(test_input))
    assert test_input == auto_nn.predict(test_input)

test_auto_encoder()
