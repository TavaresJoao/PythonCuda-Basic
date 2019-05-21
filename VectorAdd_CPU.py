# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:15:06 2019

@author: joao_
"""

import numpy as np
import time

def VectorAdd(a, b):
    return a + b

N = 128000000

A = np.ones(N, dtype=np.float32)
B = np.ones(N, dtype=np.float32)

start = time.time()
C = VectorAdd(A, B)
vector_add_time = time.time() - start

print ("%f" % vector_add_time)