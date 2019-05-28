# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:46:51 2019

@author: joao_
"""

import numpy as np
import time

from numba import vectorize, cuda

@vectorize(['float32(float32, float32)'], target='cuda')
def VectorAdd(a, b):
    return a + b

N = 128000000

A = np.ones(N, dtype=np.float32)
B = np.ones(N, dtype=np.float32)

start = time.time()
C = VectorAdd(A, B)
vector_add_time = time.time() - start

print("Elapsed (with compilation) = %s" % vector_add_time)

start = time.time()
C = VectorAdd(A, B)
vector_add_time = time.time() - start

print("Elapsed (after compilation) = %s" % vector_add_time)