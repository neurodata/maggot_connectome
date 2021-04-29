import numpy as np
from numba import jit
from graspologic.simulations import sample_edges
import time


@jit(nopython=True, fastmath=True)
def sample_rdpg_numba(P):
    out = np.zeros(P.shape)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            # out[i, j] = np.random.binomial(1, P[i, j])
            sample = np.random.uniform(0, 1)
            if sample < P[i, j]:
                out[i, j] = 1
    return out


P = np.array([[0.4, 0.5], [0.3, 0.5]])
sample_rdpg_numba(P)

for n in [2 ** 6, 2 ** 8, 2 ** 10, 2 ** 12, 2 ** 14]:
    for i in range(10):
        P = np.random.uniform(size=(n, n))
        currtime = time.time()
        sample_rdpg_numba(P)
        numba_time = time.time() - currtime
        currtime = time.time()
        sample_edges(P, directed=True, loops=True)
        numpy_time = time.time() - currtime
        ratio = numpy_time / numba_time
        print(f"Speedup ratio = {ratio}")
    print()