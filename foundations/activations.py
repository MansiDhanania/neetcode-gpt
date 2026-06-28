import numpy as np
from numpy.typing import NDArray
import math

class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        sig=[0]*len(z)
        for i in range(len(z)):
            e=math.exp(-(z[i]))
            sig[i]=np.round((1 / (float(1) + e)), 5)
        return sig

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        rel=[0]*len(z)
        for i in range(len(z)):
            rel[i]=float(max(0, z[i]))
        return rel