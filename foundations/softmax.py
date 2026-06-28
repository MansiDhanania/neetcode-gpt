import numpy as np
from numpy.typing import NDArray
import math

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        soft=[0]*len(z)
        maxz=[max(z)]*len(z)
        sub=z-maxz
        ex=[0]*len(z)
        for i in range(0, len(z)):
            ex[i]=round(math.exp(sub[i]), 4)
        added=sum(ex)
        for i in range(len(z)):
            soft[i]=round(ex[i]/added, 4)
        return soft
