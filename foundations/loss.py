import numpy as np
from numpy.typing import NDArray
import math

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        loss=0
        for i in range(len(y_true)):
            calc=(y_true[i]*math.log(y_pred[i]+(10**(-7)))+((1-y_true[i])*math.log(1-y_pred[i]+(10**(-7)))))
            loss=loss-calc
        return round((loss/len(y_true)), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        loss=[0]*len(y_true)
        for i in range(len(y_true)):
            l1=0
            for j in range(len(y_true[i])):
                l1=l1+(y_true[i][j]*y_pred[i][j])
            loss[i]=(-math.log(l1))
        p=sum(loss)/len(y_true)
        return round(p, 4)