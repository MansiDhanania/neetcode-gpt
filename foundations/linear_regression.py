import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        pred=[]
        for i in range(len(X)):
            w=0
            for j in range(len(X[i])):
                w=w+(X[i][j]*weights[j])
            if pred==0:
                pass
            else:
                pred.append(round(w, 5))
        return pred

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        mse=0
        for i in range(len(model_prediction)):
            mse=mse+((model_prediction[i][0]-ground_truth[i][0])**2)
        return round((mse/len(model_prediction)), 5)
