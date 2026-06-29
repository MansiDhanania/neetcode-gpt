import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        neuron1=[0]*len(W1)
        neuron2=[0]*len(W2)

        dW2=[0]*len(W2)
        dhid=[0]*len(neuron1)
        dW1 = [[0.0] * len(x) for _ in range(len(W1))]

        db2=[0]*len(b2)
        db1=[0]*len(b1)

        for i in range(len(W1)):
            n=0
            for j in range(len(x)):
                n=n+(x[j]*W1[i][j])
            neuron1[i]=max(0, n+b1[i])

        for i in range(len(W2)):
            n=0
            for j in range(len(neuron1)):
                n=n+(neuron1[j]*W2[i][j])
            neuron2[i]=n+b2[i]

        loss=0

        for i in range(len(neuron2)):
            error=neuron2[i]-y_true[i]
            loss=loss+(error**2)
            loss_grad=(2*error)/len(neuron2)

            db2[i]=round(loss_grad, 4)

            row=[]
            for j in range(len(neuron1)):
                row.append(loss_grad*neuron1[j])
                dhid[j]=dhid[j]+loss_grad*W2[i][j]
            dW2[i]=row
        loss=loss/len(neuron2)

        for j in range(len(neuron1)):
            if neuron1[j] == 0:
                dhid[j] = 0
            db1[j] = round(dhid[j], 4)
            
            for k in range(len(x)):
                dW1[j][k] = round(dhid[j] * x[k], 4)
        
        db2 = [round(b, 4) for b in db2]
        dW2 = [[round(w, 4) for w in row] for row in dW2]

        return {'loss':round(loss, 4), 'dW1':dW1, 'db1':db1, 'dW2':dW2, 'db2':db2}