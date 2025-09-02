import numpy as np
from .activation import Softmax


class MSELoss:
    def forward(self, A, Y):
        """
        Calculate the Mean Squared error (MSE)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss (scalar)

        Read the writeup (Hint: MSE Loss Section) for implementation details for below code snippet.
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]
        se = (A - Y) ** 2
        sse = np.sum(se)
        mse = sse / (self.N * self.C)
        return mse

    def backward(self):
        """
        Calculate the gradient of MSE Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        Read the writeup (Hint: MSE Loss Section) for implementation details for below code snippet.
        """
        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)
        return dLdA


class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss (XENT)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss (scalar)

        Read the writeup (Hint: Cross-Entropy Loss Section) for implementation details for below code snippet.
        Hint: Read the writeup to determine the shapes of all the variables.
        Note: Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]

        Ones_C = np.ones((self.C, 1))
        Ones_N = np.ones((self.N, 1))

        self.softmax = Softmax().forward(A)

        crossentropy = (-self.Y * np.log(self.softmax)) @ Ones_C
        sum_crossentropy_loss = Ones_N.T @ crossentropy
        mean_crossentropy_loss = sum_crossentropy_loss / self.N

        return mean_crossentropy_loss

    def backward(self):
        """
        Calculate the gradient of Cross-Entropy Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        Read the writeup (Hint: Cross-Entropy Loss Section) for implementation details for below code snippet.
        """
        dLdA = (self.softmax - self.Y) / self.N
        return dLdA
