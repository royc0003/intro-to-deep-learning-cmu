import numpy as np


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

        Ones_C = None  # TODO
        Ones_N = None  # TODO

        self.softmax = None  # TODO - Can you reuse your own softmax here, if not rewrite the softmax forward logic?

        crossentropy = None  # TODO
        sum_crossentropy_loss = None  # TODO
        mean_crossentropy_loss = sum_crossentropy_loss / self.N

        raise NotImplemented  # TODO - What should be the return value?

    def backward(self):
        """
        Calculate the gradient of Cross-Entropy Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        Read the writeup (Hint: Cross-Entropy Loss Section) for implementation details for below code snippet.
        """
        dLdA = None  # TODO
        raise NotImplemented  # TODO - What should be the return value?
