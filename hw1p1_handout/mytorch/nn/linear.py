import numpy as np


class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros (Hint: check np.zeros method)
        Read the writeup (Hint: Linear Layer Section Table) to identify the right shapes for `W` and `b`.
        in_features (C_in): number of input features - 2
        out_features (C_out): number of output features - 3
        """
        self.debug = debug
        # W = C_out x C_in
        self.W = np.zeros((out_features, in_features))
        # b = C_out x 1
        self.b = np.zeros((out_features, 1))

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output of linear layer with shape (N, C1)

        Read the writeup (Hint: Linear Layer Section) for implementation details for `Z`
        """
        self.A = A
        self.N = self.A.shape[0]  # store the batch size parameter of the input A

        # Think how can `self.ones` help in the calculations and uncomment below code snippet.
        self.ones = np.ones((self.N, 1))
        
        Z = self.A @ self.W.T + self.ones @ self.b.T
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (N, C1)
        :return: Gradient of loss wrt input A (N, C0)

        Read the writeup (Hint: Linear Layer Section) for implementation details below variables.
        """
        dLdA = dLdZ @ self.W  # TODO
        self.dLdW = dLdZ.T @ self.A  # TODO
        self.dLdb = dLdZ.T @ self.ones  # TODO

        if self.debug:
            self.dLdA = dLdA

        return dLdA
