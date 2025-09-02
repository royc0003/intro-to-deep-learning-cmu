import numpy as np


class BatchNorm1d:
    """
    Create your own mytorch.nn.BatchNorm1d!
    Read the writeup (Hint: Batch Normalization Section) for implementation details for the BatchNorm1d class.
    Hint: Read all the expressions given in the writeup and be CAREFUL to re-check your code.
    """

    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during inference.
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        Forward pass for batch normalization.
        :param Z: batch of input data Z (N, num_features).
        :param eval: flag to indicate training or inference mode.
        :return: batch normalized data.

        Read the writeup (Hint: Batch Normalization Section) for implementation details for the BatchNorm1d forward.
        Note: The eval parameter indicate whether it's training phase or the inference phase of the problem.
        Check the values you need to recompute when eval = False.
        """
        self.Z = Z
        self.N = Z.shape[0]
        self.M = (np.sum(self.Z, axis=0)) / self.N # note axis = 0 refers to column
        self.V = np.sum((self.Z - self.M)**2, axis=0) / self.N
        self.ones = np.ones((self.N, 1))

        if eval == False:
            # training mode
            self.NZ = (self.Z - self.M) / np.sqrt(self.V + self.eps)
            self.BZ = self.BW * self.NZ + self.Bb

            self.running_M = (self.alpha * self.running_M) + (1 - self.alpha) * self.M
            self.running_V = (self.alpha * self.running_V) + (1 - self.alpha) * self.V

        else:
            # inference mode
            self.NZ = (self.Z - self.running_M) / np.sqrt(self.running_V + self.eps)  # Use running averages
            self.BZ = self.BW * self.NZ + self.Bb  # Scale and shift

        return self.BZ

    def backward(self, dLdBZ):
        """
        Backward pass for batch normalization.
        :param dLdBZ: Gradient loss wrt the output of BatchNorm transformation for Z (N, num_features).
        :return: Gradient of loss (L) wrt batch of input batch data Z (N, num_features).

        Read the writeup (Hint: Batch Normalization Section) for implementation details for the BatchNorm1d backward.
        """
        self.dLdBb = None  # TODO: Sum over the batch dimension.
        self.dLdBW = None  # TODO: Scale gradient of loss wrt BatchNorm transformation by normalized input NZ.

        dLdNZ = None  # TODO: Scale gradient of loss wrt BatchNorm transformation output by gamma (scaling parameter).

        dLdV = None  # TODO: Compute gradient of loss backprop through variance calculation.
        dNZdM = None  # TODO: Compute derivative of normalized input with respect to mean.
        dLdM = None  # TODO: Compute gradient of loss with respect to mean.

        dLdZ = None  # TODO: Compute gradient of loss with respect to the input.
        raise NotImplemented  # TODO - What should be the return value?
