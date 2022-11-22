# -*- coding: utf-8 -*-

import tensorflow as tf
import gpflow

class LinearGibbs(gpflow.kernels.Kernel):
    """LinearGibbs is the GPflow based implementation of Gibbs kernel with the variable lengthscale which changes linearly. 

    .. math:: k(x, y) = C \\sqrt{\\frac{2l(x) l(y)}{l(x)^2+l(y)^2}} e^{-\\frac{(x-y)^2}{l(x)^2+l(y)^2}}, l(x) = Ax + B

    The kernel is applicable only to one-dimensional problems.

    
    """
    def __init__(self,B_min = 1e-6):
        """
        
        :param B_min: minimum value of lengthscale intercept. Defaults to 1e-6
        :type B_min: float
        """
        super().__init__(active_dims=[0])
        self.A = gpflow.Parameter(1.0)
        self.B = gpflow.Parameter(max(1.0,B_min) + 1e-6, transform = gpflow.utilities.positive(B_min))
        self.variance = gpflow.Parameter(1.0, transform = gpflow.utilities.positive())

    def _l(self,X):
        """Defines l(X) 

        :param X: X 
        :type X: Tensor
        :return: l(X)
        :rtype: Tensor
        """
        return self.A * X  + self.B
       
    def K(self, X, Y=None):
        """K is the covariance matrix 
        
        :param X: X 
        :type X: Tensor
        :param Y: Y, defaults to None
        :type Y: Tensor, optional
        :return: covariance matrix K(X,Y), or K(X,X) if Y = None
        :rtype: Tensor
        """
        if Y is None:
            Y = X
        lx = self._l(X) #X
        ly = tf.transpose(self._l(Y)) #Y.T
        mul_ls = tf.matmul(lx, ly)
        sq_sum_ls = tf.math.square(lx) + tf.math.square(ly)
        res = X - tf.transpose(Y)
        r2 = tf.math.square(res)
        return self.variance * 2 *mul_ls/sq_sum_ls  * tf.exp(-r2/sq_sum_ls)  # this returns a 2D tensor

    def K_diag(self, X):
        """K_diag is the diagonal of kernel K

        :param X: X
        :type X: Tensor
        :return: Diagonal of kernel
        :rtype: Tensor
        """
        return tf.fill(tf.shape(X)[:-1], self.variance)