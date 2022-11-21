import numpy as np

class EMOneDimGaussian:
    """
        EMOneDimGaussian is the implementation of EM algorithm for parameters estimation of 1D Gaussian Mixture model.
        See Sridharan, R. 2014. Gaussian mixture models and the EM algorithm for more details.
    """

    def __init__(self,n_gaussians = 2):
        """
        
        :param n_gaussians: number of gaussians
        :type n_gaussians: int 
        """
        self._n_gaussians = n_gaussians
        self._w = np.array([np.ones(n_gaussians) * 1/n_gaussians])
        self._mu = None
        self._sigma = np.array([np.ones(n_gaussians)])
        self._P = None
        self._Zsum = None
    
    def _N(self,Y):
        return 1/(np.sqrt(2*np.pi)*self._sigma.T) * np.exp(-(Y-self._mu.T)*(Y-self._mu.T)/(2*self._sigma.T*self._sigma.T))
        
    def _LL(self,Y):
        return -np.sum(np.log(np.sum(self._w.T * self._N(Y), axis = 0)))
    
    def fit(self,Y):
        """
        
        :param Y: 1D numpy array of values used for MLE
        :type Y: array-like 
        """
        self._P = np.zeros((self._n_gaussians,Y.shape[0]))
        self._Zsum = np.zeros(Y.shape[0])
        self._mu = np.array([np.quantile(Y, np.linspace(0,1,self._n_gaussians))])
        self._sigma *= np.max(Y) - np.min(Y)
        cur_LL = self._LL(Y)
        prev_LL = cur_LL
        n_iter = 10
        cur_iter = 0
        while cur_iter < n_iter:
            self._P = self._w.T * self._N(Y)
            self._P = self._P/np.sum(self._P, axis = 0)
            self._Zsum = np.sum(self._P,axis = 1)
            self._mu = np.array([np.sum(self._P * Y, axis = 1)/self._Zsum])
            self._sigma = np.array([np.sqrt(np.sum(self._P * (Y - self._mu.T) * (Y - self._mu.T), axis = 1)/self._Zsum)])
            self._w = np.array([self._Zsum/Y.shape[0]])
            cur_LL = self._LL(Y)
            if np.abs(cur_LL - prev_LL) < 1e-5:
                cur_iter += 1
            else:
                cur_iter = 0
            prev_LL = cur_LL