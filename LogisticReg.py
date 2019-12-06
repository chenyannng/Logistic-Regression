import numpy as np
from scipy import optimize

# fit logistic regression using maximum likelihood method
# data format: 2D numpy array that has last column as Y (Y equals 1 or 0), 1st column as bias term and 2nd to n-1 th column as X1 to Xn-1

class LogisticReg:

    def __init__(self, data):
        X = data[:, 0:-1]
        y = data[:, -1]
        self.X = X
        self.y = y
        self.fit_maximum_ll()
        
        
    @staticmethod
    def sigmoid(x):
        
        p = 1.0 / (1.0 + np.exp(-1.0 * x))

        return p


    def loglikelihood(self, theta):
        X = self.X
        z = np.matmul(X, theta)
        p = self.sigmoid(z)
        p1 = p[self.y == 1]
        p0 = 1 - p[self.y == 0]
        
        neg_log_ll = -np.sum(np.log(np.concatenate((p1,p0))))
        
        return neg_log_ll

    def fit_maximum_ll(self, theta0=None):
        # initialize theta      
        if theta0 is None:
            theta0 = 0.001 * np.ones(self.X.shape[1])
        xopt, fopt, iter, funcalls, warnflag = optimize.fmin(func=self.loglikelihood, x0=theta0, full_output=True)
        print('theta fitted:', xopt)
        self.theta_fit = xopt
        
        







