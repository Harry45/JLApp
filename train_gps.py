'''
Author: Arrykrishna Mootoovaloo
Email: a.mootoovaloo17@imperial.ac.uk
Status: Under development
Description: Simple script for training all the Gaussian Processes
'''

import numpy as np
from gp import gaussian_process
import helpers as hp

sigma = [-40.0]
train = True
nrestart = 5
bounds = np.repeat(np.array([[0.0, 5.0]]), 3, axis=0)
table = np.loadtxt('data_gp/' + 'cosmoTarget.txt')

for i in range(6):

    # define the GP
    gp = gaussian_process(table[:, [0, 1, i + 2]], sigma=sigma, train=train, nrestart=nrestart)

    # apply the prewhitening
    gp.transform()

    # fit for the kernel hyperparameters
    gp.fit(method='L-BFGS-B', bounds=bounds, options = {'ftol':1E-12, 'maxiter':500})

    # store the GP
    hp.store_pkl_file(gp, 'gps', 'gp_cosmo_' + str(i))

    print(gp.prediction(np.array([0.15, -0.80])))

    del gp
