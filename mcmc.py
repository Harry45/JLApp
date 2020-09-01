'''
Author: Arrykrishna Mootoovaloo
Email: a.mootoovaloo17@imperial.ac.uk
Status: Under development
Description: MCMC script for sampling the posterior
'''
import types
import importlib.machinery as im
import numpy as np
import pandas as pd
import dill
import emcee

# our scripts
import prior as pr
import moped as md
import helpers as hp


class MCMC(object):
    '''
    MCMC module for sampling the posterior distribution either:
        - using the fully theory
        - using the emulator
            - with the mean only
            - with the mean and GP uncertainty

    Inputs
    ------
    settings (external file) : a setting file which contains the prior defined

    emulator (bool) : if True, the emulator will be used

    gp_error (bool) : if True, the uncertainty of the GP will be used in the likelihood calculation

    '''

    def __init__(self, settings, emulator=True, gp_error=False):

        self.emulator = emulator
        self.gp_error = gp_error

        # MOPED part
        self.moped_module = md.MOPED()
        self.moped_module.load_data()
        self.moped_module.load_solutions()
        self.moped_module.load_moped()

        # priors
        loader = im.SourceFileLoader(settings, settings)
        self.settings = types.ModuleType(loader.name)
        loader.exec_module(self.settings)

        # prior
        self.all_priors = pr.distributions(self.settings.prior)

        # total dimension of the problem
        self.ndim = self.moped_module.ndim

    def load_gps(self, folder_name='gps/'):
        '''
        Load all the GPs which have been pre-trained.

        Inputs
        ------
        folder_name (str) : name of the folder where the GPs are stored

        Returns
        -------
        all_gps (list) : a list of all the GP module

        '''
        all_gps = []

        for i in range(self.moped_module.ndim):
            gp = hp.load_pkl_file(folder_name, 'gp_cosmo_' + str(i))
            all_gps.append(gp)

        self.all_gps = all_gps

        return all_gps

    def predictions(self, point):
        '''
        Calculate the GP prediction at a point in parameter space

        Inputs
        ------
        point (np.ndarray) : a point in parameter space

        Returns
        -------
        mean_pred (np.ndarray) : the mean prediction (1 for each GP)

        If we want the GP error, the following will be returned along with the mean prediction

        var_pred (np.ndarray) : the predicted variance (1 for each GP)
        '''

        mean_pred = np.zeros(self.ndim)

        if self.gp_error:
            var_pred = np.zeros(self.ndim)

            for i in range(self.ndim):
                mean_pred[i], var_pred[i] = self.all_gps[i].prediction(point, returnvar=self.gp_error)

            return mean_pred, var_pred

        else:
            for i in range(self.ndim):
                mean_pred[i] = self.all_gps[i].prediction(point, returnvar=self.gp_error)
            return mean_pred

    def loglike(self, parameters):
        '''
        Calculates the log-likelihood at a point in parameter space

        Inputs
        ------
        parameters (np.ndarray) : a point within the prior box

        Returns
        -------
        loglike (np.ndarray) : the log-likelihood
        '''

        if self.emulator:

            m = self.predictions(parameters[0:2])
            m = m + self.moped_module.moped_systematics(parameters[2:])

            if self.gp_error:
                m, v = self.predictions(parameters[0:2])
                m = m + self.moped_module.moped_systematics(parameters[2:])
                log_likelihood = -0.5 * np.sum(((m - self.moped_module.y_s)**2 / (np.ones(self.ndim) + v)))

            else:
                log_likelihood = -0.5 * np.sum((m - self.moped_module.y_s)**2)

        else:
            m = self.moped_module.theory_moped(parameters)
            log_likelihood = -0.5 * np.sum((m - self.moped_module.y_s)**2)

        return log_likelihood

    def logprior(self, parameters):

        '''
        Calculates the log-prior at a point in parameter space

        Inputs
        ------
        parameters (np.ndarray) : a point within the prior box

        Returns
        -------
        log_prior (np.ndarray) : the log-poterior

        '''
        prior = [self.all_priors[i].logpdf(parameters[i]) for i in range(self.ndim)]

        log_prior = np.sum(prior)

        # do not call the likelihood if point lies outside the prior
        return log_prior

    def logpost(self, parameters):

        '''
        Calculates the log-posterior at a point in parameter space

        Inputs
        ------
        parameters (np.ndarray) : a point within the prior box

        Returns
        -------
        loglike (np.ndarray) : the log-poterior

        '''

        # calculates the log prior
        log_prior = self.logprior(parameters)

        if not np.isfinite(log_prior):
            return -np.inf

        # calculates the log likelihood
        log_likelihood = self.loglike(parameters)

        # calculates the log posterior
        log_posterior = log_likelihood + log_prior

        return log_posterior

    def sampling(self, eps, n_samples=5, n_walkers=12, file_name = None):
        '''
        We sample the posterior using EMCEE.

        Inputs
        ------
        eps (np.ndarray) : the step size (see EMCEE documentation)

        n_samples (int) : the number of samples we want per walker

        n_walkers (int) : the number of walkers

        file_name (str) : if a file name is specified, the samples will be saved in the samples/ folder

        Returns
        -------

        sampler (class) : the whole EMCEE class which contains various information such as:
            - acceptance_franction (per walker)
            - flatlnprobability
            - lnprobability
        '''

        pos = [self.moped_module.mle + eps * np.random.randn(self.ndim) for i in range(n_walkers)]

        sampler = emcee.EnsembleSampler(n_walkers, self.ndim, self.logpost)

        sampler.run_mcmc(pos, n_samples)

        if file_name:

            hp.delete_file('samples', file_name)

            # del all Gaussian Processes (for small file size)
            del self.all_gps

            # delete the data covariance matrix (for small file size)
            del self.moped_module.data_cov

            hp.store_pkl_file(sampler, 'samples', file_name)

        return sampler


if __name__ == '__main__':

    mcmc_module = MCMC(settings='settings', emulator=True, gp_error=False)
    mcmc_module.load_gps('gps/')

    test_point = np.array([0.1782, -0.7105, -19.0393, -0.0516, 0.1246, 2.6184])

    print(mcmc_module.logpost(test_point))

    # eps_ = np.array([0.1158, 0.2136, 0.0254, 0.0229, 0.0061, 0.0704])

    # samples = mcmc_module.sampling(eps_, 5, 12)
