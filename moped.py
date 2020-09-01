import numpy as np
import scipy.linalg as sl
import scipy.integrate as si
import pandas as pd

# our scripts
import helpers as hp 

class MOPED(object):
    '''
    Compress the data assuming we have already precomputed the MLE

    Inputs
    ------
    eps : step size for performing gradient (see MOPED algorithm)
    '''

    def __init__(self, eps=1E-4):

        # step size
        self.eps = eps

        # spped of light
        self.speedLight = 299792.

    def load_data(self):
        '''
        Load all important data for the analysis
        '''
        self.light_curve = pd.read_csv('data/jla_lcparams.txt', header=0, sep=' ')
        self.mu_observed = pd.read_csv('data/jla_mub.txt', header=0, sep=' ')
        self.sigma_mu = np.loadtxt('data/sigma_mu.txt')

        new_name_1 = ['z', list(self.mu_observed.keys()[1:])[0]]
        new_name_2 = ['name'] + list(self.light_curve.keys()[1:])

        self.mu_observed.columns = new_name_1
        self.light_curve.columns = new_name_2

        # mb
        self.data = self.light_curve['mb'].values

        # color
        self.color = self.light_curve['color'].values

        # x1
        self.x1 = self.light_curve['x1'].values

        # redshift
        self.redshift = self.light_curve['zcmb'].values

        # log stellar mass
        self.log_stellar_m = self.light_curve['3rdvar'].values

        # new basis
        self.new_basis = np.zeros_like(self.log_stellar_m)
        self.new_basis[self.log_stellar_m >= 10.0] = 1.0

        # number of observed data
        self.ndata = int(len(self.redshift))

        # design matrix for the systematic part
        self.psi = np.zeros((self.ndata, 4))

        self.psi[:, 0] = np.ones_like(self.log_stellar_m)
        self.psi[:, 1] = self.new_basis
        self.psi[:, 2] = -self.x1
        self.psi[:, 3] = self.color

    def load_solutions(self):
        '''
        Load all solutions, that is,

        mle : the maximum likelihood estimator

        params_cov : the parameter covariance matrix

        data_cov : the data covariance matrix
        '''

        # the maximum likelihood estimator
        self.mle = np.loadtxt('solutions/MLE.txt')

        # number of dimensions in our problem
        self.ndim = len(self.mle)

        # the parameter covariance matrix
        self.params_cov = np.loadtxt('solutions/params_cov.txt')

        # the data covariance matrix
        self.data_cov = np.loadtxt('solutions/dat_cov.txt')

    def load_moped(self):
        self.b = hp.load_arrays('compressed', 'B')
        self.y_s = hp.load_arrays('compressed', 'y')
        self.Bpsi = hp.load_arrays('compressed', 'Bpsi')

    def integrand(self, z, omega_m, w0):
        '''
        Compute the integrand (function)

        Inputs
        ------
        z (float) : the redshift

        omega_m (float) : omega matter

        w0 (float) : equation of state

        Returns
        -------
        fun (np.ndarray) : the integrand
        '''

        fun = 1.0 / np.sqrt(omega_m * (1.0 + z)**3 + (1.0 - omega_m) * (1.0 + z)**(3.0 * (w0 + 1)))

        return fun

    def integral(self, z, omega_m, w0):
        '''
        Calculates the integration in limits [0, z]

        Inputs
        ------
        z (float) : the redshift

        omega_m (float) : omega matter

        w0 (float) : equation of state

        Returns
        -------
        fun (np.ndarray) : the value of the integration
        '''
        factor = (1.0 + z) * self.speedLight / 70.0

        val, err = si.quad(self.integrand, 0.0, z, args=(omega_m, w0))

        d_l = 10**5 * factor * val

        return d_l

    def expected_app_mag(self, parameters):
        '''
        Calculates the expected apparent magnitude (MB) for all redshifts

        Inputs
        ------
        parmaeters (np.ndarray) : parameters[0] is omega_matter and parameters[1] is w0

        Returns
        -------
        mb_theory (np.ndarray) : the expected apparent magnitude (mb) for all redshifts
        '''

        mb_theory = np.zeros(self.ndata)
        for i in range(self.ndata):
            mb_theory[i] = 5.0 * np.log10(self.integral(self.redshift[i], parameters[0], parameters[1]))

        return mb_theory

    def light_curve_terms(self, parameters):
        '''
        Calculate the light curve terms (the cheap part of the process)

        Inputs
        ------
        parameters (np.ndarray) : defined as
            parameters[0]: M
            parameters[1]: deltaM
            parameters[2]: alpha
            parameters[3]: beta

        Returns
        -------
        sys_part (np.ndarray) : the expected theory for the systematic part
        '''

        term_1 = parameters[0] * np.ones_like(self.log_stellar_m)

        term_2 = parameters[1] * self.new_basis

        term_3 = -parameters[2] * self.x1

        term_4 = parameters[3] * self.color

        sys_part = term_1 + term_2 + term_3 + term_4

        return sys_part

    def theory(self, params):
        '''
        Calculates the expected theory (cosmology and systematic parts)

        Inputs
        ------
        params (np.ndarray) : defined as

            0: omega_matter
            1: w0
            2: M
            3: delta_M
            4: alpha
            5: beta

        Returns
        -------
        theory_calc (np.ndarray) : the model evaluation
        '''

        # cosmology
        mb_calc = self.expected_app_mag(params[0:2])

        # systematics
        syst_calc = self.light_curve_terms(params[2:])

        # theory
        theory_calc = mb_calc + syst_calc

        return theory_calc

    def compression(self, parameters, save = True):
        '''
        Performs the MOPED data compression.

        Inputs
        ------
        parameters (np.ndarray) : the point where we want to perform the compression

        save (bool) : if True, the MOPED vectors and compressed data will be stored in the compressed/ folder

        Returns
        -------
        bs (np.ndarray) : the MOPED vectors (ndim x ndata)

        y_alphas (np.ndarray) : the compressed data (ndim)
        '''

        grad = np.zeros((self.ndim, self.ndata))
        cinv_grad_mu = np.zeros((self.ndim, self.ndata))
        grad_cinv_grad = np.zeros(self.ndim)
        bs = np.zeros((self.ndim, self.ndata))

        for i in range(self.ndim):

            parameters_plus = np.copy(parameters)
            parameters_minus = np.copy(parameters)
            parameters_plus[i] = parameters_plus[i] + self.eps
            parameters_minus[i] = parameters_minus[i] - self.eps

            theory_plus = self.theory(parameters_plus)
            theory_minus = self.theory(parameters_minus)
            grad[i] = (theory_plus - theory_minus) / (2.0 * self.eps)

            cinv_grad_mu[i] = np.linalg.solve(self.data_cov, grad[i])
            grad_cinv_grad[i] = np.dot(grad[i], cinv_grad_mu[i])

        for i in range(self.ndim):

            if (i == 0):
                bs[i] = cinv_grad_mu[i] / np.sqrt(grad_cinv_grad[i])

            else:

                dummy_numerator = np.zeros((self.ndata, int(i)))
                dummy_denominator = np.zeros(int(i))

                for j in range(i):
                    dummy_numerator[:, j] = np.dot(grad[i], bs[j]) * bs[j]
                    dummy_denominator[j] = np.dot(grad[i], bs[j])**2

                bs[i] = (cinv_grad_mu[i] - np.sum(dummy_numerator, axis=1)) / \
                    np.sqrt(grad_cinv_grad[i] - np.sum(dummy_denominator))

        y_alphas = np.dot(bs, self.data)

        for i in range(self.ndim):
            for j in range(i + 1):
                if i == j:
                    print('Dot product between {0} and {1} is :{2:.10f}'.format(
                        i, j, np.dot(bs[i], np.dot(self.data_cov, bs[j]))))

        # delete unwanted (large) quantities
        del self.data_cov

        # store the MOPED vectors
        self.b = bs

        # store the compressed data
        self.y_s = y_alphas

        # store the MOPED vectors dot with the basis (6 x 4 matrix only)
        self.Bpsi = np.dot(self.b, self.psi)

        if save:
            hp.store_arrays(self.b, 'compressed', 'B')
            hp.store_arrays(self.y_s, 'compressed', 'y')
            hp.store_arrays(self.Bpsi, 'compressed', 'Bpsi')
        return bs, y_alphas

    def theory_moped(self, parameters):
        '''
        Calculates the compressed theory using the MOPED vectors

        Inputs
        ------
        parameters (np.ndarray) : the input parameters defined as

            0: omega_matter
            1: w0
            2: M
            3: delta_M
            4: alpha
            5: beta

        Returns
        -------
        expectation (np.ndarray) : the compressed theory of size ndim (number of parameters)
        '''

        expectation = np.dot(self.b, self.theory(parameters))

        return expectation

    def moped_cosmology(self, parameters):
        '''
        Calculates the compressed theory for the expensive part only (cosmology)

        Inputs
        ------
        parmaeters (np.ndarray) : defined as
            0: omega_matter
            1: w0

        Returns
        -------
        expectation_cosmo (np.ndarray) : the compressed (cosmology) theory of size ndim
        '''

        expectation_cosmo = np.dot(self.b, self.expected_app_mag(parameters))

        return expectation_cosmo

    def moped_systematics(self, parameters):
        '''
        Calculates the compressed theory for the systematics part (very cheap)

        Inputs
        ------
        parmaeters (np.ndarray) : defined as

            0: M
            1: delta_M
            2: alpha
            3: beta

        Returns
        -------
        expectation_syst (np.ndarray) : the compressed (systematics) theory of size ndim
        '''

        expectation_syst = np.dot(self.Bpsi, parameters)

        return expectation_syst


if __name__ == '__main__':

    moped_module = MOPED(eps=1E-6)
    moped_module.load_data()
    moped_module.load_solutions()
    
    # done once    
    # moped_module.compression(moped_module.mle)
    moped_module.load_moped()
