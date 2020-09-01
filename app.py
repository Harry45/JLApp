import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pylab as plt
import matplotlib
from PIL import Image
import time

# our scripts
import mcmc as mc
import plot as pt
import text as tx
from MCEvidence import MCEvidence

def timer(start,end):
    
    hours, rem = divmod(end-start, 3600)

    minutes, seconds = divmod(rem, 60)

    time_list = np.array([int(hours), int(minutes), int(seconds)])

    return time_list

def md_justify(text: str):
    '''
    Function to write HTML text, aligned justify

    Note: Cannot write latex inline (to find a way to improve this)

    Arguments:
    text (str) : the text we want to display
    '''
    st.markdown("<p align='justify'>" + text + "</p>", unsafe_allow_html=True)


def input_parameters():
    '''
    All inputs to the application
    '''
    sample_button = st.sidebar.button('Generate MCMC Samples')

    # Emulator or Simulator
    method = st.sidebar.selectbox('Method', ['Emulator', 'Simulator'])

    if method == 'Emulator':
        emu = True
    else:
        emu = False

    # GP error (True or False)

    gp_error = st.sidebar.selectbox('GP Error', ['No', 'Yes'])

    if gp_error == 'Yes':
        gpe = True
    else:
        gpe = False

    # number of samples
    nsamples = st.sidebar.number_input('Number of samples', min_value=100, max_value=20000, step=100)
    
    # number of walkers
    nwalkers = st.sidebar.number_input('Number of walkers (chains)', min_value=12, max_value=24, step=2)
    
    # burnin fraction
    burnin = st.sidebar.number_input('Burn-in (%)', min_value=10, max_value=50, step=2)

    st.sidebar.header(r'Step Sizes, $\boldsymbol{\epsilon}$')

    s1 = st.sidebar.slider('Step size 1', 0.5, 1.5, 1.158)
    
    s2 = st.sidebar.slider('Step size 2', 1.0, 3.0, 2.136)
    
    s3 = st.sidebar.slider('Step size 3', 0.2, 0.3, 0.254)
    
    s4 = st.sidebar.slider('Step size 4', 0.2, 0.3, 0.229)
    
    s5 = st.sidebar.slider('Step size 5', 0.01, 0.10, 0.061)
    
    s6 = st.sidebar.slider('Step size 6', 0.1, 1.0, 0.704)

    # the step size
    eps = 0.1 * np.array([s1, s2, s3, s4, s5, s6])

    # a dictionary to store all quantities
    params = {'nwalkers': nwalkers, 'nsamples': nsamples, 'burnin': 0.01 * burnin, 'gp_error': gpe, 'emulator':emu, 'eps': eps}

    return sample_button, params

def run_mcmc(nsamples, nwalkers, eps, emulator = True, gp_error = False):
    '''
    Generate samples from the posterior distribution

    Inputs
    ------
    nsamples (int) : number of MCMC samples

    nwalkers (int) : the number of walkers/chain

    eps (np.ndarray) : the step size within the MCMC sampler

    Returns
    -------
    samples (np.ndarray) : the samples of size (nwalkers x nsamples x ndim)
    '''

    # MCMC Module for the emulator
    mcmc_module = mc.MCMC(settings='settings', emulator=emulator, gp_error=gp_error)

    # Load all Gaussian Processes
    mcmc_module.load_gps('gps/')

    if gp_error and emulator:
        mcmc_file_name = 'mcmc_error_gp'

    elif emulator and not gp_error:
       mcmc_file_name = 'mcmc_mean_gp'

    else:
        mcmc_file_name = 'mcmc_simulator_moped'

    # Generate MCMC samples
    samples = mcmc_module.sampling(eps, nsamples, nwalkers, mcmc_file_name)

    # concatenate first mid walkers and second mid walkers
    mid = int(nwalkers/2.)

    # number of parameters
    nparams = samples.chain.shape[-1]

    # process samples - split into two chains
    chain0 = samples.chain[0:mid].reshape((-1, nparams))
    chain1 = samples.chain[mid:].reshape((-1, nparams))

    lnP0 = samples.lnprobability[0:mid].flatten()
    lnP1 = samples.lnprobability[mid:].flatten()

    table0 = np.c_[np.ones(len(lnP0)), -lnP0, chain0]
    table1 = np.c_[np.ones(len(lnP1)), -lnP1, chain1]

    # save MCMC samples
    np.savetxt('samples/emulator_0', table0)
    np.savetxt('samples/emulator_1', table1)  

    # define global variable for number of samples in each chain
    global nsamples_emu

    # number of samples in each chain
    nsamples_emu = len(lnP1)

    return samples

def calc_evidence():

    # some setups for calculating the evidence

    split = True
    autothin = False
    setthinning = 0.5
    thinning = setthinning
    burnfraction= 0.

    if(split):
        nchains = 2
    else:
        nchains = 1

    chainfile_emu = 'samples/emulator_*'

    evi_emu = MCEvidence(chainfile_emu,split=split,kmax=0,verbose=0,priorvolume=1.,thinlen=thinning,burnlen=int(burnfraction*nsamples_emu),debug=False).evidence()

    return evi_emu

# ---------------------------------------------------------------------------------------------------------------------
# Load samples for the simulator
samples_sim = np.load('samples/mcmc_simulator.npz')['arr_0']

sample_button, params = input_parameters()

# ---------------------------------------------------------------------------------------------------------------------
st.title("MOPED and GP Emulator for JLA")
st.text('Arrykrishna Mootoovaloo (a.mootoovaloo17@imperial.ac.uk)')

# ---------------------------------------------------------------------------------------------------------------------
md_justify(tx.content['motivation'])
st.info('If you would like to skip ahead and test the app, see section **Results** below.')

# ---------------------------------------------------------------------------------------------------------------------
st.header('Cosmology')
image = Image.open('images/data_and_cov.png')
st.image(image, caption='Figure 1: Data (left panel) and covariance matrix (right panel)', use_column_width=True)
md_justify(tx.content['cosmology_1'])

st.subheader('Notations')
md_justify(tx.content['notation'])
image = Image.open('images/table_notations.png')
st.image(image, caption='Table 1: Symbols and notations to be used in this post', use_column_width=True)
md_justify(tx.content['cosmology_2'])
st.write(tx.content['cosmology_3'])

st.subheader('Related Work')
md_justify(tx.content['related_work_1'])
st.write(tx.content['related_work_2'])
md_justify(tx.content['related_work_3'])

# ---------------------------------------------------------------------------------------------------------------------
st.header('Data Compression')
md_justify(tx.content['moped_1'])
st.write(tx.content['moped_2'])
# ---------------------------------------------------------------------------------------------------------------------
st.header('Gaussian Process')
md_justify(tx.content['gp_1'])
st.write(tx.content['gp_2'])
md_justify(tx.content['gp_3'])
st.write(tx.content['gp_4'])


# ---------------------------------------------------------------------------------------------------------------------
st.header('Application')
image = Image.open('images/flowchart.jpg')
st.image(image, caption='Figure 2: Simple flowchart for different sampling possibilities', use_column_width=True)

md_justify(tx.content['application_1'])

# ---------------------------------------------------------------------------------------------------------------------
st.header('Results')
md_justify(tx.content['results'])

# Generate samples for emulator

if sample_button:

    start = time.time()
    samples = run_mcmc(nsamples=params['nsamples'], nwalkers=params['nwalkers'], eps=params['eps'], emulator = params['emulator'], gp_error = params['gp_error'])
    end = time.time()
    time_taken = timer(start = start, end = end)

    # Display time taken
    st.subheader('Performance')
    st.success('Sampling completed successfully')
    st.text('Total Number of MCMC Samples : {}'.format(samples.chain.shape[0]*samples.chain.shape[1]))
    st.text('Hour   : {}'.format(time_taken[0]))
    st.text('Minute : {}'.format(time_taken[1]))
    st.text('Second : {}'.format(time_taken[2]))

    # make triangle plot
    pt.plot_triangle(samples_sim, samples, burnin_frac=params['burnin'], emulator = params['emulator'], gp_error=params['gp_error'])
    image = Image.open('images/triangle_plot.jpg')
    st.image(image, caption='Triangle plot for the cosmological and nuisance parameters', use_column_width=True)

    st.subheader('Evidence Calculations')
    md_justify(tx.content['evidence'])

    evi_emulator = np.around(calc_evidence(), 4)[0]

    # build  dataframe to display the evidence values
    evidences = np.array([evi_emulator,-19.2312]).reshape(1,2)

    df = pd.DataFrame(evidences, columns = ['Emulator', 'Simulator'])
    df.index = ['Evidence']
    st.table(df)



# ---------------------------------------------------------------------------------------------------------------------
st.header('Conclusions')
md_justify(tx.content['conclusions'])

# ---------------------------------------------------------------------------------------------------------------------


