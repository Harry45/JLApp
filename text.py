content = {

	'motivation' : 'In this post, we use the formalism developed in <a href="https://doi.org/10.1093/mnras/staa2102">Mootoovaloo et al. 2020</a> to show that we can speed up parameter inference by a factor of around 20-40 compared to the full sumulator when applied to the JLA dataset. The speed of inference depends on whether we use the mean only or both the mean and variance of the emulator. This tool should not only be viewed as an approach to just speed up computation but also as an inference mechanism to obtain posterior densities of both cosmological and nuisance parameters.',
	
	'cosmology_1' : 'We briefly summarize the main concepts behind supernova cosmology, which is one of the main proxies to learn about the late Universe. The Joint Lightcurve Analysis (JLA) supernova dataset is a compilation of a number of supernova catalogues from various surveys. The sample consists of 740 Type Ia supernova and relevant quantities crucial for our analysis comprise of the apparent magnitudes, redshifts and light curve parameters: colour and stretch.',

	'notation' : 'Here we briefly summarise the notations we will be using throughout this post.',

	'related_work_1' : 'Machine Learning techniques have previously been explored to infer cosmological and/or nuisance parameters using the JLA dataset. Some references along the same line are in the table below.',


	'related_work_2' : r"""
					| Paper | Number of parameters | Number of simulations|
					| --- | :-: |:-:|
					|[Alsing et al. 2019](https://doi.org/10.1093/mnras/stz1960)| $6$ |$\mathcal{O}(1000)$|
					|[Alsing et al. 2019](https://doi.org/10.1093/mnras/stz1900)| $6$ |$\mathcal{O}(1000)$|
					|[Leclercq (2018)](https://doi.org/10.1103/PhysRevD.98.063511)| $2$ |$6000$|
					|[Alsing et al. 2018](https://doi.org/10.1093/mnras/sty819)| $6$ | $20000$ |
					""",

	'related_work_3' : '<br>In this work, we take a completely different approach. We focus entirely on the most expensive part of the model (the cosmology part). In other words, we emulate the MOPED coefficients related to the cosmology part only using Gaussian Processes. As a result, we require only 300 forward expensive simulations only.',

	'cosmology_2' : 'Type Ia supernova being <i>standard candles</i>, the expected apparent magnitude is given by',

	'cosmology_3' : r"""
						$$
						m_{B}=5\,\textrm{log}_{10}D_{L}\left(z\right)+M_{B}+\delta Ms-\alpha x_{1}+\beta C
						$$

						where

						$$
						s=\begin{cases}
						\begin{array}{c}
						1\\
						0
						\end{array} & \begin{array}{c}
						\textrm{log}_{10}M_{\textrm{stellar}}>10\\
						\textrm{otherwise}
						\end{array}\end{cases}
						$$

						and 

						$$
						D_{L}=\dfrac{\left(1+z\right)c}{H_{0}}\int_{0}^{z}\dfrac{dz'}{\sqrt{\Omega_{m}\left(1+z'\right)^{3}+\left(1-\Omega_{m}\right)\left(1+z'\right)^{3\left(w_{0}+1\right)}}}
						$$

						where $H_{0}$ is the Hubble constant.
					""",

	'moped_1' : 'The MOPED algorithm essentially forms linear combination of the data using a set of (normalised and orthogonal) vectors. Each vector is of size equal to the number of data points.',

	'moped_2' : r"""
					The first MOPED vector is given by

					$$
					\boldsymbol{b}_{1}=\dfrac{\textsf{\textbf{C}}^{-1}\boldsymbol{\mu}_{,1}}{\sqrt{\boldsymbol{\mu}_{,1}^{\textrm{T}}\textsf{\textbf{C}}^{-1}\boldsymbol{\mu}_{,1}}}
					$$

					and the subsequent ones are given by

					$$
					\boldsymbol{b}_{i}=\dfrac{\textsf{\textbf{C}}^{-1}\boldsymbol{\mu}_{,i}-\sum\limits_{j=1}\limits^{i-1}(\boldsymbol{\mu}_{,i}^{\textrm{T}}\boldsymbol{b}_{j})\boldsymbol{b}_{j}}{\sqrt{\boldsymbol{\mu}_{,i}^{\textrm{T}}\textsf{\textbf{C}}^{-1}\boldsymbol{\mu}_{,i}-\sum\limits_{j=1}\limits^{i-1}(\boldsymbol{\mu}_{,i}^{\textrm{T}}\boldsymbol{b}_{j})^{2}}}
					$$

					where $\boldsymbol{\mu}$ is the theory, $\boldsymbol{m}_{B}$ in this case. The compressed data and theory are

					$$
					\boldsymbol{y} = \textsf{\textbf{B}}^{\textrm{T}}\boldsymbol{x}
					$$

					and 

					$$
					\langle\boldsymbol{y}\rangle = \textsf{\textbf{B}}^{\textrm{T}}\boldsymbol{\mu}.
					$$

				""",

	'gp_1' : r"""
				A Gaussian Process is a stochastic process, which has a set of variables which is jointly distributed as a multivariate normal distribution. In essence, it is a distribution over function with a continuous domain. In our case, we consider a regression of the form:
			""",

	'gp_2' : r"""
				$$
				\boldsymbol{y} = \boldsymbol{f} + \boldsymbol{\epsilon},
				$$

				and the posterior distribution of $\boldsymbol{f}$ is analytical and the predictive distribution is given by

				$$
				p(f_{*}|\boldsymbol{\theta}_{*},\boldsymbol{y}) = \mathcal{N}\left(\boldsymbol{k}_{*}^{\textrm{T}}\textsf{\textbf{K}}_{y}^{-1}\boldsymbol{y}, k_{**}-\boldsymbol{k}_{*}^{\textrm{T}}\textsf{\textbf{K}}_{y}^{-1}\boldsymbol{k}_{*}\right)
				$$

				where 
				- $k(\cdot,\cdot)$ refers to the kernel computed for each pair of inputs and 
				- $\textsf{\textbf{K}}_{y} = \textsf{\textbf{K}} + \mathbf{\Sigma}$ and $\mathbf{\Sigma}$ is the noise covariance matrix.
				""",

	'gp_3': r"""
			An important choice is the kernel hyperparameter(s) and this is learned by maximising the marginal likelihood, which is given by
			""",

	'gp_4' : r"""
			$$
			\textrm{log }p(\boldsymbol{y}) = -\frac{1}{2}(\boldsymbol{y}-\boldsymbol{m})^{\textrm{T}}\textsf{\textbf{K}}_{y}^{-1}(\boldsymbol{y}-\boldsymbol{m})-\frac{1}{2}\textrm{log}|\textsf{\textbf{K}}_{y}|+\textrm{constant}.
			$$
			where $\boldsymbol{m}$ is the mean of the GP and is usually set to zero.
			""",

	'application_1' : r"""
					From the flowchart above, there are various possibilities to run this application. In this case, we focus only on the left side, that is, with compression only, although we will compare all results with the samples obtained from the uncompressed data.<br><br>The sampling procedure can be done by either choosing the emulator or the simulator. However, on a high-end desktop, the simulator takes almost 1 hour to generate 120 000 samples and we do not recommend using it, although it can also be used to confirm existing result, that is, result corresponding to the uncompressed dataset.<br><br>In addition, if the emulator is chosen, we can either include or exclude the GP uncertainty. The latter is expected to take longer to run compared to when using the mean of the GP. Moreover, we can also specify the number of samples per chain (walkers) and the burn-in fraction (as a percentage of the chain length). Finally, the user is also able to adjust the step sizes for each parameter to enhance the sampling procedure. By default, they are set to the best possible values in our experiment.
					""",

	'results' : r"""
				At this point, please use the left panel to configure the sampler and once this is done, click on 'Generate MCMC Samples'. Once the samples (either from the emulator with or without the GP uncertainty or the simulator) are generated, the marginalised posterior distributions of all parameters (cosmological and nuisance) are shown below.
				""",

	'conclusions':r"""
					In this interactive application, we have shown how we can build an emulator by combining Gaussian Process formalisms and the MOPED data compression algorithm. On our high-end computer, the full simulator takes around 1 hour to generate 120 000 MCMC samples while the GP with the mean takes around 1 minute 30 seconds. Note that this might differ, depending on the compute engine. However, there are two main take-home messages. First, the emulator is faster compared to the simulator and the same idea extends to other more expensive applications. Second, we have a new method for performing parameter inference.
					""",
	'evidence': r"""
				Once we have the MCMC samples, we can use MCEvidence developed by <a href="https://arxiv.org/pdf/1704.03472.pdf">Heavens et al. 2017</a> to quickly calculate the evidence of the emulator.
				"""


}

