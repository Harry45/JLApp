import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from getdist import plots, MCSamples
import matplotlib.pylab as plt

def plot_triangle(sim_samples, gp_samples, burnin_frac=0.1, emulator = True, gp_error=False):

    if gp_error and emulator:
        label_gp = 'GP (Error)'

    elif emulator and not gp_error:
       label_gp = 'GP (Mean)'

    else:
        label_gp = 'Simulator (MOPED)'

    ndim = sim_samples.shape[-1]

    names = ["x%s" % i for i in range(ndim)]

    labels = [r"\Omega_{m}", r"w_{0}", r"M_{B}", r"\delta M", r"\alpha", r"\beta"]

    # for the simulator
    burnin = int(burnin_frac * sim_samples.shape[1])
    samples_exact = sim_samples[:, burnin:, :].reshape((-1, ndim))
    cut_samps = samples_exact[samples_exact[:, 0] >= 0.0, :]
    samples1 = MCSamples(samples=cut_samps, names=names, labels=labels, ranges={'x0': (0.0, None)})

    # for the emulator
    burnin = int(burnin_frac * gp_samples.chain.shape[1])
    samples_emu = gp_samples.chain[:, burnin:, :].reshape((-1, ndim))
    cut_samps = samples_emu[samples_emu[:, 0] >= 0.0, :]
    samples2 = MCSamples(samples=cut_samps, names=names, labels=labels, ranges={'x0': (0.0, None)})

    # setups for plotting
    sim_color = '#EEC591'
    gp_color = 'Blue'
    alpha_tri = 0.1
    red_patch = mpatches.Patch(color=sim_color, label='Simulator', alpha=alpha_tri)
    gp_line = Line2D([0], [0], color=gp_color, linewidth=3, linestyle='--', label=label_gp)
    rec_leg = [red_patch, gp_line]

    contours = np.array([0.68, 0.95])

    G = plots.getSubplotPlotter(subplot_size=3.5)
    samples1.updateSettings({'contours': [0.68, 0.95]})

    G.triangle_plot([samples1], filled=True, line_args={'lw': 3, 'color': sim_color}, contour_colors=[sim_color])
    G.settings.num_plot_contours = 2
    plt.legend(handles=rec_leg, loc='best', prop={'size': 25}, bbox_to_anchor=(0.7, 6.0), borderaxespad=0.)
    G.settings.alpha_filled_add = alpha_tri
    for i in range(0, 6):
        for j in range(0, i + 1):
            if i != j:
                ax = G.subplots[i, j]

                a, b = G.get_param_array(samples2, ['x' + str(j), 'x' + str(i)])
                density = G.sample_analyser.get_density_grid(samples2, a, b)
                density.contours = density.getContourLevels(contours)
                contour_levels = density.contours
                ax.contour(
                    density.x,
                    density.y,
                    density.P,
                    sorted(contour_levels),
                    colors=gp_color,
                    linewidths=3,
                    linestyles='--')

                ax.tick_params(labelsize=20)
                ax.yaxis.label.set_size(20)
                ax.xaxis.label.set_size(20)
            else:
                ax = G.subplots[i, j]

                dense = samples2.get1DDensity('x' + str(i))
                dense.normalize(by='max')
                ax.plot(dense.x, dense.P, lw=3, c=gp_color, linestyle='--')

                ax.tick_params(labelsize=20)
                ax.yaxis.label.set_size(20)
                ax.xaxis.label.set_size(20)
    plt.savefig('images/triangle_plot.jpg', bbox_inches='tight', transparent = False)
    plt.close()
