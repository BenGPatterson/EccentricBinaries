import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

def plot_min_max_lines(samples_e, samples_SNR, match_key, e_vals, max_line, min_line, max_interp, min_interp, e_max=0.4):

    # Compute kde of ecc-SNR at points on 2d grid
    kde_samples = np.array([samples_e, samples_SNR])
    kernel = gaussian_kde(kde_samples)
    kde_x, kde_y = np.mgrid[np.min(samples_e):np.max(samples_e):101j, np.min(samples_SNR):np.max(samples_SNR):101j]
    inside_lines = (kde_y.flatten() <= max_interp(kde_x.flatten())*1.1)*(kde_y.flatten() >= min_interp(kde_x.flatten())*0.9)
    kde_z = np.zeros(len(kde_x.flatten()))
    kde_z[inside_lines] = kernel(np.vstack([kde_x.flatten()[inside_lines], kde_y.flatten()[inside_lines]]))
    kde_z = kde_z.reshape(kde_x.shape)

    # Calculate contour levels for 90% confidence intervals
    # t = np.linspace(0, np.max(kde_z), 1000)
    # integral = ((kde_z >= t[:, None, None]) * kde_z).sum(axis=(1,2))
    # integral /= np.max(integral)
    # f = interp1d(integral, t)
    # t_contours = f([0.9])
    # plt.contour(kde_x, kde_y, kde_z, levels=t_contours, colors=[cmap(0.999)], linewidths=1, linestyles='dashed', zorder=-7/4)

    # Plot density
    x_coords = kde_x[:,0]
    y_coords = kde_y[0]
    dx = (x_coords[1]-x_coords[0])/2.
    dy = (y_coords[1]-y_coords[0])/2.
    extent = [x_coords[0]-dx, x_coords[-1]+dx, y_coords[0]-dy, y_coords[-1]+dy]
    cmap = mpl.colormaps['Greens']
    plt.imshow(kde_z.T, interpolation='bicubic', cmap=cmap, extent=extent, origin='lower', aspect='auto',zorder=-8/4)

    # Plot min/max lines
    plt.fill_between(e_vals, np.min(min_line), min_line, color='w', zorder=-6/4)
    plt.fill_between(e_vals, max_line, np.max(max_line), color='w', zorder=-6/4)
    plt.plot(e_vals, max_line, c='k', zorder=1)
    plt.plot(e_vals, min_line, c='k', zorder=1)

    # Plot formatting
    plt.xlabel('e_10')
    plt.xlim(0, e_max)
    plt.ylabel('SNR fraction')
    plt.ylim(0, max_interp(e_max))

def SNR_fill_between(low, high, e_vals, interp, c, z):
    e_vals = np.arange(0, np.max(e_vals)+0.001, 0.001)
    SNR_vals = interp(e_vals)
    SNR_max_e = e_vals[SNR_vals<=high][np.argmax(SNR_vals[SNR_vals<=high])]
    SNR_fill_lower = np.max([np.full_like(e_vals[SNR_vals<=high], low), interp(e_vals[SNR_vals<=high])], axis=0)
    plt.fill_between(e_vals[SNR_vals<=high], SNR_fill_lower, np.full_like(e_vals[SNR_vals<=high], high), color=c, zorder=z)

def ecc_fill_between(low, high, e_vals, interp, c, z):
    e_vals = np.arange(np.max([low, np.min(e_vals)]), np.min([high, np.max(e_vals)-0.001])+0.001, 0.001)
    e_fill_upper = interp(e_vals)
    plt.fill_between(e_vals, 0, e_fill_upper, color=c, zorder=z)

def plot_gradient(samples, interp, fill_func, c, e_vals):

    # Plot SNR distribution
    kde = gaussian_kde(samples)
    dens_vals = np.arange(0, np.max(samples)+0.001, 0.001)
    kde_vals = kde(dens_vals)
    kde_max = np.max(kde_vals)
    cmap = mpl.colormaps[c]
    fill_func(0, 1, e_vals, interp, cmap(0.995), -100/100)
    for i in range(99):
        idx = np.argwhere(np.diff(np.sign(kde_vals - kde_max*(99-i)/100))).flatten()
        if len(idx) == 0:
            continue
        if kde_vals[idx[0]] > kde_vals[idx[0]+1]:
            fill_func(0, dens_vals[idx[0]], e_vals, interp, cmap(i/100), -i/100)
            idx = idx[1:]
            if len(idx) == 0:
                continue
        if kde_vals[idx[-1]+1] > kde_vals[idx[-1]]:
            fill_func(dens_vals[idx[-1]], 1, e_vals, interp, cmap(i/100), -i/100)
            idx = idx[:-1]
            if len(idx) == 0:
                continue
        for j in range(0, len(idx), 2):
            fill_func(dens_vals[idx[j]], dens_vals[idx[j+1]], e_vals, interp, cmap(i/100), -i/100)

def SNR_subplot(samples_SNR, prior_SNR, likeL_SNR, SNR_max, meas_SNR=None):

    # Plot histograms
    _ = plt.hist(samples_SNR, bins=50, histtype='step', density=True, range=(0, SNR_max), color='C0', lw=1.5, zorder=2, label='Overall')
    _ = plt.hist(prior_SNR, bins=50, histtype='step', density=True, range=(0, SNR_max), color='C4', ls='dashed', zorder=1, label='Prior')
    _ = plt.hist(likeL_SNR, bins=50, histtype='step', density=True, range=(0, SNR_max), color='C3', ls='dashed', zorder=1, label='Likelihood')

    # Plot measured SNR
    if meas_SNR is not None:
        plt.axvline(meas_SNR, c='darkturquoise', ls='dotted', lw=2, zorder=0, label='Measured')

    # Plot formatting
    plt.xlim(0, SNR_max)
    plt.legend(frameon=False)
    plt.xlabel('SNR fraction')
    plt.gca().axes.get_yaxis().set_visible(False)

def ecc_subplot(samples_e, prior_e, e_max, true_e=None):

    # Plot histograms
    _ = plt.hist(samples_e, bins=50, histtype='step', density=True, range=(0, e_max), color='C1', lw=1.5, zorder=2, label='Overall')
    _ = plt.hist(prior_e, bins=50, histtype='step', density=True, range=(0, e_max), color='C4', ls='dashed', zorder=1, label='Prior')

    # Plot quantiles
    quantiles = np.quantile(samples_e, [0.05, 0.5, 0.95])
    plt.axvline(quantiles[0], c='k', ls='dashed')
    plt.axvline(quantiles[1], c='k', ls='solid')
    plt.axvline(quantiles[2], c='k', ls='dashed')
    print(quantiles)

    # Plot true eccentricity
    if true_e is not None:
        plt.axvline(true_e, c='hotpink', ls='dotted', lw=2, zorder=10)

    # Plot formatting
    plt.xlim(0, e_max)
    plt.legend(frameon=False)
    plt.xlabel('e_10')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

def plot_SNR2ecc(match_grid, samples_e, samples_SNR, prior_e, prior_SNR, likeL_SNR, true_e=None, meas_SNR=None, e_max=0.4, two_ecc_harms=True):

    # Key for matches
    if two_ecc_harms:
        match_key = 'h1_h-1_h0_pc'
    else:
        match_key = 'h1_h0'

    # Useful parameters
    e_vals = match_grid['metadata']['degen_params']['ecc10']
    max_line = np.max(match_grid[match_key], axis=1)
    min_line = np.min(match_grid[match_key], axis=1)
    max_interp = interp1d(e_vals, max_line)
    min_interp = interp1d(e_vals, min_line)

    # Start main subplot
    plt.figure(figsize=(6.4*1.5,4.8*1.5))
    plt.subplot(2, 2, 3)
    main_ax = plt.gca()

    # Plot min max lines and contours
    plot_min_max_lines(samples_e, samples_SNR, match_key, e_vals, max_line, min_line, max_interp, min_interp, e_max=e_max)

    # Plot SNR and eccentricity gradients
    plot_gradient(samples_SNR, max_interp, SNR_fill_between, 'Blues_r', e_vals)
    plot_gradient(samples_e, min_interp, ecc_fill_between, 'Oranges_r', e_vals)

    # Plot true eccentricity
    if true_e is not None:
        true_e_lbl = f'True value:\ne_10 = {true_e:.3f}'
        plt.axvline(true_e, c='hotpink', ls='dotted', lw=2, zorder=10, label=true_e_lbl)
        plt.legend(frameon=False)

    # SNR subplot
    plt.subplot(2, 2, 4)
    SNR_subplot(samples_SNR, prior_SNR, likeL_SNR, max_interp(e_max), meas_SNR=meas_SNR)

    # Ecc subplot
    plt.subplot(2, 2, 1, sharex=main_ax)
    ecc_subplot(samples_e, prior_e, e_max, true_e=true_e)

    # Subplot formatting
    main_ax.xaxis.get_major_ticks()[-1].set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)