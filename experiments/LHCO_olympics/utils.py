import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_jet_features(lhco, trajectories, feature, xlim, time_step=None, d_step=None, figsize=(6,3), save_path=None):

    dic = {'p_x':(0,4), 'p_y':(1,5), 'p_z':(2,6), 'm':(3,7)}
    i, j = dic[feature]

    N =  float(trajectories.shape[0])
    step = int(N/2) if time_step is None else int(time_step)
    d_step = int(N/10) if d_step is None else int(d_step)

    _, ax = plt.subplots(1, 2, figsize=figsize)
    xmin, xmax, dx = xlim[0], xlim[1], xlim[2]
    bins = np.arange(xmin, xmax, dx)
    bool_log=False

    ax[0].hist(lhco.background[...,i], bins=bins, histtype='stepfilled', color='gold', label='SR background', log= bool_log, alpha=0.4)
    ax[0].hist(lhco.source[...,i], bins=bins, histtype='step', color='blue', label='SB1 source', log=bool_log)
    if d_step > 0: ax[0].hist(trajectories[step - d_step][...,i], bins=bins, histtype='step', color='purple', label='CFM t={}'.format((step-d_step)/N), ls=':', lw=0.5, log=bool_log)
    ax[0].hist(trajectories[step][...,i], bins=bins, histtype='step', color='purple', label='CFM t={}'.format(step/N), ls='--', lw=0.75, log=bool_log)
    if d_step > 0: ax[0].hist(trajectories[step + d_step][...,i], bins=bins, histtype='step', color='purple', label='CFM t={}'.format((step+d_step)/N), ls=':',  lw=0.5,log=bool_log)
    ax[0].hist(trajectories[-1][...,i], bins=bins, histtype='step', color='red', label='CFM t=1', ls='--', lw=0.5,log=bool_log)
    ax[0].hist(lhco.target[...,i], bins=bins, histtype='step', color='red', label='SB2 target', log=bool_log)
    ax[0].set_xlabel(r'${}$ leading jet'.format(feature))
    ax[0].legend(fontsize=5, loc='upper right')

    ax[1].hist(lhco.background[...,j], bins=bins, histtype='stepfilled', color='gold', log=bool_log, alpha=0.4, label='SR background')
    ax[1].hist(lhco.source[...,j], bins=bins, histtype='step', color='blue', log=bool_log, label='SB1 source')
    if d_step > 0: ax[1].hist(trajectories[step - d_step][...,j], bins=bins, histtype='step', color='purple', ls=':', lw=0.5, log=bool_log, label='CFM t={}'.format((step-d_step)/N))
    ax[1].hist(trajectories[step][...,j], bins=bins, histtype='step', color='purple', ls='--', lw=0.75,log=bool_log, label='CFM t={}'.format(step/N))
    if d_step > 0: ax[1].hist(trajectories[step + d_step][...,j], bins=bins, histtype='step', color='purple', ls=':', lw=0.5, log=bool_log, label='CFM t={}'.format((step+d_step)/N))
    ax[1].hist(trajectories[-1][...,j], bins=bins, histtype='step', color='red', ls='--', lw=0.5, log= bool_log, label='CFM t=1')
    ax[1].hist(lhco.target[...,j], bins=bins, histtype='step', color='red', log=bool_log, label='SB2 target')
    ax[1].set_xlabel(r'${}$ subleading jet'.format(feature))
    ax[1].legend(fontsize=5, loc='upper right')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_dijet_mass(lhco, trajectories, bins=np.arange(2700, 6000, 20), time_step=None, d_step=None, figsize=(4,4), save_path=None):

    N = float(trajectories.shape[0])
    step = int(N/2) if time_step is None else int(time_step)
    d_step = 0 if d_step is None else int(d_step)
    
    mjj_sb1 = mjj(lhco.source)
    mjj_sb2 = mjj(lhco.target)
    mjj_sb1 = mjj(lhco.source)

    mjj_sr_bckgrd = mjj(lhco.background)
    mjj_sr_early = mjj(trajectories[step - d_step])
    mjj_sr_middle = mjj(trajectories[step])
    mjj_sr_late = mjj(trajectories[step + d_step])        
    
    _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.hist(mjj_sr_bckgrd, bins=bins, histtype='stepfilled', color='gold', label='SR background', log=True, alpha=0.4,  density=True)
    ax.hist(mjj_sb1, bins=bins, histtype='step', color='blue', label='SB1 source', log=True, density=True)
    if d_step > 0: ax.hist(mjj_sr_early, bins=bins, histtype='step', color='purple', ls=':', lw = 0.5, label='CFM, t={}'.format((step-d_step)/N), log=True, density=True)
    ax.hist(mjj_sr_middle, bins=bins, histtype='step', color='purple', ls='--', lw = 0.5, label='CFM, t={}'.format(step/N), log=True, density=True)
    if d_step > 0: ax.hist(mjj_sr_late, bins=bins, histtype='step', color='purple', ls=':', lw = 0.5, label='CFM, t={}'.format((step+d_step)/N), log=True, density=True)
    ax.hist(mjj(trajectories[-1]), bins=bins, histtype='step', color='red', ls='--', lw = 0.5, label='CFM, t=1.0', log=True, density=True)
    ax.hist(mjj_sb2, bins=bins, histtype='step', color='red', label='SB2 target', log=True, density=True)
    ax.set_xlabel(r'$m_{jj}$')
    ax.legend(fontsize=5, loc='upper right')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def mjj(data, window=[-np.inf, np.inf]):
    ''' data = [pt_1, eta_1, phi_1, m_1, pt_2, eta_2, phi_2, m_2] '''
    pt_1, pt_2 = data[..., 0], data[..., 4]
    eta_1, eta_2 = data[..., 1], data[..., 5]
    phi_1, phi_2 = data[..., 2], data[..., 6]
    m_1, m_2 = data[..., 3], data[..., 7]

    px_1 = pt_1 * torch.cos(phi_1)
    py_1 = pt_1 * torch.sin(phi_1)
    pz_1 = torch.sqrt(pt_1**2 + m_1**2) * torch.sinh(eta_1)
    e_1 = torch.sqrt(px_1**2 + py_1**2 + pz_1**2 + m_1**2)

    px_2 = pt_2 * torch.cos(phi_2)
    py_2 = pt_2 * torch.sin(phi_2)
    pz_2 = torch.sqrt(pt_2**2 + m_2**2) * torch.sinh(eta_2)
    e_2 = torch.sqrt(px_2**2 + py_2**2 + pz_2**2 + m_2**2)

    mjj = torch.sqrt((e_1 + e_2)**2 - (px_1 + px_2)**2 - (py_1 + py_2)**2 - (pz_1 + pz_2)**2)
    mjj = mjj[(mjj > window[0]) & (mjj < window[1])]
    return mjj


def plot_interpolation(lhco, pipeline,  mass_window, figsize=(14,6), 
                       features=[r'$m_{jj}$', r'$m_{j}$ leading', r'$\Delta m_j$', r'$\tau_{21}$ leading',  r'$\tau_{21}$ sub-leading'],
                       bins=[(2500, 5000, 30), (0, 1400, 50), (-1250, 1250, 100), (-0.25, 1.25, 0.025), (-0.25, 1.25, 0.025)], 
                       log=True, density=True, save_path=None):
    
    x = torch.mean(pipeline.trajectories[...,0], dim=-1) - lhco.background[...,0].mean()
    idx = torch.argmin(torch.abs(x))
    interpolation = pipeline.trajectories[idx]  
    mask = (interpolation[...,0] > mass_window[0]) & (interpolation[...,0] < mass_window[1])
    fig, axs = plt.subplots(2, 5, figsize=figsize, gridspec_kw={'height_ratios': [5, 1]})

    for f in range(len(features)):
        b = np.arange(*bins[f])

        # Top row: Plotting the histograms
        ax = axs[0, f]
        ax.hist(lhco.source[...,f], bins=b, histtype='step', color='darkred', label='SB1 (source)', log=log, density=density)
        ax.hist(interpolation[mask][...,f], bins=b, histtype='step', color='k', label='t=0.5', log=log, density=density)
        ax.hist(interpolation[...,f], bins=b, histtype='step', color='k', ls=':',  log=log, density=density)
        ax.hist(lhco.target[...,f], bins=b, histtype='step', color='darkblue',  label='SB2 (target)', log=log, density=density)
        ax.hist(lhco.background[...,f],bins=b, histtype='stepfilled', color='gray', alpha=0.3, label='SR', log=log, density=density)
        ax.set_xticklabels([])  # Hide x-axis labels for the top row
        if f==0: ax.legend(loc='upper right', fontsize=8)

        # Second row: Plotting the ratio of histograms
        counts_sb1, _ = np.histogram(lhco.source[...,f], bins=b, density=density)
        counts_sb2, _ = np.histogram(lhco.target[...,f], bins=b, density=density)
        counts_interpolation, _ = np.histogram(interpolation[mask][...,f], bins=b, density=density)
        counts_background, _ = np.histogram(lhco.background[...,f], bins=b, density=density)


        ratio_sb1 = np.divide(counts_sb1, counts_background, out=np.zeros_like(counts_sb1), where=counts_background!=0)
        ratio_sb2 = np.divide(counts_sb2, counts_background, out=np.zeros_like(counts_sb2), where=counts_background!=0)
        ratio_interpolation = np.divide(counts_interpolation, counts_background, out=np.zeros_like(counts_interpolation), where=counts_background!=0)

        ax_ratio = axs[1, f]

        ax_ratio.plot(b[:-1], ratio_sb1, drawstyle='steps-post', color='darkred')
        ax_ratio.plot(b[:-1], ratio_sb2, drawstyle='steps-post', color='darkblue')
        ax_ratio.plot(b[:-1], ratio_interpolation, drawstyle='steps-post', color='k')

        ax_ratio.set_ylim(0.8, 1.2, 0) 
        ax_ratio.set_xlabel(features[f])
        ax_ratio.axhline(1, color='gray', linestyle='-', lw=0.75)
        ax_ratio.axhline(0.9, color='gray', linestyle='--', lw=0.75)
        ax_ratio.axhline(1.1, color='gray', linestyle='--', lw=0.75)

        for tick in ax_ratio.xaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        for tick in ax_ratio.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)  
        if f == 0:
            ax_ratio.set_ylabel('ratio', fontsize=8)
        ax_ratio.set_yticks([0.8,  0.9,  1,  1.1,  1.2])
    
    if save_path is not None: plt.savefig(save_path)
    plt.show()
    plt.close()



def plot_interpolation_combined(lhco, pipeline_forward, pipeline_backward, mass_window, figsize=(14,6), 
                                features=[r'$m_{jj}$', r'$m_{j}$ leading', r'$\Delta m_j$', r'$\tau_{21}$ leading',  r'$\tau_{21}$ sub-leading'],
                                bins=[(2500, 5000, 30), (0, 1400, 50), (-1250, 1250, 100), (-0.25, 1.25, 0.025), (-0.25, 1.25, 0.025)], 
                                log=True, density=True):
    
    x_forward = torch.mean(pipeline_forward.trajectories[...,0], dim=-1) - lhco.background[...,0].mean()
    idx_forward = torch.argmin(torch.abs(x_forward))
    interpolation_forward = pipeline_forward.trajectories[idx_forward]  

    x_backward = torch.mean(pipeline_backward.trajectories[...,0], dim=-1) - lhco.background[...,0].mean()
    idx_backward = torch.argmin(torch.abs(x_backward))
    interpolation_backward = pipeline_backward.trajectories[idx_backward] 

    interpolation = torch.cat([interpolation_forward, interpolation_backward], dim=0)

    fig, axs = plt.subplots(2, 5, figsize=figsize, gridspec_kw={'height_ratios': [5, 1]})

    for f in range(len(features)):
        b = np.arange(*bins[f])

        # Top row: Plotting the histograms
        ax = axs[0, f]
        ax.hist(lhco.source[...,f], bins=b, histtype='step', color='darkred', label='SB1 (source)', log=log, density=density)
        ax.hist(interpolation[...,f], bins=b, histtype='step', color='k', label='t=0.5', log=log, density=density)
        ax.hist(lhco.target[...,f], bins=b, histtype='step', color='darkblue',  label='SB2 (target)', log=log, density=density)
        ax.hist(lhco.background[...,f],bins=b, histtype='stepfilled', color='gray', alpha=0.3, label='SR', log=log, density=density)
        ax.set_xticklabels([])  # Hide x-axis labels for the top row
        if f==0: ax.legend(loc='upper right', fontsize=8)

        # Second row: Plotting the ratio of histograms
        counts_sb1, _ = np.histogram(lhco.source[...,f], bins=b, density=density)
        counts_sb2, _ = np.histogram(lhco.target[...,f], bins=b, density=density)
        counts_interpolation, _ = np.histogram(interpolation[...,f], bins=b, density=density)
        counts_background, _ = np.histogram(lhco.background[...,f], bins=b, density=density)

        ratio_sb1 = np.divide(counts_sb1, counts_background, out=np.zeros_like(counts_sb1), where=counts_background!=0)
        ratio_sb2 = np.divide(counts_sb2, counts_background, out=np.zeros_like(counts_sb2), where=counts_background!=0)
        ratio_interpolation = np.divide(counts_interpolation, counts_background, out=np.zeros_like(counts_interpolation), where=counts_background!=0)

        ax_ratio = axs[1, f]
        ax_ratio.plot(b[:-1], ratio_sb1, drawstyle='steps-post', color='darkred')
        ax_ratio.plot(b[:-1], ratio_sb2, drawstyle='steps-post', color='darkblue')
        ax_ratio.plot(b[:-1], ratio_interpolation, drawstyle='steps-post', color='k')
        ax_ratio.set_ylim(0.8, 1.2, 0) 
        ax_ratio.set_xlabel(features[f])
        ax_ratio.axhline(1, color='gray', linestyle='-', lw=0.75)
        ax_ratio.axhline(0.9, color='gray', linestyle='--', lw=0.75)
        ax_ratio.axhline(1.1, color='gray', linestyle='--', lw=0.75)

        for tick in ax_ratio.xaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        for tick in ax_ratio.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)  
        if f == 0:
            ax_ratio.set_ylabel('ratio', fontsize=8)
        ax_ratio.set_yticks([0.8,  0.9,  1,  1.1,  1.2])

    plt.tight_layout()
    plt.show()