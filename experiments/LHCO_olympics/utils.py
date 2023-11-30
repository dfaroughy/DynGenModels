import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import matplotlib.ticker as ticker


def load_bridge_pipelines(bridge, 
                          model_fwd,
                          model_bwd=None, 
                          num_samples=100000000,
                          device='cpu',
                          model='best'):

    #...load forward bridge:

    from DynGenModels.configs.lhco_configs import LHCOlympics_HighLevel_MLP_CondFlowMatch as Configs
    from DynGenModels.datamodules.lhco.datasets import LHCOlympicsHighLevelDataset
    from DynGenModels.datamodules.lhco.dataloader import LHCOlympicsDataLoader 
    from DynGenModels.models.deep_nets import MLP
    from DynGenModels.trainer.trainer import DynGenModelTrainer

    configs_fwd = Configs().load(model_fwd + '/config.json')
    configs_fwd.DEVICE = device  
    configs_fwd.workdir = model_fwd 
    lhco_fwd = LHCOlympicsHighLevelDataset(configs_fwd)
    cfm_fwd  = DynGenModelTrainer(dynamics = bridge(configs_fwd),
                                  model = MLP(configs_fwd), 
                                  dataloader = LHCOlympicsDataLoader(lhco_fwd , configs_fwd), 
                                  configs = configs_fwd)
    cfm_fwd.load(model=model)

    #...load backward bridge:
    if model_bwd is not None:

        from DynGenModels.configs.lhco_configs import LHCOlympics_HighLevel_MLP_CondFlowMatch as Configs
        from DynGenModels.datamodules.lhco.datasets import LHCOlympicsHighLevelDataset
        from DynGenModels.datamodules.lhco.dataloader import LHCOlympicsDataLoader 
        from DynGenModels.models.deep_nets import MLP
        from DynGenModels.trainer.trainer import DynGenModelTrainer

        configs_bwd = Configs().load(model_bwd + '/config.json')
        configs_bwd.DEVICE = device  
        configs_bwd.workdir = model_bwd
        lhco_bwd = LHCOlympicsHighLevelDataset(configs_bwd, exchange_target_with_source=True)
        cfm_bwd = DynGenModelTrainer(dynamics = bridge(configs_bwd),
                                    model = MLP(configs_bwd), 
                                    dataloader = LHCOlympicsDataLoader(lhco_bwd, configs_bwd), 
                                    configs = configs_bwd)
        cfm_bwd.load(model=model)
    else:
        configs_bwd = None
        lhco_bwd = None
        cfm_bwd = None

    #...get pipelines:

    from DynGenModels.pipelines.SamplingPipeline import FlowMatchPipeline 
    from DynGenModels.datamodules.lhco.dataprocess import PreProcessLHCOlympicsHighLevelData, PostProcessLHCOlympicsHighLevelData

    pipeline_fwd = FlowMatchPipeline(trained_model=cfm_fwd, 
                                     configs=configs_fwd, 
                                     preprocessor=PreProcessLHCOlympicsHighLevelData,
                                     postprocessor=PostProcessLHCOlympicsHighLevelData,
                                     best_epoch_model=True)
    if model_bwd is not None:
        pipeline_bwd = FlowMatchPipeline(trained_model=cfm_bwd, 
                                         configs=configs_bwd, 
                                         preprocessor=PreProcessLHCOlympicsHighLevelData,
                                         postprocessor=PostProcessLHCOlympicsHighLevelData,
                                         best_epoch_model=True)


    #...generate samples:

    pipeline_fwd.generate_samples(input_source=lhco_fwd.source[:num_samples])
    if model_bwd is not None: pipeline_bwd.generate_samples(input_source=lhco_bwd.source[:num_samples])

    return pipeline_fwd, pipeline_bwd, lhco_fwd, lhco_bwd


def plot_interpolation(lhco, pipeline,  mass_window, figsize=(14,6), 
                       features=[r'$m_{jj}$', r'$m_{j}$ leading', r'$\Delta m_j$', r'$\tau_{21}$ leading',  r'$\tau_{21}$ sub-leading'],
                       bins=[(2500, 5000, 30), (0, 1400, 50), (-1250, 1250, 100), (-0.25, 1.25, 0.025), (-0.25, 1.25, 0.025)], 
                       log=True, density=True, save_path=None):
    
    x = torch.mean(pipeline.trajectories[...,0], dim=-1) - lhco.background[...,0].mean()
    idx = torch.argmin(torch.abs(x))
    interpolation = pipeline.trajectories[idx]  
    mask = (interpolation[...,0] > mass_window[0]) & (interpolation[...,0] < mass_window[1])
    mask_back = (lhco.background[...,0] > mass_window[0]) & (lhco.background[...,0] < mass_window[1])

    fig, axs = plt.subplots(2, 5, figsize=figsize, gridspec_kw={'height_ratios': [5, 1]})

    for f in range(len(features)):
        b = np.arange(*bins[f])

        # Top row: Plotting the histograms
        ax = axs[0, f]
        ax.hist(lhco.source[...,f], bins=b, histtype='step', color='darkred', label='SB1 (source)', log=log, density=density)
        ax.hist(interpolation[mask][...,f], bins=b, histtype='step', color='k', label='t=0.5', log=log, density=density)
        ax.hist(interpolation[...,f], bins=b, histtype='step', color='k', ls=':',  log=log, density=density)
        ax.hist(lhco.target[...,f], bins=b, histtype='step', color='darkblue',  label='SB2 (target)', log=log, density=density)
        ax.hist(lhco.background[mask_back][...,f],bins=b, histtype='stepfilled', color='gray', alpha=0.3, label='SR', log=log, density=density)
        ax.set_xticklabels([])  # Hide x-axis labels for the top row
        if f==0: ax.legend(loc='upper right', fontsize=8)

        # Second row: Plotting the ratio of histograms
        counts_sb1, _ = np.histogram(lhco.source[...,f], bins=b, density=density)
        counts_sb2, _ = np.histogram(lhco.target[...,f], bins=b, density=density)
        counts_interpolation, _ = np.histogram(interpolation[mask][...,f], bins=b, density=density)
        counts_background, _ = np.histogram(lhco.background[mask_back][...,f], bins=b, density=density)

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
    else: plt.show()
    plt.close()



def plot_interpolation_low_level(lhco, pipeline,  mass_window,  time_stop_feature='mjj', figsize=(12,6), 
                                features=['mjj', 'px_j1', 'py_j1', 'pz_j1', 'e_j1'],
                                bins=[(2700, 4200, 40), (-2000, 2000, 100), (-2000, 2000, 100), (-5000, 5000, 200), (600, 4000, 100)], 
                                log=True, density=True, save_path=None, show=False):    
    dic = {'mjj':0, 'delta_Rjj':1, 'delta_mjj':2, 'delta_ptjj':3, 'delta_etajj':4,
           'pt_j1':5, 'eta_j1':6, 'phi_j1':7, 'm_j1':8, 'pt_j2':9, 'eta_j2':10, 'phi_j2':11, 'm_j2':12,
           'px_j1':13, 'py_j1':14, 'pz_j1':15, 'e_j1':16, 'px_j2':17, 'py_j2':18, 'pz_j2':19, 'e_j2':20}

    N = pipeline.num_sampling_steps

    def get_features(features):
        px_j1, py_j1, pz_j1, e_j1 = features[...,0], features[...,1], features[...,2], features[...,3]
        px_j2, py_j2, pz_j2, e_j2 = features[...,4], features[...,5], features[...,6], features[...,7]
        m_j1 = torch.sqrt(e_j1**2 - px_j1**2 - py_j1**2 - pz_j1**2)
        m_j2 = torch.sqrt(e_j2**2 - px_j2**2 - py_j2**2 - pz_j2**2)
        pt_j1,  pt_j2 = torch.sqrt(px_j1**2 + py_j1**2), torch.sqrt(px_j2**2 + py_j2**2)  
        eta_j1, eta_j2 = 0.5 * np.log( (e_j1 + pz_j1) / (e_j1 - pz_j1)), 0.5 * np.log((e_j2 + pz_j2) / (e_j2 - pz_j2))
        phi_j1, phi_j2 = np.arctan2(py_j1, px_j1), np.arctan2(py_j2, px_j2)    
        delta_Rjj = np.sqrt((phi_j1 - phi_j2)**2 + (eta_j1 - eta_j2)**2)   
        mjj = torch.sqrt((e_j1 + e_j2)**2 - (px_j1 + px_j2)**2 - (py_j1 + py_j2)**2 - (pz_j1 + pz_j2)**2) 
        delta_mjj = torch.abs(m_j2 - m_j1)
        delta_ptjj = torch.abs(pt_j2 - pt_j1)
        delta_etajj = torch.abs(eta_j2 - eta_j1)

        all_feats = torch.concat([mjj[:, None], delta_Rjj[:, None], delta_mjj[:, None], delta_ptjj[:, None], delta_etajj[:, None],
                                  pt_j1[:, None], eta_j1[:, None], phi_j1[:, None], m_j1[:, None],
                                  pt_j2[:, None], eta_j2[:, None], phi_j2[:, None], m_j2[:, None],
                                  px_j1[:, None], py_j1[:, None], pz_j1[:, None], e_j1[:, None],
                                  px_j2[:, None], py_j2[:, None], pz_j2[:, None], e_j2[:, None]], dim=-1)
        return all_feats

    source, target, background = get_features(lhco.source), get_features(lhco.target), get_features(lhco.background)
    trajectories = []
    for trajectory in pipeline.trajectories: trajectories.append(get_features(trajectory))
    trajectories = torch.stack(trajectories, dim=0)

    assert time_stop_feature in dic, f'Feature {time_stop_feature} not in {dic.keys()}'
    x = torch.mean(trajectories[...,dic[time_stop_feature]], dim=-1) - background[...,dic[time_stop_feature]].mean()
    idx = torch.argmin(torch.abs(x))
    interpolation = trajectories[idx]  
    mask = (interpolation[...,0] > mass_window[0]) & (interpolation[...,0] < mass_window[1])
    mask_back = (background[...,0] > mass_window[0]) & (background[...,0] < mass_window[1])

    fig, axs = plt.subplots(2, len(features), figsize=figsize, gridspec_kw={'height_ratios': [len(features), 1]})
    
    for i,f in enumerate([dic[f] for f in features if f in dic]):
        b = np.arange(*bins[i])
        
        # First row: Plotting the ratio of histograms
        ax = axs[0, i]
        ax.hist(source[...,f], bins=b, histtype='step', color='darkred', label='SB1 (source)', log=log, density=density)
        # ax.hist(trajectories[N//4][...,f], bins=b, histtype='step', color='darkred', ls=':', lw=0.75, label='t=0.25', log=log, density=density)
        ax.hist(interpolation[mask][...,f], bins=b, histtype='step', color='k', label='interpolation (t=0.5)', log=log, density=density)
        # ax.hist(trajectories[3*N//4][...,f], bins=b, histtype='step', color='purple', ls=':', lw=0.75, label='t=0.75', log=log, density=density)
        # ax.hist(trajectories[-1][...,f], bins=b, histtype='step', color='darkblue', ls=':', lw=0.75, label='t=1',log=log, density=density)
        ax.hist(target[...,f], bins=b, histtype='step', color='darkblue',  label='SB2 (target)', log=log, density=density)
        ax.hist(background[mask_back][...,f],bins=b, histtype='stepfilled', color='gray', alpha=0.3, label='SR', log=log, density=density)
        ax.set_xticklabels([])   
        if f == 0: ax.legend(loc='lower left', fontsize=6)

        # Second row: Plotting the ratio of histograms
        counts_interpolation, _ = np.histogram(interpolation[mask][...,f], bins=b, density=density)
        counts_background, _ = np.histogram(background[mask_back][...,f], bins=b, density=density)
        ratio_interpolation = np.divide(counts_interpolation, counts_background, out=np.zeros_like(counts_interpolation), where=counts_background!=0)

        ax_ratio = axs[1, i]
        ax_ratio.plot(b[:-1], ratio_interpolation, drawstyle='steps-post', color='k')
        ax_ratio.set_ylim(0.8, 1.2, 0) 
        ax_ratio.set_xlabel(features[i])
        ax_ratio.axhline(1, color='gray', linestyle='-', lw=0.75)
        ax_ratio.axhline(0.9, color='gray', linestyle='--', lw=0.75)
        ax_ratio.axhline(1.1, color='gray', linestyle='--', lw=0.75)

        for tick in ax_ratio.xaxis.get_major_ticks(): tick.label.set_fontsize(9)
        for tick in ax_ratio.yaxis.get_major_ticks(): tick.label.set_fontsize(9)  
        ax_ratio.set_yticks([0.8,  0.9,  1,  1.1,  1.2])
    
    if save_path is not None: plt.savefig(save_path)
    if show: plt.show()
    plt.close()




def plot_interpolation_combined(lhco, pipeline_fwd, pipeline_bwd, mass_window, figsize=(14,6), 
                                features=[r'$m_{jj}$', r'$m_{j}$ leading', r'$\Delta m_j$', r'$\tau_{21}$ leading',  r'$\tau_{21}$ sub-leading'],
                                bins=[(12000, 8000, 30), (0, 1400, 50), (-1250, 1250, 100), (-0.25, 1.25, 0.025), (-0.25, 1.25, 0.025)], 
                                log=True, density=True, save_path_fwd=None, save_path_bwd=None):
    
    #...forward interpolation:
    x_fwd = torch.mean(pipeline_fwd.trajectories[...,0], dim=-1) - lhco.background[...,0].mean()
    idx_fwd = torch.argmin(torch.abs(x_fwd))
    interpolation_fwd = pipeline_fwd.trajectories[idx_fwd]  

    #...backward interpolation:
    x_bwd = torch.mean(pipeline_bwd.trajectories[...,0], dim=-1) - lhco.background[...,0].mean()
    idx_bwd = torch.argmin(torch.abs(x_bwd))
    interpolation_bwd = pipeline_bwd.trajectories[idx_bwd] 

    #...combined interpolation:
    interpolation = torch.cat([interpolation_fwd, interpolation_bwd], dim=0)
    mask = (interpolation[...,0] > mass_window[0]) & (interpolation[...,0] < mass_window[1])
    mask_bask = (lhco.background[...,0] > mass_window[0]) & (lhco.background[...,0] < mass_window[1])

    #...plotting:

    fig, axs = plt.subplots(2, 5, figsize=figsize, gridspec_kw={'height_ratios': [5, 1]})

    for f in range(len(features)):
        b = np.arange(*bins[f])

        # Top row: Plotting the histograms
        ax = axs[0, f]
        if f==0: 
            ax.hist(lhco.source[...,f], bins=b, histtype='stepfilled', color='darkred', alpha=0.3, label=r'SB$_1$', log=log, density=density)
            ax.hist(pipeline_bwd.trajectories[-1][...,f], bins=b, histtype='step', color='darkblue',  label=r'$t=1$ (SB$_2\to$ SB$_1$)', log=log, density=density)
            ax.hist(interpolation[mask][...,f], bins=b, histtype='step', color='k',label=r'$t=0.5$' , log=log, density=density)
            ax.hist(interpolation[...,f], bins=b, histtype='step', color='k', ls=':', log=log, density=density)
            ax.hist(pipeline_fwd.trajectories[-1][...,f], bins=b, histtype='step', color='darkred', label=r'$t=1$ (SB$_1\to$ SB$_2$)', log=log, density=density)
            ax.hist(lhco.target[...,f], bins=b, histtype='stepfilled', color='darkblue', alpha=0.3, label=r'SB$_2$', log=log, density=density)
        if f!=0: ax.hist(interpolation[mask][...,f], bins=b, histtype='step', color='k', log=log, density=density)
        ax.hist(lhco.background[mask_bask][...,f],bins=b, histtype='stepfilled', color='gray', alpha=0.3, label='SR', log=log, density=density)
        ax.set_xticklabels([]) 
        if f==0: 
            if log: ax.set_ylim(1e-6, 1e-1)
            ax.legend(loc='upper right', fontsize=6, labelspacing=0.1)

        # Second row: Plotting the ratio of histograms
        counts_sb1, _ = np.histogram(lhco.source[...,f], bins=b, density=density)
        counts_sb2, _ = np.histogram(lhco.target[...,f], bins=b, density=density)
        counts_fwd_target, _ = np.histogram(pipeline_fwd.trajectories[-1][...,f], bins=b, density=density)
        counts_bwd_target, _ = np.histogram(pipeline_bwd.trajectories[-1][...,f], bins=b, density=density)
        counts_interpolation, _ = np.histogram(interpolation[mask][...,f], bins=b, density=density)
        counts_background, _ = np.histogram(lhco.background[mask_bask][...,f], bins=b, density=density)
        
        # Ratio plots:
        ratio_sb1 = np.divide(counts_bwd_target, counts_sb1, out=np.zeros_like(counts_interpolation), where=counts_sb1!=0)
        ratio_sb2 = np.divide(counts_fwd_target, counts_sb2, out=np.zeros_like(counts_interpolation), where=counts_sb2!=0)
        ratio_interpolation = np.divide(counts_interpolation, counts_background, out=np.zeros_like(counts_interpolation), where=counts_background!=0)

        ax_ratio = axs[1, f]
        ax_ratio.plot(b[:-1], ratio_interpolation, drawstyle='steps-post', color='k')
        if f==0: 
            ax_ratio.plot(b[:-1], ratio_sb1, drawstyle='steps-post', color='darkblue')
            ax_ratio.plot(b[:-1], ratio_sb2, drawstyle='steps-post', color='darkred')
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
    if save_path_fwd is not None: plt.savefig(save_path_fwd)
    if save_path_bwd is not None: plt.savefig(save_path_bwd)
    plt.show()


def contour_plot(ax, data1, data2, bins, filled=True, lw=1, ls='-', color='k'):
    H, xedges, yedges = np.histogram2d(data1, data2, bins=bins, density=True)
    H = gaussian_filter(H, sigma=1.5)  # Smoothing
    H /= H.sum()
    sorted_hist = np.sort(H.flatten())
    cumulative = np.cumsum(sorted_hist)
    lesser_sigma = sorted_hist[np.searchsorted(cumulative, 0.2)]
    less_sigma = sorted_hist[np.searchsorted(cumulative, 0.35)]
    one_sigma = sorted_hist[np.searchsorted(cumulative, 0.68)]
    two_sigma = sorted_hist[np.searchsorted(cumulative, 0.95)]
    levels = [lesser_sigma, less_sigma, one_sigma, two_sigma]
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters)
    if H.shape != X.shape: H = H.T
    if filled: ax.contourf(X, Y, H, levels=levels, cmap=color, alpha=0.8)
    else:  ax.contour(X, Y, H, levels=levels, colors=color, linewidths=lw)


def plot_interpolation_combined_corner(lhco, 
                                       pipeline_fwd, 
                                       pipeline_bwd, 
                                       mass_window, 
                                       figsize=(14,14), 
                                       features=[r'$m_{jj}$', r'$m_{j}$ leading', r'$\Delta m_j$', r'$\tau_{21}$ leading',  r'$\tau_{21}$ sub-leading'],
                                       bins=[(12000, 8000, 30), (0, 1400, 50), (-1250, 1250, 100), (-0.25, 1.25, 0.025), (-0.25, 1.25, 0.025)], 
                                       color='k',
                                       ls='-',
                                       lw=1,
                                       log=True, 
                                       density=True, 
                                       save_path_fwd=None, 
                                       save_path_bwd=None, 
                                       filled_contours=False):

    #...forward interpolation:
    x_fwd = torch.mean(pipeline_fwd.trajectories[...,0], dim=-1) - lhco.background[...,0].mean()
    idx_fwd = torch.argmin(torch.abs(x_fwd))
    interpolation_fwd = pipeline_fwd.trajectories[idx_fwd]  

    #...backward interpolation:
    x_bwd = torch.mean(pipeline_bwd.trajectories[...,0], dim=-1) - lhco.background[...,0].mean()
    idx_bwd = torch.argmin(torch.abs(x_bwd))
    interpolation_bwd = pipeline_bwd.trajectories[idx_bwd]

    print(r'$t_f=${}, $t_b={}$'.format(idx_fwd/1000, idx_bwd/1000))

    #...combined interpolation:
    interpolation = torch.cat([interpolation_fwd, interpolation_bwd], dim=0)
    mask = (interpolation[...,0] > mass_window[0]) & (interpolation[...,0] < mass_window[1])
    mask_bask = (lhco.background[...,0] > mass_window[0]) & (lhco.background[...,0] < mass_window[1])

    # ...plotting:
    fig = plt.figure(figsize=figsize)
    num_features = len(features)
    gs = gridspec.GridSpec(num_features + 1, num_features, height_ratios=[1]*num_features + [0.5])  # Adjust the last number for aspect ratio

    for j in range(num_features):
        for i in range(j, num_features):
            ax = plt.subplot(gs[i, j])
            bins_i = np.arange(*bins[i])
            bins_j = np.arange(*bins[j])

            # Diagonal: 1D histograms
            if i == j:
                ax.hist(interpolation[mask][...,i], bins=bins_i, histtype='step', color=color, lw=lw, ls=ls, log=log, density=density)
                ax.hist(lhco.background[mask_bask][...,i], bins=bins_i, histtype='stepfilled', alpha=0.4, color='slategray', log=log, density=density)
                ax.set_ylabel("Density" if i == 0 else "")
                if i==0: ax.set_ylim(0, 0.005)  
                ax.set_xlim(bins_i[0], bins_i[-1])
                ax.set_title(features[i], fontsize=10)

            # Off-diagonal: 2D contour/histogram plots
            if i > j:
                data_i = interpolation[mask][...,i].cpu().numpy()
                data_j = interpolation[mask][...,j].cpu().numpy()
                contour_plot(ax, data_j, data_i, bins=(bins_j, bins_i), color=color, lw=lw, ls=ls, filled=filled_contours)
                data_i = lhco.background[mask_bask][...,i].cpu().numpy()
                data_j = lhco.background[mask_bask][...,j].cpu().numpy()
                contour_plot(ax, data_j, data_i, bins=(bins_j, bins_i), color='bone_r', filled=~filled_contours)
                if j==0: ax.set_ylabel(features[i])
                ax.set_xlim(bins_j[0], bins_j[-1])
                ax.set_ylim(bins_i[0], bins_i[-1])
                ax.tick_params(direction='in', which='both', length=6)

            ax.set_xticklabels([])
            
            if j > 0: ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            else: 
                ax.axvline(mass_window[0], color='darkred', linestyle='--', lw=0.75)
                ax.axvline(mass_window[1], color='darkred', linestyle='--', lw=0.75)
                ax.tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False, labelbottom=False, labelleft=True)
            ax.grid(True, linestyle=':', color='gray', linewidth=0.5)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        
    # Ratio plots:
    for f in range(len(features)):
        b = np.arange(*bins[f])
        ax_ratio = plt.subplot(gs[num_features, f])
        counts_interpolation, _ = np.histogram(interpolation[mask][...,f], bins=b, density=density)
        counts_background, _ = np.histogram(lhco.background[mask_bask][...,f], bins=b, density=density)
        
        # Ratio plots:
        ratio_interpolation = np.divide(counts_interpolation, counts_background, out=np.zeros_like(counts_interpolation), where=counts_background!=0)
        ax_ratio.plot(b[:-1], ratio_interpolation, drawstyle='steps-post', color=color)
        ax_ratio.set_ylim(0.8, 1.2, 0) 
        ax_ratio.set_xlabel(features[f])
        ax_ratio.axhline(1, color='darkred', linestyle='-', lw=0.75)
        ax_ratio.set_xlim(b[0], b[-1])
        
        for tick in ax_ratio.xaxis.get_major_ticks(): tick.label.set_fontsize(9)
        for tick in ax_ratio.yaxis.get_major_ticks(): tick.label.set_fontsize(9)  
        if f == 0: 
            ax_ratio.axvline(mass_window[0], color='darkred', linestyle='--', lw=0.75)
            ax_ratio.axvline(mass_window[1], color='darkred', linestyle='--', lw=0.75)
            ax_ratio.set_ylabel('ratio', fontsize=8)
            ax_ratio.set_yticks([0.8,  0.9,  1,  1.1,  1.2])
        else: 
            ax_ratio.set_yticks([0.8,  0.9,  1,  1.1,  1.2])
            ax_ratio.tick_params(axis='both', which='both', bottom=True, top=False, left=False, right=False, labelbottom=True, labelleft=False) 
        ax_ratio.grid(True, linestyle=':', color='gray', linewidth=0.5)

    plt.subplots_adjust(wspace=0.1, hspace=0.15)  # Adjust these values as needed
    if save_path_fwd is not None: plt.savefig(save_path_fwd)
    if save_path_bwd is not None: plt.savefig(save_path_bwd)
    plt.show()




def plot_marginal_fits_corner(lhco, 
                            pipeline_fwd, 
                            pipeline_bwd, 
                            mass_window, 
                            figsize=(14,14), 
                            features=[r'$m_{jj}$', r'$m_{j}$ leading', r'$\Delta m_j$', r'$\tau_{21}$ leading',  r'$\tau_{21}$ sub-leading'],
                            bins=[(12000, 8000, 30), (0, 1400, 50), (-1250, 1250, 100), (-0.25, 1.25, 0.025), (-0.25, 1.25, 0.025)], 
                            ls='-',
                            lw=1,
                            log=True, 
                            density=True, 
                            save_path_fwd=None, 
                            save_path_bwd=None, 
                            filled_contours=False):

    #...forward interpolation:
    x_fwd = torch.mean(pipeline_fwd.trajectories[...,0], dim=-1) - lhco.background[...,0].mean()
    idx_fwd = torch.argmin(torch.abs(x_fwd))
    interpolation_fwd = pipeline_fwd.trajectories[idx_fwd]  

    #...backward interpolation:
    x_bwd = torch.mean(pipeline_bwd.trajectories[...,0], dim=-1) - lhco.background[...,0].mean()
    idx_bwd = torch.argmin(torch.abs(x_bwd))
    interpolation_bwd = pipeline_bwd.trajectories[idx_bwd]

    print(r'$t_f=${}, $t_b={}$'.format(idx_fwd/1000, idx_bwd/1000))

    #...combined interpolation:
    interpolation = torch.cat([interpolation_fwd, interpolation_bwd], dim=0)
    mask = (interpolation[...,0] > mass_window[0]) & (interpolation[...,0] < mass_window[1])
    mask_bask = (lhco.background[...,0] > mass_window[0]) & (lhco.background[...,0] < mass_window[1])

    # ...plotting:
    fig = plt.figure(figsize=figsize)
    num_features = len(features)
    gs = gridspec.GridSpec(num_features + 1, num_features, height_ratios=[1]*num_features + [0.5])  # Adjust the last number for aspect ratio

    for j in range(num_features):
        for i in range(num_features):
            ax = plt.subplot(gs[i, j])
            bins_i = np.arange(*bins[i])
            bins_j = np.arange(*bins[j])

            # Diagonal: 1D histograms
            if i == j:
                ax.hist(pipeline_fwd.target[...,i], bins=bins_i, histtype='step', color='darkblue', lw=lw, ls=ls, log=log, density=density)
                ax.hist(lhco.target[...,i], bins=bins_i, histtype='stepfilled', alpha=0.2, color='darkblue', log=log, density=density)
                ax.hist(pipeline_bwd.target[...,i], bins=bins_i, histtype='step', color='darkred', lw=lw, ls=ls, log=log, density=density)
                ax.hist(lhco.source[...,i], bins=bins_i, histtype='stepfilled', alpha=0.2, color='darkred', log=log, density=density)
                ax.set_ylabel("Density" if i == 0 else "")
                ax.set_xlim(bins_i[0], bins_i[-1])

            # Upper Off-diagonal: 2D contour/histogram plots
            if i > j:
                data_i = pipeline_fwd.target[...,i].cpu().numpy()
                data_j = pipeline_fwd.target[...,j].cpu().numpy()
                contour_plot(ax, data_j, data_i, bins=(bins_j, bins_i), color='darkblue', lw=lw, ls=ls, filled=filled_contours)
                data_i = lhco.target[...,i].cpu().numpy()
                data_j = lhco.target[...,j].cpu().numpy()
                contour_plot(ax, data_j, data_i, bins=(bins_j, bins_i), color='Blues', filled=~filled_contours)
                if j==0: ax.set_ylabel(features[i])
                ax.set_xlim(bins_j[0], bins_j[-1])
                ax.set_ylim(bins_i[0], bins_i[-1])
                ax.tick_params(direction='in', which='both', length=6)

            # Lower Off-diagonal: 2D contour/histogram plots
            if i < j:
                data_i = pipeline_bwd.target[...,i].cpu().numpy()
                data_j = pipeline_bwd.target[...,j].cpu().numpy()
                contour_plot(ax, data_j, data_i, bins=(bins_j, bins_i), color='darkred', lw=lw, ls=ls, filled=filled_contours)
                data_i = lhco.source[...,i].cpu().numpy()
                data_j = lhco.source[...,j].cpu().numpy()
                contour_plot(ax, data_j, data_i, bins=(bins_j, bins_i), color='Reds', filled=~filled_contours)
                if j==0: 
                    ax.set_ylabel(features[i])
                ax.set_xlim(bins_j[0], bins_j[-1])
                ax.set_ylim(bins_i[0], bins_i[-1])
                ax.tick_params(direction='in', which='both', length=6)
                if i==0: 
                    ax.set_title(features[j], fontsize=10)
                    ax.axhline(mass_window[0], color='darkred', linestyle='--', lw=0.75)
                    ax.axhline(mass_window[1], color='darkred', linestyle='--', lw=0.75)

            ax.set_xticklabels([])
            
            if j > 0: ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            else: 
                ax.axvline(mass_window[0], color='darkred', linestyle='--', lw=0.75)
                ax.axvline(mass_window[1], color='darkred', linestyle='--', lw=0.75)
                ax.tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False, labelbottom=False, labelleft=True)
                
            ax.grid(True, linestyle=':', color='gray', linewidth=0.5)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        
    # Ratio plots:
    for f in range(len(features)):
        b = np.arange(*bins[f])
        ax_ratio = plt.subplot(gs[num_features, f])
        counts_sb1, _ = np.histogram(lhco.source[...,f], bins=b, density=density)
        counts_sb2, _ = np.histogram(lhco.target[...,f], bins=b, density=density)
        counts_fwd_target, _ = np.histogram(pipeline_fwd.trajectories[-1][...,f], bins=b, density=density)
        counts_bwd_target, _ = np.histogram(pipeline_bwd.trajectories[-1][...,f], bins=b, density=density)
        
        # Ratio plots:
        ratio_sb1 = np.divide(counts_bwd_target, counts_sb1, out=np.zeros_like(counts_fwd_target), where=counts_sb1!=0)
        ratio_sb2 = np.divide(counts_fwd_target, counts_sb2, out=np.zeros_like(counts_bwd_target), where=counts_sb2!=0)
        ax_ratio.plot(b[:-1], ratio_sb1, drawstyle='steps-post', color='darkred')
        ax_ratio.plot(b[:-1], ratio_sb2, drawstyle='steps-post', color='darkblue')
        ax_ratio.set_ylim(0.8, 1.2, 0) 
        ax_ratio.set_xlabel(features[f])
        ax_ratio.axhline(1, color='darkred', linestyle='-', lw=0.75)
        ax_ratio.set_xlim(b[0], b[-1])
        
        for tick in ax_ratio.xaxis.get_major_ticks(): tick.label.set_fontsize(9)
        for tick in ax_ratio.yaxis.get_major_ticks(): tick.label.set_fontsize(9)  
        if f == 0: 
            ax_ratio.axvline(mass_window[0], color='darkred', linestyle='--', lw=0.75)
            ax_ratio.axvline(mass_window[1], color='darkred', linestyle='--', lw=0.75)
            ax_ratio.set_ylabel('ratio', fontsize=8)
            ax_ratio.set_yticks([0.8,  0.9,  1,  1.1,  1.2])
        else: 
            ax_ratio.set_yticks([0.8,  0.9,  1,  1.1,  1.2])
            ax_ratio.tick_params(axis='both', which='both', bottom=True, top=False, left=False, right=False, labelbottom=True, labelleft=False) 
        ax_ratio.grid(True, linestyle=':', color='gray', linewidth=0.5)


    plt.subplots_adjust(wspace=0.1, hspace=0.15)  # Adjust these values as needed
    if save_path_fwd is not None: plt.savefig(save_path_fwd)
    if save_path_bwd is not None: plt.savefig(save_path_bwd)
    plt.show()


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


