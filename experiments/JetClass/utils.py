import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_jet_trajectories(trained_model, bins=[50,50,50], figsize=(15,5)):

    x0 = trained_model.pipeline.trajectories[0].reshape(-1,3).detach().cpu().numpy()
    x1 = trained_model.pipeline.trajectories[-1].reshape(-1,3).detach().cpu().numpy()
    target = trained_model.dataset.target[:x0.shape[0]].detach().cpu().numpy()
    
    _, axs = plt.subplots(1, 3, figsize=figsize)
    axs[0].hist(target[..., 0].flatten(), bins=bins[0],log=True, color='darkblue', alpha=0.25,  histtype='stepfilled', label='top', density=True)
    axs[0].hist(x0[..., 0].flatten(), bins=bins[0],log=True, color='darkred', histtype='step', label='qcd source  (t=0)', density=True)
    axs[0].hist(x1[..., 0].flatten(), bins=bins[0],log=True, color='darkblue', histtype='step', label='target (t=1)', density=True)

    axs[1].hist(target[..., 1].flatten(), bins=bins[1],log=True, color='darkblue',  alpha=0.25,  histtype='stepfilled', label='top', density=True)
    axs[1].hist(x0[..., 1].flatten(), bins=bins[1], log=True, color='darkred', histtype='step', label='qcd (source)', density=True)
    axs[1].hist(x1[..., 1].flatten(), bins=bins[0],log=True, color='darkblue', histtype='step', label='target (t=1)', density=True)

    axs[2].hist(target[..., 2].flatten(), bins=bins[2],log=True, color='darkblue',  alpha=0.25,  histtype='stepfilled', label='top', density=True)
    axs[2].hist(x0[..., 2].flatten(), bins=bins[2], log=True, color='darkred', histtype='step', label='source (t=0)', density=True)
    axs[2].hist(x1[..., 2].flatten(), bins=bins[0],log=True, color='darkblue', histtype='step', label='target (t=1)', density=True)

    axs[0].set_xlabel(r'$p_t^{rel}$ constituents')
    axs[1].set_xlabel(r'$\Delta\eta$ constituents')
    axs[2].set_xlabel(r'$\Delta\phi$ constituents')
    axs[0].set_ylabel('density')
    axs[2].legend(loc='upper right', fontsize=7)
    plt.tight_layout()
    plt.show()


def results_plots(trained_model, save_dir=None, features=[r'$p^{\rm rel}_T$', r'$\Delta\eta$', r'$\Delta\phi$'], num_particles=100000, figsize=(10, 3)):
    
    jet0 = trained_model.pipeline.trajectories[0].reshape(-1,3).detach().cpu().numpy()
    jet1 = trained_model.pipeline.trajectories[-1].reshape(-1,3).detach().cpu().numpy()
    jet_true = trained_model.dataset.target[:jet0.shape[0]].detach().cpu().numpy()
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, height_ratios=[5, 1])
    gs.update(hspace=0.05) 
    
    for idx, feature in enumerate(features):
        ax = fig.add_subplot(gs[idx])
        h, bins, _ = ax.hist(jet_true[..., idx].flatten()[:num_particles], bins=100, log=True   , color='silver', density=True, label='truth')
        h0, _, _ = ax.hist(jet0[..., idx].flatten()[:num_particles], bins=100, log=True, color=['gold', 'darkblue', 'darkred'][idx], ls='--', histtype='step', density=True, lw=0.75, label='source (t=0)')
        h1, _, _ = ax.hist(jet1[..., idx].flatten()[:num_particles], bins=100, log=True, color=['gold', 'darkblue', 'darkred'][idx], histtype='step', density=True, lw=0.75 , label='target (t=1)')
        ax.set_xticklabels([])
        ax.set_xticks([])
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        ax.legend(fontsize=6)

        # Ratio plot:
        
        ax_ratio = fig.add_subplot(gs[idx + 3])
        ratio = np.divide(h1, h, out=np.ones_like(h), where=h != 0)
        ax_ratio.plot(0.5 * (bins[:-1] + bins[1:]), ratio, color=['gold', 'darkblue', 'darkred'][idx],lw=0.75)
        ax_ratio.set_ylim(0.5, 1.5, 0) # Adjust this as needed
        ax_ratio.set_xlabel(feature)
        ax_ratio.axhline(1, color='gray', linestyle='--', lw=0.75)
        for tick in ax_ratio.xaxis.get_major_ticks():
            tick.label.set_fontsize(7)
        for tick in ax_ratio.yaxis.get_major_ticks():
            tick.label.set_fontsize(5)  
        if idx == 0:
            ax_ratio.set_ylabel('ratio', fontsize=8)
        ax_ratio.set_yticks([0.5, 1, 1.5])
    if save_dir is not None:
        plt.savefig(save_dir + '/particle_features.pdf')
    plt.show()