
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def results_plots(data, 
                  generated=None, 
                  comparator=None, 
                  save_path=None, 
                  bins=100, 
                  model='generated',
                  features=[r'$\theta$', r'$\phi$', r'$E$ [GeV]'], 
                  num_particles=None):
    
    num_particles = 100000000 if num_particles is None else num_particles
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[5, 1])
    gs.update(hspace=0.1) 
    
    for idx, feature in enumerate(features):
        ax = fig.add_subplot(gs[idx])
        h1, Bins, _ = ax.hist(data[..., idx][:num_particles], bins=bins, color='silver', label='Fermi data')
        if generated is not None:
            h2, _, _ = ax.hist(generated[..., idx][:num_particles], bins=bins, color=['gold', 'darkblue', 'darkred'][idx], histtype='step', lw=0.75, label=model)
            ax.set_xticklabels([])
            ax.set_xticks([])
            for tick in ax.yaxis.get_major_ticks():
               tick.label.set_fontsize(8)
        else: ax.set_xlabel(feature)
        ax.legend(loc='upper right', fontsize=8)

        if generated is not None:

            if comparator=='ratio':
                ax_ratio = fig.add_subplot(gs[idx + 3])
                ratio = np.divide(h1, h2, out=np.ones_like(h2), where=h2 != 0)
                ax_ratio.plot(0.5 * (bins[:-1] + bins[1:]), ratio, color=['gold', 'darkblue', 'darkred'][idx],lw=0.75)
                ax_ratio.set_ylim(0.5, 1.5, 0) 
                ax_ratio.set_xlabel(feature)
                ax_ratio.axhline(1, color='gray', linestyle='--', lw=0.75)
                for tick in ax_ratio.xaxis.get_major_ticks():
                    tick.label.set_fontsize(7)
                for tick in ax_ratio.yaxis.get_major_ticks():
                    tick.label.set_fontsize(5)  
                if idx == 0:
                    ax_ratio.set_ylabel('ratio', fontsize=8)
                ax_ratio.set_yticks([0.5, 1, 1.5])

            if comparator=='pull':
                ax_pull = fig.add_subplot(gs[idx + 3])            
                pull = np.divide(h2 - h1, np.sqrt(h1), out=np.ones_like(h1), where=h1 != 0)
                ax_pull.plot(0.5 * (Bins[:-1] + Bins[1:]), pull, color=['gold', 'darkblue', 'darkred'][idx],lw=0.75)
                ax_pull.set_ylim(-5, 5, 0) # Adjust this as needed
                ax_pull.set_xlabel(feature)
                ax_pull.axhline(0, color='k', linestyle='-', lw=1)
                ax_pull.axhline(-1, color='gray', linestyle=':', lw=0.5)
                ax_pull.axhline(1, color='gray', linestyle=':', lw=0.75)
                ax_pull.axhline(2, color='gray', linestyle=':', lw=0.75)
                ax_pull.axhline(-2, color='gray', linestyle=':', lw=0.75)
                ax_pull.axhline(-3, color='gray', linestyle=':', lw=0.75)
                ax_pull.axhline(3, color='gray', linestyle=':', lw=0.75)
                ax_pull.axhline(4, color='gray', linestyle=':', lw=0.75)
                ax_pull.axhline(-4, color='gray', linestyle=':', lw=0.75)
                for tick in ax_pull.xaxis.get_major_ticks():
                    tick.label.set_fontsize(7)
                for tick in ax_pull.yaxis.get_major_ticks():
                    tick.label.set_fontsize(5)  
                if idx == 0:
                    ax_pull.set_ylabel('pull', fontsize=8)
                ax_pull.set_yticks([-5, -2, 0, 2, 5])

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def results_2D_plots(data, save_path=None, features=[r'$\theta$', r'$\phi$', r'$E$ [GeV]'], gridsize=500, cmap='plasma'):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.667))
    ax[0].hexbin(data[..., 1], data[..., 0], cmap=cmap, gridsize=gridsize)
    ax[0].set_xlabel(features[1])
    ax[0].set_ylabel(features[0])
    ax[1].hexbin(data[..., 0], data[..., 2], cmap=cmap, gridsize=gridsize)
    ax[1].set_xlabel(features[0])
    ax[1].set_ylabel(features[2])
    ax[2].hexbin(data[..., 1], data[..., 2], cmap=cmap, gridsize=gridsize)
    ax[2].set_xlabel(features[1])
    ax[2].set_ylabel(features[2])
    if save_path is not None:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()