import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def results_plots(gaia_data, 
                  generated=None, 
                  save_dir=None, 
                  features=[r'$x$', r'y$', r'$z$'], 
                  num_particles=100000,
                  name_file='gaia'):
    
    fig = plt.figure(figsize=(10, 3))
    gs = gridspec.GridSpec(2, 3, height_ratios=[5, 1])
    gs.update(hspace=0.1) 
    
    for idx, feature in enumerate(features):
        ax = fig.add_subplot(gs[idx])
        h1, bins, _ = ax.hist(gaia_data[..., idx].flatten()[:num_particles], bins=100,  color='silver', density=True, log=True)
        if generated is not None:
            h2, _, _ = ax.hist(generated[..., idx].flatten()[:num_particles], bins=100, color=['gold', 'darkblue', 'darkred'][idx], histtype='step', density=True, log=True, lw=0.75)
            ax.set_xticklabels([])
            ax.set_xticks([])
            for tick in ax.yaxis.get_major_ticks():
               tick.label.set_fontsize(8)
        else:
            ax.set_xlabel(feature)
        
        # Ratio plot
        if generated is not None:
            ax_ratio = fig.add_subplot(gs[idx + 3])
            ratio = np.divide(h1, h2, out=np.ones_like(h2), where=h2 != 0)
            ax_ratio.plot(0.5 * (bins[:-1] + bins[1:]), ratio, color=['gold', 'darkblue', 'darkred'][idx],lw=0.75)
            ax_ratio.set_ylim(0.5, 1.5, 0) # Adjust this as needed
            ax_ratio.set_xlabel(feature)
            ax_ratio.axhline(1, color='gray', linestyle='--', lw=0.75)
            for tick in ax_ratio.xaxis.get_major_ticks():
               tick.label.set_fontsize(7)
            for tick in ax_ratio.yaxis.get_major_ticks():
              tick.label.set_fontsize(6)  
            if idx == 0:
                ax_ratio.set_ylabel('ratio', fontsize=8)
            ax_ratio.set_yticks([0.5, 1, 1.5])
    if save_dir is not None:
        plt.savefig(save_dir + '/'+name_file+'.pdf')
    plt.show()


