import numpy as np
import matplotlib.pyplot as plt


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