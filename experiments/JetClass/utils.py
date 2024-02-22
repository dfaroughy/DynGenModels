import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.gridspec as gridspec

def coord_transform(features, output_coords='pt_eta_phi_m'):

    if output_coords == 'pt_eta_phi_m':
        px, py, pz, e = features[...,0], features[...,1], features[...,2], features[...,3]
        pt = torch.sqrt(px**2 + py**2)
        eta = 0.5 * torch.log( (e + pz) / (e - pz))
        phi = torch.arctan2(py, px)
        m = torch.sqrt(e**2 - px**2 - py**2 - pz**2)
        return torch.stack([pt, eta, phi, m], dim=1)

    elif output_coords == 'px_py_pz_e':
        pt, eta, phi, m = features[...,0], features[...,1], features[...,2], features[...,3]
        mt = torch.sqrt(m**2 + pt**2)
        px, py, pz = pt * torch.cos(phi), pt * torch.sin(phi), mt * torch.sinh(eta)
        e = torch.sqrt(m**2 + px**2 + py**2 + pz**2)    
        return torch.stack([px, py, pz, e], dim=1)


def get_feats_constiutuents(massless_constituents):
    x = massless_constituents.reshape(-1,3)
    x = torch.cat([x, torch.zeros_like(x)[..., None, 0]], dim=-1) # add zero masses
    x = coord_transform(x, output_coords='px_py_pz_e').reshape(-1, massless_constituents.shape[1],4).sum(-2) # sum over particles four-momenta
    return coord_transform(x, output_coords='pt_eta_phi_m') # convert to pt, eta, phi, m
    

def plot_consitutents(trained_model, 
                        save_dir=None, 
                        features=[r'$p^{\rm rel}_T$', r'$\Delta\eta$', r'$\Delta\phi$'], 
                        bins=[np.arange(0,1,0.01), np.arange(-2.5, 2.5, 0.05), np.arange(-1, 1, 0.02)], 
                        num_particles=100000, 
                        figsize=(10, 3)):
    
    jet0 = trained_model.pipeline.trajectories[0].reshape(-1,3).detach().cpu().numpy()
    jet1 = trained_model.pipeline.trajectories[-1].reshape(-1,3).detach().cpu().numpy()
    jet_true = trained_model.dataset.target[:jet0.shape[0]].detach().cpu().numpy()
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, height_ratios=[5, 1])
    gs.update(hspace=0.05) 
    
    for idx, feature in enumerate(features):
        ax = fig.add_subplot(gs[idx])
        h, b, _ = ax.hist(jet_true[..., idx].flatten()[:num_particles], bins=bins[idx], log=True , color='silver', density=True, label='target (truth)')
        h0, _, _ = ax.hist(jet0[..., idx].flatten()[:num_particles], bins=bins[idx], log=True, color=['gold', 'darkblue', 'darkred'][idx], ls='--', histtype='step', density=True, lw=0.75, label='source (t=0)')
        h1, _, _ = ax.hist(jet1[..., idx].flatten()[:num_particles], bins=bins[idx], log=True, color=['gold', 'darkblue', 'darkred'][idx], histtype='step', density=True, lw=0.75 , label='gen. target (t=1)')
        ax.set_xticklabels([])
        ax.set_xticks([])
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        ax.legend(fontsize=6)

        # Ratio plot:
        
        ax_ratio = fig.add_subplot(gs[idx + 3])
        ratio = np.divide(h1, h, out=np.ones_like(h), where=h != 0)
        ax_ratio.plot(0.5 * (b[:-1] + b[1:]), ratio, color=['gold', 'darkblue', 'darkred'][idx],lw=0.75)
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
    else:
        plt.show()


def plot_jets(trained_model, 
                save_dir=None,
                figsize=(12, 2),
                features=[r'$p_t$', r'$\eta$', r'$\phi$', r'$m$'],
                bins = [np.arange(0.9,1.1,0.001), 
                        np.arange(-0.1,0.1,0.001), 
                        np.arange(-0.1,0.1,0.001), 
                        np.arange(0.,1,0.01)
                        ]):

    jet0 = get_feats_constiutuents(trained_model.dataset.source)
    jet1 = get_feats_constiutuents(trained_model.pipeline.trajectories[-1])
    jet_target = get_feats_constiutuents(trained_model.dataset.target)

    fig, ax = plt.subplots(1, 4, figsize=figsize)

    for i in range(4):
        ax[i].hist(jet0[...,i], bins=bins[i], log=True, color=['gold', 'darkblue', 'darkred', 'purple'][i], histtype='step', ls='--', density=True, lw=0.75, label='source (t=0)')
        ax[i].hist(jet1[...,i], bins=bins[i], log=True, color=['gold', 'darkblue', 'darkred', 'purple'][i], histtype='step', density=True, lw=0.75, label='gen. target (t=1)')
        ax[i].hist(jet_target[...,i], bins=bins[i], log=True, color='silver', histtype='stepfilled', density=True, lw=0.75, label='target (truth)')
        ax[i].set_xlabel(features[i])
        ax[i].legend(fontsize=6)
        for tick in ax[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in ax[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
    if save_dir is not None:
        plt.savefig(save_dir + '/jet_features.pdf')
    else:
        plt.show()