import numpy as np
import matplotlib.pyplot as plt

def plot_loss(valid_loss, train_loss, workdir='./'):
    fig, _ = plt.subplots(figsize=(4,4))
    plt.plot(range(len(valid_loss)), np.array(valid_loss), color='r', lw=1, label='valid')
    plt.plot(range(len(train_loss)), np.array(train_loss), color='b', lw=1, label='train', alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    fig.tight_layout()
    plt.savefig(workdir+'/losses.png')
    plt.close()


