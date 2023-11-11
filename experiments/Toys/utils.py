import matplotlib.pyplot as plt

def plot_trajs(trajectories, num_sampling_steps=100):
    fig, ax = plt.subplots(1,1, figsize=(4,4))
    for i in range(num_sampling_steps): 
        plt.scatter(trajectories[i][:,0],trajectories[i][:,1], s=0.1, color='gray', alpha=0.2)
    plt.scatter(trajectories[0][:,0], trajectories[0][:,1], s=1, color='red')
    plt.scatter(trajectories[-1][:,0],trajectories[-1][:,1], s=1, color='blue')
    plt.xlim(-7,7)
    plt.ylim(-7,7)
    plt.xticks([])
    plt.yticks([])
    plt.show()