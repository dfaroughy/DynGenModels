import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

from DynGenModels.models.architectures.LeNet import LeNet5 

def mnist_classifier(pipeline, plot_histogram=False):
    classifier = LeNet5(num_classes=10)
    classifier.load_state_dict(torch.load('/home/df630/DynGenModels/src/DynGenModels/metrics/mnist/LeNet5_MNIST.pth'))
    classifier.eval()
    classes = classifier(pipeline.trajectories[-1].view(-1, 1, 28, 28).clip(0, 1)).argmax(dim=1)
    classes = classes.tolist()
    if plot_histogram:
        plt.subplots(figsize=(2.5,2.5))
        unique, counts = np.unique(classes, return_counts=True)
        plt.bar(unique, counts)
        plt.xticks(range(10))
        plt.show()
    else:
        return classes
    
def plot_image_evolution(pipeline, nrow=10, figsize=(3, 3),  step_list=None):
    plt.figure(figsize=figsize)
    step_list = [i for i in range(0, pipeline.trajectories.shape[0], 10)] if step_list is None else step_list
    res = torch.cat([pipeline.trajectories[step, :nrow] for step in step_list], dim=0)
    grid = make_grid(res.view([-1, 1, 28, 28]).clip(0, 1), value_range=(0, 1), padding=0, nrow=nrow)
    img = ToPILImage()(grid)
    plt.imshow(img)
    plt.axis('off')