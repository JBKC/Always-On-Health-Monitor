'''
File containing various analysis tools for during and after training
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import inspect

def plot_weight_dist(model):
    for name, param in model.named_parameters():
        print(name)
        if param.requires_grad and 'conv' in name:  # Focus on convolutional layers
            weights = param.data.cpu().numpy().flatten()  # Flatten the weights
            plt.figure(figsize=(8, 6))
            plt.hist(weights, bins=50, alpha=0.75)
            plt.title(f'Weight Distribution for {name}')
            plt.xlabel('Weight values')
            plt.ylabel('Frequency')
            plt.show()
    return

activations = {}

def forward_hook(module, input, output):
    '''
    Hook function to capture forward activations.
    '''
    activations[module] = output.detach()

def register_hooks(layer):
    '''
    Registers forward hooks for given layer(s)
    '''
    return layer.register_forward_hook(forward_hook)  # Register forward hook for each layer

def plot_activation_map(activation_map):
    '''
    Plot the heatmap of the activations for each filter in the given layer as a single grid
    '''

    activation_map = activation_map[0].detach().cpu().numpy()  # Ensure it's a NumPy array


    activation_map_normalized = activation_map / np.max(np.abs(activation_map))  # normalise

    # Plot the heatmap
    plt.figure(figsize=(8, 6))  # Adjust size to make the heatmap square
    plt.imshow(activation_map_normalized, aspect='auto', cmap='seismic', interpolation='nearest')
    plt.colorbar(label='Activation Intensity')

    plt.title(f"Activation for")
    plt.xlabel('sample index')
    plt.ylabel('filter index')
    plt.show()



def main():
    return

if __name__ == '__main__':
    main()