'''
File containing various analysis tools for during and after training
'''

import torch
import numpy as np
import matplotlib.pyplot as plt

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

def register_hook(layer):
    '''
    Registers a forward hook to a specific layer.
    '''
    return layer.register_forward_hook(forward_hook)

def plot_activation_map(activation_map):
    '''
    Plot the heatmap of the activations for each filter in the given layer as a single grid
    '''
    # Take the first input in the batch and detach from the computational graph
    activation_map = activation_map[0].detach().cpu().numpy()  # Shape: (num_filters, num_samples)

    # Normalize the activation values to the range [-1, 1] for consistent heatmap scaling
    activation_map_normalized = activation_map / np.max(np.abs(activation_map))  # Normalize based on absolute max

    # Plot the heatmap
    plt.figure(figsize=(8, 6))  # Adjust size to make the heatmap square
    plt.imshow(activation_map_normalized, aspect='auto', cmap='seismic', interpolation='nearest')
    plt.colorbar(label='Activation Intensity')

    plt.title(f"Activation Map Grid (Filters vs Samples)")
    plt.xlabel('Sequence Position (Samples)')
    plt.ylabel('Filter Index')
    plt.show()



def main():
    return

if __name__ == '__main__':
    main()