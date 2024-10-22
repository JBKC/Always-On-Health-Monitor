'''
File containing various analysis tools for during and after training
'''

import torch
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

def plot_activation_maps(activation_map):
    '''
    Plot the activation maps from a given layer
    '''
    for filter_idx in range(activation_map.shape[1]):  # Iterate over filters (channels)
        plt.figure(figsize=(8, 2))
        plt.plot(activation_map[0, filter_idx].numpy())  # Plot activations for the first input (batch index 0)
        plt.title(f"Activation Map - Filter {filter_idx}")
        plt.show()



def main():
    return

if __name__ == '__main__':
    main()