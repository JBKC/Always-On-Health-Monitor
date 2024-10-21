'''
File containing analysis for during and after training
'''

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




def main():
    return

if __name__ == '__main__':
    main()