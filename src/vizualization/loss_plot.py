import matplotlib.pyplot as plt
import numpy as np

def draw_loss(t_loss, val_loss):
    # Plot train and validation loss
    plt.figure(figsize=(10, 6))
    # train_losses = []
    # val_losses = []
    xaxis = np.arange(0, 30, 1)
    plt.plot(xaxis, t_loss, label='Training Loss')
    plt.plot(xaxis, val_loss, label='Validation Loss')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # Set the tick locations
    plt.xticks(xaxis)

    # Display the plot
    plt.legend(loc='best')
    plt.show()