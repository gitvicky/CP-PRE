import matplotlib.pyplot as plt
from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm


def subplots_2d(values, titles, flatten=False):
    num_subplots = len(values)
    
    # Calculate the number of rows and columns for the subplots
    if flatten == True: 
        num_rows = 1 
        num_cols = num_subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5))

    else:
        num_rows = (num_subplots + 1) // 2
        num_cols = 2
    
        # Create a figure and subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))
        
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    
    for i, (value, title) in enumerate(zip(values, titles)):
        # Plot the value using imshow on the corresponding subplot
        im = axes[i].imshow(value, cmap='jet')
        
        # Set the title for the subplot
        axes[i].set_title(title)
        
        # Remove the axis ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        # Add a colorbar to the subplot
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        axes[i].tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

    # Remove any unused subplots
    for i in range(num_subplots, len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust the spacing between subplots
    plt.tight_layout()
    
    # Display the plot
    plt.show()

import matplotlib.pyplot as plt


def subplots_1d(x_values, y_values, titles):
    num_subplots = len(y_values)
    
    # Calculate the number of rows and columns for the subplots
    num_rows = (num_subplots + 1) // 2
    num_cols = 2
    
    # Create a figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4 * num_rows))
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    
    for i, (x, y, title) in enumerate(zip(x_values, y_values, titles)):
        # Plot the values on the corresponding subplot
        axes[i].plot(x, y[0], color='black')
        axes[i].plot(x, y[1], color='blue')
        
        # Set the title and labels for the subplot
        axes[i].set_title(title)
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        
        # Set the grid
        axes[i].grid(True)
    
    # Remove any unused subplots
    for i in range(num_subplots, len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust the spacing between subplots
    plt.tight_layout()
    
    # Display the plot
    plt.show()