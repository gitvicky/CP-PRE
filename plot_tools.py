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


def subplots_1d(x_values, y_values, indices, title=None):
    num_subplots = len(indices)
    num_vars = len(y_values)
    colors = ['black', 'blue', 'green', 'red']
    
    # Calculate the number of rows and columns for the subplots
    num_rows = (num_subplots + 1) // 2
    num_cols = 2
    
    # Create a figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4 * num_rows))
   
    # Set the overall plot title
    if title is not None:
        plt.suptitle(title)

    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    
    for ii, idx in enumerate(indices): 
        # Plot the values on the corresponding subplot
        for var in range(num_vars):
            axes[ii].plot(x_values, y_values[list(y_values.keys())[var]][idx],
                        color=colors[var], label=list(y_values.keys())[var])
        # axes[ii].plot(x_values, y_values[list(y_values.keys())[1]][idx], 
        #               color='blue', label=list(y_values.keys())[1])        
        # Set the title and labels for the subplot
        axes[ii].set_title("t = " + str(idx))
        axes[ii].set_xlabel('X')
        axes[ii].set_ylabel('Y')
        axes[ii].legend()
        
        # Set the grid
        axes[ii].grid(True)
    
    # Remove any unused subplots
    for i in range(num_subplots, len(axes)):
        fig.delaxes(axes[i])
    

    # Adjust the spacing between subplots
    plt.tight_layout()
    
    # Display the plot
    plt.show()