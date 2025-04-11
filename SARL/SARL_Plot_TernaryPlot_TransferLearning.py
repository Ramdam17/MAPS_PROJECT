import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pandas as pd
import copy 

def create_ternary_plot(data, colormap='viridis', title='Task Retention in Ternary Space', 
                       vmin=None, vmax=None, ax=None):
    """
    Create a ternary plot from triples of values.
    
    Parameters:
    - data: DataFrame with columns 'task_loss', 'weight_reg_loss', 'feature_loss', 'retention'
            where the first three sum to 1.0 for each row
    - colormap: Matplotlib colormap to use
    - title: Plot title
    - vmin: Minimum value for colorbar (if None, use min of data)
    - vmax: Maximum value for colorbar (if None, use max of data)
    - ax: Axis to plot on (for use in subplots)
    
    Returns:
    - fig, tax: The matplotlib figure and axes
    """
    # Extract data
    x = data['task_loss'].values
    y = data['weight_reg_loss'].values
    z = data['feature_loss'].values
    values = data['retention'].values
    
    # Check if the sum of components is approximately 1
    sums = x + y + z
    if not np.allclose(sums, 1.0, atol=1e-2):
        print("Warning: Some points don't sum to 1.0. Normalizing data.")
        x = x / sums
        y = y / sums
        z = z / sums
    
    # Convert to 2D coordinates for plotting
    # For a ternary plot, we need to convert (x,y,z) to (a,b) points
    a = 0.5 * (2 * y + z) / (x + y + z)
    b = 0.5 * np.sqrt(3) * z / (x + y + z)
    
    # If no axes are provided, create a new plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use provided min/max or calculate from data
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
    
    # Create a triangulation and plot the contour
    triang = tri.Triangulation(a, b)
    levels = np.linspace(vmin, vmax, 11)
    contour = ax.tricontourf(triang, values, levels=levels, cmap=colormap, vmin=vmin, vmax=vmax)
    
    # Add scatter points to show the actual data points
    scatter = ax.scatter(a, b, c=values, cmap=colormap, edgecolor='k', s=50, zorder=10, 
                         vmin=vmin, vmax=vmax)
    
    # Add color bar with percentage notation
    cbar = plt.colorbar(contour, ax=ax)
    
    # Modify the tick labels to show percentages
    tick_locs = cbar.get_ticks()
    tick_labels = [f"{int(x*100)}%" for x in tick_locs]
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_labels)
    
    cbar.set_label('Retention')
    
    # Draw the triangle
    triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-')
    
    # Add tick marks and labels
    # Bottom axis (Task Loss)
    bottom_ticks = np.linspace(0, 1, 6)
    for tick in bottom_ticks:
        # Position and label for bottom axis
        ax.text(tick, -0.05, f"{int((1-tick)*100)}", ha='center', va='top')
        # Small tick mark
        ax.plot([tick, tick], [0, -0.02], 'k-')
    
    # Left axis (Weight Reg Loss)
    for i, tick in enumerate(bottom_ticks[1:]):
        # Calculate position for left axis ticks
        x_pos = 0.5 - tick/2
        y_pos = np.sqrt(3)/2 * (1 - tick)
        ax.text(x_pos - 0.05, y_pos, f"{int(tick*100)}", ha='right', va='center')
        # Small tick mark - calculate a short perpendicular line
        dx = 0.02 * np.cos(np.pi/6)
        dy = 0.02 * np.sin(np.pi/6)
        ax.plot([x_pos, x_pos - dx], [y_pos, y_pos - dy], 'k-')
    ax.set_title(title)
    ax.set_axis_off()
    return ax  # Return the axis object for further customization if needed

# Example of usage with your own data
my_data = pd.DataFrame([
    {'task_loss': 0.1, 'weight_reg_loss': 0.1, 'feature_loss': 0.8, 'retention': 0.0},
    {'task_loss': 0.1, 'weight_reg_loss': 0.2, 'feature_loss': 0.7, 'retention': 0.1},
    {'task_loss': 0.1, 'weight_reg_loss': 0.3, 'feature_loss': 0.6, 'retention': 0.1},
    {'task_loss': 0.1, 'weight_reg_loss': 0.4, 'feature_loss': 0.5, 'retention': 0.3},
    {'task_loss': 0.1, 'weight_reg_loss': 0.5, 'feature_loss': 0.4, 'retention': 0.8},
    {'task_loss': 0.1, 'weight_reg_loss': 0.6, 'feature_loss': 0.3, 'retention': 0.8},
    {'task_loss': 0.1, 'weight_reg_loss': 0.7, 'feature_loss': 0.2, 'retention': 0.5},
    {'task_loss': 0.1, 'weight_reg_loss': 0.8, 'feature_loss': 0.1, 'retention': 0.0},
    {'task_loss': 0.1, 'weight_reg_loss': 0.9, 'feature_loss': 0.0, 'retention': 0.6},
    
    {'task_loss': 0.2, 'weight_reg_loss': 0.1, 'feature_loss': 0.7, 'retention': 0.0},
    {'task_loss': 0.2, 'weight_reg_loss': 0.2, 'feature_loss': 0.6, 'retention': 0.1},
    {'task_loss': 0.2, 'weight_reg_loss': 0.3, 'feature_loss': 0.5, 'retention': 0.1},
    {'task_loss': 0.2, 'weight_reg_loss': 0.4, 'feature_loss': 0.4, 'retention': 0.3},
    {'task_loss': 0.2, 'weight_reg_loss': 0.5, 'feature_loss': 0.3, 'retention': 0.8},
    {'task_loss': 0.2, 'weight_reg_loss': 0.6, 'feature_loss': 0.2, 'retention': 0.8},
    {'task_loss': 0.2, 'weight_reg_loss': 0.7, 'feature_loss': 0.1, 'retention': 0.5},
    {'task_loss': 0.2, 'weight_reg_loss': 0.8, 'feature_loss': 0.0, 'retention': 0.0},
    

    {'task_loss': 0.3, 'weight_reg_loss': 0.1, 'feature_loss': 0.6, 'retention': 0.6},
    {'task_loss': 0.3, 'weight_reg_loss': 0.2, 'feature_loss': 0.5, 'retention': 0.2},
    {'task_loss': 0.3, 'weight_reg_loss': 0.3, 'feature_loss': 0.4, 'retention': 1.0},
    {'task_loss': 0.3, 'weight_reg_loss': 0.4, 'feature_loss': 0.3, 'retention': 0.9},
    {'task_loss': 0.3, 'weight_reg_loss': 0.5, 'feature_loss': 0.2, 'retention': 0.5},
    {'task_loss': 0.3, 'weight_reg_loss': 0.6, 'feature_loss': 0.1, 'retention': 0.2},
    {'task_loss': 0.3, 'weight_reg_loss': 0.7, 'feature_loss': 0.0, 'retention': 0.0},

    {'task_loss': 0.4, 'weight_reg_loss': 0.1, 'feature_loss': 0.5, 'retention': 0.6},
    {'task_loss': 0.4, 'weight_reg_loss': 0.2, 'feature_loss': 0.4, 'retention': 0.2},
    {'task_loss': 0.4, 'weight_reg_loss': 0.3, 'feature_loss': 0.3, 'retention': 1.0},
    {'task_loss': 0.4, 'weight_reg_loss': 0.4, 'feature_loss': 0.2, 'retention': 0.9},
    {'task_loss': 0.4, 'weight_reg_loss': 0.5, 'feature_loss': 0.1, 'retention': 0.5},
    {'task_loss': 0.4, 'weight_reg_loss': 0.6, 'feature_loss': 0.0, 'retention': 0.2},
    
    {'task_loss': 0.5, 'weight_reg_loss': 0.1, 'feature_loss': 0.4, 'retention': 0.0},
    {'task_loss': 0.5, 'weight_reg_loss': 0.2, 'feature_loss': 0.3, 'retention': 0.4},
    {'task_loss': 0.5, 'weight_reg_loss': 0.3, 'feature_loss': 0.2, 'retention': 2.8},
    {'task_loss': 0.5, 'weight_reg_loss': 0.4, 'feature_loss': 0.1, 'retention': 2.3},
    {'task_loss': 0.5, 'weight_reg_loss': 0.5, 'feature_loss': 0.0, 'retention': 0.4},

    {'task_loss': 0.6, 'weight_reg_loss': 0.1, 'feature_loss': 0.3, 'retention': 0.0},
    {'task_loss': 0.6, 'weight_reg_loss': 0.2, 'feature_loss': 0.2, 'retention': 0.4},
    {'task_loss': 0.6, 'weight_reg_loss': 0.3, 'feature_loss': 0.1, 'retention': 2.8},
    {'task_loss': 0.6, 'weight_reg_loss': 0.4, 'feature_loss': 0.0, 'retention': 2.3},
    
    {'task_loss': 0.7, 'weight_reg_loss': 0.1, 'feature_loss': 0.2, 'retention': 0.2},
    {'task_loss': 0.7, 'weight_reg_loss': 0.2, 'feature_loss': 0.1, 'retention': 1.5},
    {'task_loss': 0.7, 'weight_reg_loss': 0.3, 'feature_loss': 0.0, 'retention': 0.3},

    {'task_loss': 0.8, 'weight_reg_loss': 0.1, 'feature_loss': 0.1, 'retention': 0.2},
    {'task_loss': 0.8, 'weight_reg_loss': 0.2, 'feature_loss': 0.0, 'retention': 1.5},
    
    {'task_loss': 0.9, 'weight_reg_loss': 0.1, 'feature_loss': 0.0, 'retention': 0.5},
    {'task_loss': 1.0, 'weight_reg_loss': 0.0, 'feature_loss': 0.0, 'retention': 0.5},
    {'task_loss': 0.0, 'weight_reg_loss': 1.0, 'feature_loss': 0.0, 'retention': 4.3},
    {'task_loss': 0.0, 'weight_reg_loss': 0.0, 'feature_loss': 1.0, 'retention': 0.7},

])

max_my_data= my_data.copy()
stv_my_data=my_data.copy()


seed_values = {
    (0.1, 0.1, 0.8): [0.0, 0.2, 0.0, 0.7, 0.1],
    (0.1, 0.2, 0.7): [0.4, 0.7, 0.0, 0.2, 0.0],
    (0.1, 0.3, 0.6): [0.2, 0.4, 0.3, 0.6, 0.2],
    
    (0.1, 0.4, 0.5): [0.5, 0.2, 0.1, 0.3, 0.4],
    (0.1, 0.5, 0.4): [0.5, 1.0, 0.4, 0.4, 1.3],
    (0.1, 0.6, 0.3): [0.5, 2.0, 0.1, 0.8, 1.7],
    
    (0.1, 0.7, 0.2): [2.5, 2.1, 1.7, 1.1, 0.1],
    (0.1, 0.8, 0.1): [0.2, 0.5, 1.1, 1.4, 0.2],
    (0.1, 0.9, 0.0): [0.0, 1.4, 0.4, 0.9, 0.7],
    
    ##########

    (0.2, 0.1, 0.7): [0.5, 0.4, 0.4, 0.4, 0.7],
    (0.2, 0.2, 0.6): [0.0, 0.2, 0.0, 0.4, 0.3],
    (0.2, 0.3, 0.5): [0.1, 0.1, 0.2, 0.1, 0.5],
    
    (0.2, 0.4, 0.4): [0.6, 1.3, 0.0, 0.8, 1.2],
    (0.2, 0.5, 0.3): [0.0, 2.8, 1.0, 1.0, 0.1],
    (0.2, 0.6, 0.2): [1.7, 0.6, 0.5, 0.6, 0.6],
    
    (0.2, 0.7, 0.1): [0.5, 0.1, 1.4, 0.6, 0.8],
    (0.2, 0.8, 0.0): [2.3, 0.3, 0.6, 0.5, 0.0],
    
    ###########
    
    (0.3, 0.1, 0.6): [0.5, 0.1, 0.5, 0.8, 0.5],
    (0.3, 0.2, 0.5): [0.9, 0.0, 0.4, 1.0, 0.1],
    (0.3, 0.3, 0.4): [0.4, 0.0, 0.6, 0.2, 0.3],
    
    (0.3, 0.4, 0.3): [0.4, 0.3, 0.4, 0.6, 0.5],
    (0.3, 0.5, 0.2): [0.7, 0.5, 0.6, 3.0, 0.0],
    (0.3, 0.6, 0.1): [0.4, 2.1, 2.2, 1.8, 0.6],
    
    (0.3, 0.7, 0.0): [0.7, 1.3, 0.8, 0.2, 1.1],

    ###########

    (0.4, 0.1, 0.5): [0.0, 0.4, 0.5, 0.8, 0.2],
    (0.4, 0.2, 0.4): [0.0, 0.0, 0.1, 0.5, 0.5],
    (0.4, 0.3, 0.3): [0.4, 0.5, 0.7, 1.6, 0.3],
    
    (0.4, 0.4, 0.2): [2.6, 0.1, 0.6, 1.9, 3.1],
    (0.4, 0.5, 0.1): [1.0, 0.5, 0.4, 0.3, 1.9],
    (0.4, 0.6, 0.0): [0.9, 0.0, 1.7, 0.4, 1.1],


    ###########
    
    (0.5, 0.1, 0.4): [0.5, 0.6, 0.9, 0.7, 0.6],
    (0.5, 0.2, 0.3): [0.4, 0.0, 0.3, 0.7, 0.7],
    (0.5, 0.3, 0.2): [0.7, 0.2, 0.6, 1.3, 0.8],
    
    (0.5, 0.4, 0.1): [1.0, 1.1, 0.1, 0.4, 0.8],
    (0.5, 0.5, 0.0): [1.3, 0.0, 1.7, 0.3, 0.4],

    ###########
    
    (0.6, 0.1, 0.3): [0.9, 0.6, 0.8, 1.0, 0.0],
    (0.6, 0.2, 0.2): [1.2, 0.1, 0.3, 0.3, 0.4],
    (0.6, 0.3, 0.1): [1.6, 0.6, 2.5, 0.2, 0.3],
    
    (0.6, 0.4, 0.0): [0.4, 0.4, 0.1, 1.2, 0.7],


    ###########

    (0.7, 0.1, 0.2): [0.9, 0.7, 0.3, 0.2, 0.7],
    (0.7, 0.2, 0.1): [0.7, 0.4, 0.0, 0.0, 0.3],
    (0.7, 0.3, 0.0): [0.3, 0.4, 1.1, 2.0, 1.0],

    ###########

    (0.8, 0.1, 0.1): [0.6, 0.6, 0.4, 0.5, 0.5],
    (0.8, 0.2, 0.0): [0.5, 0.0, 0.6, 0.5, 0.7],

    ###########
 
    (0.9, 0.1, 0.0): [0.4, 0.2, 0.7, 0.6, 0.4],

    ###########

    (1.0, 0.0, 0.0): [0.2, 0.3, 0.4, 0.5, 0.4],
    (0.0, 1.0, 0.0): [3.2, 3.6, 3.7, 4.5, 3.6],
    
    (0.0, 0.0, 1.0): [0.0, 1.0, 0.7, 0.5, 0.5],  # Example from your message
}

original_reward = np.mean([3.6, 2.7, 4.0, 5.0, 3.1])

for index, row in my_data.iterrows():
    key = (row['task_loss'], row['weight_reg_loss'], row['feature_loss'])
    if key in seed_values:
        my_data.at[index, 'retention'] = np.mean(seed_values[key])
        max_my_data.at[index, 'retention'] = np.max(seed_values[key]/original_reward)
        stv_my_data.at[index, 'retention'] = np.std(seed_values[key]/original_reward)

my_data['retention'] = my_data['retention'] / original_reward

# Find min and max normalized retention values for each dataset
min_retention = my_data['retention'].min()
max_retention = my_data['retention'].max()

min_retention_maxvariant = max_my_data['retention'].min()
max_retention_maxvariant = max_my_data['retention'].max()

min_retention_stv = stv_my_data['retention'].min()
max_retention_stv = stv_my_data['retention'].max()

# Create filtered datasets (excluding the specific point)
# For mean data
filtered_data = my_data.copy()
filtered_index = filtered_data[
    (filtered_data['task_loss'] == 0.0) & 
    (filtered_data['weight_reg_loss'] == 1.0) & 
    (filtered_data['feature_loss'] == 0.0)
].index
filtered_data = filtered_data.drop(filtered_index)
modified_max_retention = filtered_data['retention'].max()

# For max data
filtered_data_maxvariant = max_my_data.copy()
filtered_index_maxvariant = filtered_data_maxvariant[
    (filtered_data_maxvariant['task_loss'] == 0.0) &
    (filtered_data_maxvariant['weight_reg_loss'] == 1.0) &
    (filtered_data_maxvariant['feature_loss'] == 0.0)
].index
filtered_data_maxvariant = filtered_data_maxvariant.drop(filtered_index_maxvariant)
modified_max_retention_maxvariant = filtered_data_maxvariant['retention'].max()

# For std data
filtered_data_stdvariant = stv_my_data.copy()
filtered_index_stdvariant = filtered_data_stdvariant[
    (filtered_data_stdvariant['task_loss'] ==.0) &
    (filtered_data_stdvariant['weight_reg_loss'] == 1.0) &
    (filtered_data_stdvariant['feature_loss'] == 0.0)
].index
filtered_data_stdvariant = filtered_data_stdvariant.drop(filtered_index_stdvariant)
modified_max_retention_stdvariant = filtered_data_stdvariant['retention'].max()

# Create a 3x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(12, 24))

# Row 1: Mean retention plots
create_ternary_plot(
    my_data,
    title='Mean Task Retention (Full Range)',
    vmin=min_retention,
    vmax=max_retention,
    ax=axes[0, 0]
)

create_ternary_plot(
    my_data,
    title='Mean Task Retention (Excluding weight regularization loss = 1.0)',
    vmin=min_retention,
    vmax=modified_max_retention,
    ax=axes[0, 1]
)

# Row 2: Max retention plots
create_ternary_plot(
    max_my_data,
    title='Max Task Retention (Full Range)',
    vmin=min_retention_maxvariant,
    vmax=max_retention_maxvariant,
    ax=axes[1, 0]
)

create_ternary_plot(
    max_my_data,
    title='Max Task Retention (Excluding weight regularization loss = 1.0)',
    vmin=min_retention_maxvariant,
    vmax=modified_max_retention_maxvariant,
    ax=axes[1, 1]
)

# Row 3: Standard deviation plots
create_ternary_plot(
    stv_my_data,
    title='Std Task Retention (Full Range)',
    vmin=min_retention_stv,
    vmax=max_retention_stv,
    ax=axes[2, 0]
)

create_ternary_plot(
    stv_my_data,
    title='Std Task Retention (Excluding weight regularization loss = 1.0)',
    vmin=min_retention_stv,
    vmax=modified_max_retention_stdvariant,
    ax=axes[2, 1]
)

# Add a main title for the entire figure
fig.suptitle('Task Retention Analysis in Ternary Space', fontsize=20, y=0.995)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.99])  # The rect parameter reserves space for the suptitle
plt.subplots_adjust(hspace=0.3)  # Add more space between rows

plt.show()

# Find the row(s) with highest retention value after filtering (for reporting purposes)
highest_retention_row = filtered_data[filtered_data['retention'] == modified_max_retention]
highest_retention_values = highest_retention_row[['task_loss', 'weight_reg_loss', 'feature_loss']]
print(f"Best mean retention value after filtering: {modified_max_retention}")
print(f"Corresponding values (task_loss, weight_reg_loss, feature_loss):")
print(highest_retention_values)
