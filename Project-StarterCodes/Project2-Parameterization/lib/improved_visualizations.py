import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

def plot_input_distributions(variables):
    """
    Create improved histograms for input variables.
    
    Parameters:
        variables (dict): Dictionary of variable names and their data
    """
    plt.figure(figsize=(15, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, (name, data) in enumerate(variables.items()):
        plt.subplot(3, 3, i + 1)
        plt.hist(data, bins=20, color=colors[i % len(colors)], 
                 edgecolor="black", alpha=0.7, linewidth=1.5)
        plt.title(name, fontsize=12, fontweight='bold')
        plt.xlabel("Value", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
    
    return plt.gcf()

def plot_diffusivity_distributions(SF0):
    """
    Create improved histograms for diffusivity distributions.
    
    Parameters:
        SF0: Array of diffusivity values
    """
    plt.figure(figsize=(15, 10))
    
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.hist(SF0[:, i].values.flatten(), bins=20, 
                 color="#2ca02c", edgecolor="black", 
                 alpha=0.8, linewidth=1.2)
        plt.title(f"Layer {i+1} diffusivity", fontsize=11, fontweight='bold')
        plt.xlabel("Diffusivity (m$^2$s$^{-1}$)", fontsize=9)
        plt.ylabel("Frequency", fontsize=9)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    plt.tight_layout(pad=1.5)
    return plt.gcf()

def plot_correlation_heatmap(df, k_std, k_mean):
    """
    Create improved correlation heatmap.
    
    Parameters:
        df: DataFrame with input and output variables
        k_std: Standard deviation values for normalization
        k_mean: Mean values for normalization
    """
    correlation_matrix = df.corr()
    sf0_correlation = correlation_matrix.loc["SF0_1":"SF0_16", ["l0", "b00", "ustar0", "h0", "lat0", "heat0", "tx0"]]
    
    plt.figure(figsize=(14, 8))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    ax = sns.heatmap(sf0_correlation, annot=True, cmap=cmap, fmt=".2f", 
                    cbar=True, square=True, linewidths=.5)
    
    # Add bold labels
    plt.title("Correlation between Input Variables and Diffusivity by Layer", 
              fontsize=16, fontweight='bold')
    
    plt.xlabel("Input Variables", fontsize=14)
    plt.ylabel("Diffusivity Layers (from bottom to top)", fontsize=14)
    
    # Add informative annotation
    plt.annotate(
        "Note: SF0_1 is at the bottom of the OSBL, SF0_16 is at the top",
        xy=(0.5, -0.15), xycoords='axes fraction', 
        ha='center', fontsize=12, 
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
    )
    
    return plt.gcf()

def plot_shape_functions(sig, sfmin, sfmax, z, z1):
    """
    Create improved visualization of shape functions.
    
    Parameters:
        sig: Sigma values
        sfmin: Minimum shape function values
        sfmax: Maximum shape function values
        z: z values for universal shape function
        z1: Universal shape function values
    """
    plt.figure(figsize=(8, 10))
    
    # Enhanced styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12
    
    # Plot shape functions
    plt.fill_betweenx(sig, sfmin, sfmax, color='lightblue', 
                     alpha=0.7, edgecolor='blue', linewidth=1.2,
                     label='SMC shape function range')
    
    plt.plot(z1[::-1], z, color='darkred', linestyle='--', 
            linewidth=2.5, label='Universal shape function')
    
    # Add clear labeling
    plt.title('Shape Functions in Ocean Surface Boundary Layer', 
             fontsize=16, fontweight='bold')
    
    plt.xlabel(r'$g(\sigma)$ (normalized diffusivity)', fontsize=14)
    plt.ylabel(r'$\sigma$ (normalized depth)', fontsize=14)
    
    # Add an annotation to better explain
    plt.annotate('Surface', xy=(0.8, 0.02), xytext=(0.5, 0.05),
                arrowprops=dict(arrowstyle='->'))
    
    plt.annotate('Bottom of mixed layer', xy=(0.8, 0.98), xytext=(0.4, 0.95),
                arrowprops=dict(arrowstyle='->'))
    
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(alpha=0.3)
    
    return plt.gcf()

def plot_training_validation_loss(loss_array):
    """
    Plot improved training and validation loss curves.
    
    Parameters:
        loss_array: Array containing epoch, training loss, and validation loss
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(loss_array[:, 0], loss_array[:, 1], 
            color='blue', linewidth=2.5, label="Training Loss")
    
    plt.plot(loss_array[:, 0], loss_array[:, 2], 
            color='red', linewidth=2.5, label="Validation Loss")
    
    # Add a horizontal line showing best validation loss
    best_val_epoch = np.argmin(loss_array[:, 2])
    best_val_loss = loss_array[best_val_epoch, 2]
    
    plt.axhline(y=best_val_loss, color='green', linestyle='--', alpha=0.7)
    plt.axvline(x=loss_array[best_val_epoch, 0], color='green', linestyle='--', alpha=0.7)
    
    plt.annotate(f'Best validation: {best_val_loss:.4f}',
                xy=(loss_array[best_val_epoch, 0], best_val_loss),
                xytext=(loss_array[best_val_epoch, 0] + 50, best_val_loss + 0.02),
                arrowprops=dict(arrowstyle='->'))
    
    # Add shading for convergence zones
    plt.fill_between(loss_array[:, 0], loss_array[:, 1], loss_array[:, 2], 
                    alpha=0.2, color='gray', label='Generalization gap')
    
    # Enhanced formatting
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training and Validation Loss Over Epochs", 
             fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    return plt.gcf()

def plot_enhanced_performance(model, x, valid_x, y, valid_y, k_mean, k_std):
    """
    Enhanced version of the performance_sigma_point function with better visualizations.
    
    Parameters:
        model: Trained neural network model
        x: Training input data
        valid_x: Validation input data
        y: Training output data
        valid_y: Validation output data
        k_mean: Mean values for normalization
        k_std: Standard deviation values for normalization
    """
    # Configuration for better plots
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

    # Get predictions
    y_pred_train = model(x)
    y_pred_test = model(valid_x)

    # Convert to numpy arrays
    ycpu = y.cpu().detach().numpy()
    ytestcpu = valid_y.cpu().detach().numpy()
    yptraincpu = y_pred_train.cpu().detach().numpy()
    yptestcpu = y_pred_test.cpu().detach().numpy()

    # Calculate statistics
    ystd = np.zeros(16)
    yteststd = np.zeros(16)
    ypstd = np.zeros(16)
    ypteststd = np.zeros(16)
    yerr = np.zeros(16)
    kappa_mean = np.zeros(16)

    for i in range(16):
        ystd[i] = np.std(np.exp(ycpu[:, i] * k_std[i] + k_mean[i]))
        yteststd[i] = np.std(np.exp(ytestcpu[:, i] * k_std[i] + k_mean[i]))
        ypstd[i] = np.std(np.exp(yptraincpu[:, i] * k_std[i] + k_mean[i]))
        ypteststd[i] = np.std(np.exp(yptestcpu[:, i] * k_std[i] + k_mean[i]))
        yerr[i] = np.std(np.exp(ytestcpu[:, i] * k_std[i] + k_mean[i]) - 
                         np.exp(yptestcpu[:, i] * k_std[i] + k_mean[i]))
        kappa_mean[i] = np.mean(np.exp(ycpu[:, i] * k_std[i] + k_mean[i]))

    # Create a 2x2 grid figure with better organization
    fig = plt.figure(figsize=(15, 12))
    
    # Setup grid with custom sizes
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])
    
    # Panel 1: Box plots of network output differences
    ax1 = fig.add_subplot(gs[0, :2])
    ind = np.arange(0, 16)
    ind_tick = np.arange(1, 17)[::-1]
    
    box_props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray")
    
    for i in range(16):
        box = ax1.boxplot(ytestcpu[:, i] - yptestcpu[:, i], 
                         vert=False, positions=[i], 
                         patch_artist=True,
                         showfliers=False, 
                         whis=(5, 95), 
                         widths=0.5)
        
        # Style the boxes
        for item in ['boxes', 'whiskers', 'medians', 'caps']:
            plt.setp(box[item], color='blue')
        
        plt.setp(box['boxes'], facecolor='lightblue')
    
    ax1.set_xlim([-2.0, 2.0])
    ax1.set_yticks(ind)
    ax1.set_yticklabels(ind_tick)
    ax1.set_title('(a) Neural Network Output Differences', fontweight='bold')
    ax1.set_ylabel('Layer (1=bottom, 16=top)', fontweight='bold')
    ax1.set_xlabel('Difference (true - predicted)', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotation explaining what the plot shows
    ax1.text(0.02, 0.95, 
            "This shows how well the model output matches\nthe true values across vertical layers",
            transform=ax1.transAxes, 
            bbox=box_props,
            fontsize=10)

    # Panel 2: Box plots of shape function differences
    ax2 = fig.add_subplot(gs[0, 2:])
    
    for i in range(16):
        box = ax2.boxplot(np.exp(ytestcpu[:, i] * k_std[i] + k_mean[i]), 
                         vert=False, 
                         positions=[i], 
                         patch_artist=True,
                         showfliers=False, 
                         whis=(5, 95), 
                         widths=0.5)
        
        # Style the boxes
        for item in ['boxes', 'whiskers', 'medians', 'caps']:
            plt.setp(box[item], color='green')
        
        plt.setp(box['boxes'], facecolor='lightgreen')
    
    ax2.set_yticks(ind)
    ax2.set_yticklabels([])
    ax2.set_title('(b) Shape Function g(σ) Distribution', fontweight='bold')
    ax2.set_xlabel('g(σ) (normalized diffusivity)', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotation explaining what the plot shows
    ax2.text(0.02, 0.95, 
            "Distribution of diffusivity values\nacross vertical layers",
            transform=ax2.transAxes, 
            bbox=box_props,
            fontsize=10)

    # Panel 3: Sample shape functions
    ax3 = fig.add_subplot(gs[1, :2])
    
    # Sample 5 random validation cases
    sample_indices = np.random.choice(len(valid_x), 5, replace=False)
    sigma = np.linspace(0, 1, 16)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, idx in enumerate(sample_indices):
        # Get predictions and true values
        pred_vals = np.exp(yptestcpu[idx] * k_std + k_mean)
        true_vals = np.exp(ytestcpu[idx] * k_std + k_mean)
        
        # Normalize
        pred_norm = pred_vals / np.max(pred_vals)
        true_norm = true_vals / np.max(true_vals)
        
        # Plot with inverted y-axis to match ocean convention
        ax3.plot(true_norm, sigma, 'o-', color=colors[i], label=f'True {i+1}', alpha=0.7)
        ax3.plot(pred_norm, sigma, '--', color=colors[i], label=f'Pred {i+1}', alpha=0.7)
    
    # Customize the plot
    ax3.invert_yaxis()
    ax3.set_xlabel('g(σ) (normalized)', fontweight='bold')
    ax3.set_ylabel('σ (normalized depth)', fontweight='bold')
    ax3.set_title('(c) Sample Shape Function Predictions', fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Create a custom legend with better organization
    handles, labels = ax3.get_legend_handles_labels()
    true_handles = handles[::2]
    pred_handles = handles[1::2]
    true_labels = labels[::2]
    pred_labels = labels[1::2]
    
    legend1 = ax3.legend(true_handles, true_labels, loc='upper left', 
                         title='True Values', fontsize=8)
    ax3.add_artist(legend1)
    ax3.legend(pred_handles, pred_labels, loc='upper right',
              title='Predictions', fontsize=8)
    
    # Add arrows showing surface and bottom
    ax3.annotate('Surface', xy=(0.5, 0.02), xytext=(0.3, 0.1),
                arrowprops=dict(arrowstyle='->'), fontsize=10)
    
    ax3.annotate('Bottom', xy=(0.5, 0.98), xytext=(0.3, 0.9),
                arrowprops=dict(arrowstyle='->'), fontsize=10)

    # Panel 4: Error histogram
    ax4 = fig.add_subplot(gs[1, 2:])
    
    # Create error histograms for each layer
    error_by_layer = np.zeros((16, len(valid_x)))
    
    for i in range(16):
        error_by_layer[i] = np.exp(ytestcpu[:, i] * k_std[i] + k_mean[i]) - \
                            np.exp(yptestcpu[:, i] * k_std[i] + k_mean[i])
    
    # Plot errors by layer as a heatmap
    im = ax4.imshow(error_by_layer, aspect='auto', cmap='RdBu_r',
                   vmin=-0.2, vmax=0.2, extent=[0, len(valid_x), 16.5, 0.5])
    
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Error (true - predicted)', fontweight='bold')
    
    ax4.set_title('(d) Error Distribution by Layer', fontweight='bold')
    ax4.set_xlabel('Validation Sample Index', fontweight='bold')
    ax4.set_ylabel('Layer (1=bottom, 16=top)', fontweight='bold')
    
    # Add informative text about errors
    mean_abs_error = np.mean(np.abs(error_by_layer))
    max_abs_error = np.max(np.abs(error_by_layer))
    
    ax4.text(0.02, 0.05, 
            f"Mean absolute error: {mean_abs_error:.4f}\nMax absolute error: {max_abs_error:.4f}",
            transform=ax4.transAxes, 
            bbox=box_props,
            fontsize=10)
    
    # Adjust layout and add a main title
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('Evaluation of Neural Network Vertical Mixing Model', 
                fontsize=16, fontweight='bold')
    
    return fig

def plot_hyperparameter_comparison(scores, hid_array, lays, loss_array):
    """
    Create an enhanced visualization of hyperparameter comparison.
    
    Parameters:
        scores: Performance scores for different configurations
        hid_array: Array of hidden layer sizes
        lays: Array of layer counts
        loss_array: Array with training and validation losses
    """
    plt.figure(figsize=(15, 8))
    
    # First subplot: Hyperparameter comparison
    plt.subplot(1, 2, 1)
    
    markers = ['o', 's']
    colors = ['#1f77b4', '#ff7f0e']
    
    for i, lay in enumerate(lays):
        plt.plot(hid_array, scores[:, i], 
                marker=markers[i], 
                color=colors[i],
                markersize=10,
                linewidth=2,
                label=f'{lay} layer{"s" if lay > 1 else ""}')
    
    # Add horizontal line for typical performance threshold
    threshold = 0.7  # Example value
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
    plt.text(hid_array[0], threshold+0.02, f'Typical performance threshold ({threshold})',
             fontsize=10, color='red')
    
    # Formatting
    plt.xlabel('Hidden Nodes per Layer', fontsize=14, fontweight='bold')
    plt.ylabel('Linear Correlation Coefficient', fontsize=14, fontweight='bold')
    plt.title('Performance by Network Architecture', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)
    
    # Optional: Annotate the best configuration
    best_score_idx = np.unravel_index(np.argmax(scores), scores.shape)
    best_nodes = hid_array[best_score_idx[0]]
    best_layers = lays[best_score_idx[1]]
    best_score = scores[best_score_idx]
    
    plt.annotate(f'Best: {best_layers} layers, {best_nodes} nodes\nScore: {best_score:.3f}',
                xy=(best_nodes, best_score), 
                xytext=(best_nodes-15, best_score-0.1),
                arrowprops=dict(arrowstyle='->'),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Second subplot: Loss curves for best configuration
    plt.subplot(1, 2, 2)
    
    if loss_array is not None:
        plt.plot(loss_array[:, 0], loss_array[:, 1], 
                color='blue', linewidth=2, label='Training Loss')
        plt.plot(loss_array[:, 0], loss_array[:, 2], 
                color='red', linewidth=2, label='Validation Loss')
        
        # Add a line showing the gap
        plt.fill_between(loss_array[:, 0], 
                        loss_array[:, 1], 
                        loss_array[:, 2],
                        alpha=0.2, color='gray', 
                        label='Generalization Gap')
        
        plt.xlabel('Epochs', fontsize=14, fontweight='bold')
        plt.ylabel('Loss', fontsize=14, fontweight='bold')
        plt.title(f'Loss for {best_layers} layers, {best_nodes} nodes', 
                 fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12)
    
    plt.tight_layout()
    return plt.gcf()

def plot_cluster_analysis(data, cluster_assignments, n_clusters):
    """
    Visualize the results of clustering analysis.
    
    Parameters:
        data: Input data with features and shape functions
        cluster_assignments: Array of cluster assignments
        n_clusters: Number of clusters
    """
    # Setup
    f = data[:, 0]      # Coriolis parameter
    B0 = data[:, 1]     # Surface Buoyancy Flux
    ustar = data[:, 2]  # Surface Friction Velocity
    h = data[:, 3]      # Boundary Layer Depth
    
    # Figure for cluster visualization
    plt.figure(figsize=(16, 12))
    
    # Define colors for each cluster
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Panel 1: Clusters in physical parameter space
    plt.subplot(2, 2, 1)
    for i in range(n_clusters):
        mask = cluster_assignments == i
        plt.scatter(f[mask], B0[mask], 
                   c=colors[i % len(colors)], 
                   label=f'Cluster {i+1}', 
                   alpha=0.7, s=30, edgecolor='none')
    
    plt.title('Clusters in Physical Parameter Space', fontsize=16, fontweight='bold')
    plt.xlabel('Coriolis Parameter (f) [s$^{-1}$]', fontsize=14)
    plt.ylabel('Surface Buoyancy Flux (B0) [m$^2$/s$^3$]', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)
    
    # Panel 2: ustar vs h colored by cluster
    plt.subplot(2, 2, 2)
    for i in range(n_clusters):
        mask = cluster_assignments == i
        plt.scatter(ustar[mask], h[mask], 
                   c=colors[i % len(colors)], 
                   label=f'Cluster {i+1}', 
                   alpha=0.7, s=30, edgecolor='none')
    
    plt.title('Clusters by Friction Velocity and Boundary Layer Depth', 
             fontsize=16, fontweight='bold')
    plt.xlabel('Surface Friction Velocity (ustar) [m/s]', fontsize=14)
    plt.ylabel('Boundary Layer Depth (h) [m]', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)
    
    # Panel 3: Mean shape functions by cluster
    plt.subplot(2, 2, 3)
    
    # Define vertical coordinate
    sigma = np.linspace(0, 1, 16)
    
    for i in range(n_clusters):
        mask = cluster_assignments == i
        shape_functions = data[mask, 4:20]  # Shape function values
        
        # Calculate mean shape function
        mean_shape = np.mean(shape_functions, axis=0)
        std_shape = np.std(shape_functions, axis=0)
        
        # Normalize mean
        max_val = np.max(mean_shape)
        if max_val > 0:  # Avoid division by zero
            mean_shape_norm = mean_shape / max_val
            std_shape_norm = std_shape / max_val
            
            # Plot mean with error band
            plt.plot(mean_shape_norm, sigma, 
                    color=colors[i % len(colors)], 
                    linewidth=2, 
                    label=f'Cluster {i+1} (n={np.sum(mask)})')
            
            plt.fill_betweenx(sigma, 
                             mean_shape_norm - std_shape_norm, 
                             mean_shape_norm + std_shape_norm,
                             color=colors[i % len(colors)], 
                             alpha=0.2)
    
    # Universal shape function
    z = np.linspace(0, 1, 100)
    z1 = z * (1-z)**2
    z1 = z1 / np.max(z1)
    plt.plot(z1, z, color='black', linestyle='--', 
            linewidth=2, label='Universal')
    
    plt.gca().invert_yaxis()  # Invert to match oceanographic convention
    plt.title('Mean Shape Functions by Cluster', fontsize=16, fontweight='bold')
    plt.xlabel('g(σ) (normalized)', fontsize=14)
    plt.ylabel('σ (depth)', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='best')
    
    # Panel 4: Cluster model performance improvement
    plt.subplot(2, 2, 4)
    
    # This would need actual performance metrics to complete
    # For now, placeholder for conceptual design
    
    methods = ['Universal', 'Non-clustered\nNN', 'Clustered\nNN']
    example_perf = [0.5, 0.7, 0.8]  # Placeholder performance metrics
    
    bars = plt.bar(methods, example_perf, color=['gray', 'lightblue', 'green'])
    
    # Add percentage improvements
    baseline = example_perf[0]
    for i, bar in enumerate(bars[1:], 1):
        improvement = (example_perf[i] - baseline) / baseline * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'+{improvement:.1f}%',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Metric (e.g., R²)', fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle('Cluster Analysis of Ocean Mixing Profiles', 
                fontsize=18, fontweight='bold')
    
    return plt.gcf()