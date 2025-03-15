import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

# Not currently used in the workflow, but keeping for potential future use
def plot_silhouette_scores(silhouette_scores):
    """
    Plot silhouette scores for different numbers of clusters.
    
    Parameters:
        silhouette_scores: List of tuples (n_clusters, score)
    """
    if not silhouette_scores:
        print("No silhouette scores to plot")
        return None
        
    plt.figure(figsize=(10, 6))
    
    # Sort by cluster number
    silhouette_scores = sorted(silhouette_scores, key=lambda x: x[0])
    
    # Plot scores
    plt.plot([s[0] for s in silhouette_scores], [s[1] for s in silhouette_scores], 
             'o-', linewidth=2, markersize=10, color='#1f77b4')
    
    # Add best score marker
    best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
    plt.scatter(best_k, best_score, s=200, c='red', zorder=3, alpha=0.6)
    
    # Add annotation with some protection against out-of-bounds positioning
    y_range = max([s[1] for s in silhouette_scores]) - min([s[1] for s in silhouette_scores])
    if y_range == 0:
        y_range = 0.1  # Prevent division by zero
        
    plt.annotate(f'Best: k={best_k}, score={best_score:.4f}',
                xy=(best_k, best_score),
                xytext=(best_k+0.5, best_score-0.05*y_range),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.title('Optimal Number of Clusters Based on Silhouette Score', fontsize=16, fontweight='bold')
    
    # Set sensible x-ticks
    min_k = min([s[0] for s in silhouette_scores])
    max_k = max([s[0] for s in silhouette_scores])
    plt.xticks(range(min_k, max_k+1))
    
    plt.tight_layout()
    
    return plt.gcf()

def plot_cluster_distributions(features, cluster_assignments, n_clusters, feature_names=None):
    """
    Plot distribution of each feature across different clusters.
    
    Parameters:
        features: Input features used for clustering
        cluster_assignments: Cluster labels
        n_clusters: Number of clusters
        feature_names: Names of the features
    """
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(features.shape[1])]
    
    # Set up colors for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Create a figure with one subplot per feature
    n_features = features.shape[1]
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 3*n_features))
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[i] if n_features > 1 else axes
        for j in range(n_clusters):
            mask = cluster_assignments == j
            sns.kdeplot(features[mask, i], ax=ax, color=colors[j], 
                       label=f'Cluster {j}', fill=True, alpha=0.3)
        
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Distribution of {feature_name} by Cluster', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Only add legend to first subplot
        if i == 0:
            ax.legend(fontsize=10, title="Clusters")
    
    plt.tight_layout()
    return fig

def plot_shape_functions_by_cluster(data_load_main, cluster_assignments, n_clusters):
    """
    Plot average shape functions for each cluster.
    
    Parameters:
        data_load_main: Raw data containing shape functions
        cluster_assignments: Cluster labels
        n_clusters: Number of clusters
    """
    # Set up figure
    plt.figure(figsize=(10, 8))
    
    # Set up colors for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Get shape functions and sigma levels
    shape_functions = data_load_main[:, 4:20]  # Shape functions in columns 4-19
    sigma_levels = np.linspace(0, 1, 16)  # 16 sigma levels
    
    # Calculate mean shape function for each cluster
    for i in range(n_clusters):
        mask = cluster_assignments == i
        cluster_shape_functions = shape_functions[mask]
        
        # Calculate number of samples in this cluster
        n_samples = np.sum(mask)
        
        # Normalize each shape function individually
        normalized_shapes = np.zeros_like(cluster_shape_functions)
        for j in range(len(cluster_shape_functions)):
            max_val = np.max(cluster_shape_functions[j])
            if max_val > 0:
                normalized_shapes[j] = cluster_shape_functions[j] / max_val
        
        # Calculate mean and std of normalized shape functions
        mean_shape = np.mean(normalized_shapes, axis=0)
        std_shape = np.std(normalized_shapes, axis=0)
        
        # Plot mean with std band
        plt.plot(mean_shape, sigma_levels, 'o-', color=colors[i], 
                linewidth=2, label=f'Cluster {i} (n={n_samples})')
        plt.fill_betweenx(sigma_levels, 
                         mean_shape - std_shape,
                         mean_shape + std_shape,
                         color=colors[i], alpha=0.2)
    
    # Add the universal shape function for comparison
    z = np.linspace(0, 1, 100)
    z1 = z * (1-z)**2
    z1 = z1 / np.max(z1)
    # Note: z1[::-1] reverses the array to match the original paper's orientation
    plt.plot(z1[::-1], z, 'k--', linewidth=2, label='Universal')
    
    # Customize plot
    # CHANGED: Removed inversion of y-axis to have σ=0 at top and σ=1 at bottom
    # plt.gca().invert_yaxis()  # Invert to match oceanographic convention
    plt.xlabel('g(σ) (Normalized Diffusivity)', fontsize=12)
    plt.ylabel('σ (Normalized Depth)', fontsize=12)
    plt.title('Mean Shape Functions by Cluster', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    
    # Add annotations for surface and bottom
    plt.annotate('Surface (σ=0)', xy=(0.5, 0.02), xytext=(0.6, 0.05),
                arrowprops=dict(arrowstyle='->', color='black'), color='black')
    plt.annotate('Bottom of Mixed Layer (σ=1)', xy=(0.5, 0.98), xytext=(0.6, 0.95),
                arrowprops=dict(arrowstyle='->', color='black'), color='black')
    
    return plt.gcf()

# Removed plot_cluster_2d_projections function as it's not used in the current workflow

def plot_cluster_size_distribution(cluster_assignments, n_clusters):
    """
    Create a pie chart showing the distribution of samples across clusters.
    
    Parameters:
        cluster_assignments: Cluster labels
        n_clusters: Number of clusters
    """
    # Count samples in each cluster
    cluster_counts = np.bincount(cluster_assignments)
    
    # Set up figure
    plt.figure(figsize=(8, 8))
    
    # Create pie chart
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    plt.pie(cluster_counts, labels=[f'Cluster {i}\n({count} samples)' for i, count in enumerate(cluster_counts)],
           autopct='%1.1f%%', colors=colors, startangle=90)
    
    plt.title('Cluster Size Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return plt.gcf()

def plot_cluster_centers(kmeans, feature_names, scaler=None):
    """
    Plot a bar chart of cluster centers for each feature.
    
    Parameters:
        kmeans: Trained KMeans model
        feature_names: Names of the features
        scaler: StandardScaler used to scale features (optional)
    """
    # Get cluster centers
    centers = kmeans.cluster_centers_
    
    # If scaler is provided, transform centers back to original scale
    if scaler is not None:
        centers = scaler.inverse_transform(centers)
    
    # Set up figure
    n_clusters, n_features = centers.shape
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 3*n_features))
    
    # Colors for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Plot each feature
    for i in range(n_features):
        ax = axes[i] if n_features > 1 else axes
        bars = ax.bar(range(n_clusters), centers[:, i], color=colors)
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(centers[:, i]),
                   f'{centers[j, i]:.4f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_xticks(range(n_clusters))
        ax.set_xticklabels([f'Cluster {j}' for j in range(n_clusters)])
        ax.set_ylabel(feature_names[i], fontsize=12)
        ax.set_title(f'Cluster Centers: {feature_names[i]}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def plot_model_performance_comparison(baseline_node_losses, cluster_node_losses):
    """
    Create a bar chart comparing baseline and cluster model performance for each node.
    
    Parameters:
        baseline_node_losses: List of losses for baseline model by node
        cluster_node_losses: List of losses for cluster model by node
    """
    # Calculate improvement percentage
    improvements = [(baseline - cluster) / baseline * 100 
                   for baseline, cluster in zip(baseline_node_losses, cluster_node_losses)]
    
    # Set up figure
    plt.figure(figsize=(10, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(range(16), improvements, 
                  color=['green' if imp > 0 else 'red' for imp in improvements])
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width + 0.5 if width > 0 else width - 5
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                f'{improvements[i]:.1f}%', 
                va='center', fontsize=9,
                color='black' if width > 0 else 'white')
    
    # Customize plot
    plt.yticks(range(16), [f'Layer {16-i}' for i in range(16)])  # Reverse order
    plt.xlabel('Improvement (%)', fontsize=12)
    plt.title('Performance Improvement by Layer (Cluster vs Baseline)', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Add average improvement
    avg_improvement = np.mean(improvements)
    plt.axvline(avg_improvement, color='blue', linestyle='--', linewidth=1.5)
    plt.text(avg_improvement + 0.5, 0, f'Avg: {avg_improvement:.1f}%', 
            va='bottom', fontsize=10, color='blue')
    
    plt.tight_layout()
    return plt.gcf()

def plot_sample_predictions(valid_indices, valid_clusters, baseline_preds, cluster_preds, valid_y):
    """
    Plot sample predictions from baseline and cluster models compared to ground truth.
    
    Parameters:
        valid_indices: Indices of samples to plot
        valid_clusters: Cluster assignments for validation data
        baseline_preds: Predictions from baseline model
        cluster_preds: Predictions from cluster models
        valid_y: Ground truth values
    """
    # Number of samples to plot
    n_samples = len(valid_indices)
    
    # Set up figure
    fig, axes = plt.subplots(1, n_samples, figsize=(5*n_samples, 5))
    
    # If only one sample, wrap axes in a list
    if n_samples == 1:
        axes = [axes]
    
    # Plot each sample
    sigma_levels = np.linspace(0, 1, 16)  # 16 sigma levels
    
    for i, idx in enumerate(valid_indices):
        ax = axes[i]
        
        # Get predictions and ground truth
        baseline_pred = baseline_preds[idx]
        cluster_pred = cluster_preds[idx]
        true_values = valid_y[idx]
        cluster_id = valid_clusters[idx]
        
        # Normalize for better visualization
        baseline_norm = baseline_pred / np.max(baseline_pred)
        cluster_norm = cluster_pred / np.max(cluster_pred)
        true_norm = true_values / np.max(true_values)
        
        # Plot
        ax.plot(true_norm, sigma_levels, 'ko-', linewidth=2, label='Ground Truth')
        ax.plot(baseline_norm, sigma_levels, 'b--', linewidth=2, label='Baseline')
        ax.plot(cluster_norm, sigma_levels, 'g-.', linewidth=2, label='Cluster')
        
        # Calculate error metrics
        baseline_err = np.mean(np.abs(baseline_norm - true_norm))
        cluster_err = np.mean(np.abs(cluster_norm - true_norm))
        improvement = (baseline_err - cluster_err) / baseline_err * 100
        
        # Customize plot
        # CHANGED: Removed inversion of y-axis to match the shape_functions plot
        # ax.invert_yaxis()  # Invert to match oceanographic convention
        ax.set_xlabel('g(σ) (Normalized)', fontsize=12)
        if i == 0:
            ax.set_ylabel('σ (Normalized Depth)', fontsize=12)
        ax.set_title(f'Sample from Cluster {cluster_id}\nImprovement: {improvement:.1f}%', 
                    fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add annotations
        ax.text(0.05, 0.05, 'Surface (σ=0)', transform=ax.transAxes, fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8))
        ax.text(0.05, 0.95, 'Bottom (σ=1)', transform=ax.transAxes, fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8))
        
        # Only add legend to first plot
        if i == 0:
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_error_distributions(baseline_preds, cluster_preds, valid_y):
    """
    Plot histograms of prediction errors for baseline and cluster models.
    
    Parameters:
        baseline_preds: Predictions from baseline model
        cluster_preds: Predictions from cluster models
        valid_y: Ground truth values
    """
    # Calculate errors
    baseline_errors = (baseline_preds - valid_y).flatten()
    cluster_errors = (cluster_preds - valid_y).flatten()
    
    # Set up figure
    plt.figure(figsize=(10, 6))
    
    # Create histograms
    plt.hist(baseline_errors, bins=50, alpha=0.5, color='blue', label='Baseline Model')
    plt.hist(cluster_errors, bins=50, alpha=0.5, color='green', label='Cluster Models')
    
    # Add mean lines
    plt.axvline(np.mean(baseline_errors), color='blue', linestyle='--', linewidth=2)
    plt.axvline(np.mean(cluster_errors), color='green', linestyle='--', linewidth=2)
    
    # Customize plot
    plt.xlabel('Prediction Error', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Error Distribution: Baseline vs Cluster Models', fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)
    
    # Add summary statistics
    plt.text(0.02, 0.95, 
            f'Baseline Mean: {np.mean(baseline_errors):.4f}\nCluster Mean: {np.mean(cluster_errors):.4f}',
            transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return plt.gcf()

def plot_overall_performance_comparison(baseline_loss, cluster_loss):
    """
    Create a bar chart comparing overall performance of baseline and cluster models.
    
    Parameters:
        baseline_loss: Overall loss for baseline model
        cluster_loss: Overall loss for cluster models
    """
    # Calculate improvement
    improvement = (baseline_loss - cluster_loss) / baseline_loss * 100
    
    # Set up figure
    plt.figure(figsize=(8, 6))
    
    # Create bar chart
    bars = plt.bar(['Baseline Model', 'Cluster Models'], 
                 [baseline_loss, cluster_loss],
                 color=['blue', 'green'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12)
    
    # Add improvement arrow and label
    plt.annotate(f'{improvement:.2f}% improvement', 
                xy=(1, cluster_loss),
                xytext=(1, (baseline_loss + cluster_loss) / 2),
                arrowprops=dict(arrowstyle='->'),
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Customize plot
    plt.ylabel('Mean Absolute Error', fontsize=14)
    plt.title('Overall Model Performance', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return plt.gcf()