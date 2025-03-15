"""
Clustered model architectures for ocean mixing parameterization.

This module provides classes for different clustering approaches:
1. InputClusteredModel: Clusters based on input features
2. OutputClusteredModel: Clusters based on output shape functions
"""

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import copy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

class BaseClusteredModel:
    """Base class for clustered ocean mixing models"""
    
    def __init__(self, n_clusters=4, random_state=42):
        """
        Initialize the clustered model
        
        Args:
            n_clusters: Number of clusters to use
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_models = {}
        self.clusterer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, X, y, valid_X=None, valid_y=None, stats=None):
        """
        Cluster data and train models for each cluster
        
        Args:
            X: Input features (numpy array or tensor)
            y: Target values (numpy array or tensor)
            valid_X: Validation input features
            valid_y: Validation target values
            stats: Statistics for normalization
        """
        raise NotImplementedError
        
    def predict(self, X):
        """
        Make predictions using appropriate cluster models
        
        Args:
            X: Input features (numpy array or tensor)
            
        Returns:
            Predictions, cluster assignments
        """
        raise NotImplementedError
        
    def evaluate(self, X, y, stats=None):
        """
        Evaluate model performance
        
        Args:
            X: Input features (numpy array or tensor)
            y: Target values (numpy array or tensor)
            stats: Statistics for denormalization
            
        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError
        
    def _train_model(self, model, X, y, valid_X=None, valid_y=None, stats=None, 
                    epochs=3000, lr=1e-3, patience=50):
        """
        Train a PyTorch model with early stopping
        
        Args:
            model: PyTorch model to train
            X: Input features (tensor)
            y: Target values (tensor)
            valid_X: Validation input features
            valid_y: Validation target values
            stats: Statistics for normalization
            epochs: Maximum number of epochs
            lr: Learning rate
            patience: Early stopping patience
            
        Returns:
            Trained model, training losses
        """
        # Ensure data is on the correct device
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        else:
            X = X.to(self.device)
            
        if isinstance(y, np.ndarray):
            y = torch.FloatTensor(y).to(self.device)
        else:
            y = y.to(self.device)
        
        # Validation data
        if valid_X is not None and valid_y is not None:
            if isinstance(valid_X, np.ndarray):
                valid_X = torch.FloatTensor(valid_X).to(self.device)
            else:
                valid_X = valid_X.to(self.device)
                
            if isinstance(valid_y, np.ndarray):
                valid_y = torch.FloatTensor(valid_y).to(self.device)
            else:
                valid_y = valid_y.to(self.device)
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.L1Loss(reduction='mean')
        losses = []
        
        # Get normalization statistics if needed
        if stats is not None:
            k_mean, k_std = stats['k_mean'], stats['k_std']
            k_mean_tensor = torch.tensor(k_mean, dtype=torch.float32).to(self.device)
            k_std_tensor = torch.tensor(k_std, dtype=torch.float32).to(self.device)
        
        # Early stopping variables
        best_loss = float('inf')
        best_model = None
        no_improve = 0
        
        # Training loop
        with tqdm(total=epochs, desc="Training") as pbar:
            for epoch in range(epochs):
                # Training step
                model.train()
                optimizer.zero_grad()
                
                # Forward pass
                train_pred = model(X)
                
                # Compute loss
                loss = loss_fn(train_pred, y)
                
                # Backpropagation
                loss.backward()
                optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    if valid_X is not None and valid_y is not None:
                        valid_pred = model(valid_X)
                        
                        # Calculate loss in original space if stats provided
                        if stats is not None:
                            train_loss = torch.mean(torch.abs(
                                torch.exp(train_pred * k_std_tensor + k_mean_tensor) - 
                                torch.exp(y * k_std_tensor + k_mean_tensor)
                            ))
                            
                            valid_loss = torch.mean(torch.abs(
                                torch.exp(valid_pred * k_std_tensor + k_mean_tensor) - 
                                torch.exp(valid_y * k_std_tensor + k_mean_tensor)
                            ))
                        else:
                            train_loss = torch.mean(torch.abs(train_pred - y))
                            valid_loss = torch.mean(torch.abs(valid_pred - valid_y))
                    else:
                        # Use training loss if no validation data
                        train_loss = torch.mean(torch.abs(train_pred - y))
                        valid_loss = train_loss
                
                # Store losses
                losses.append((epoch, train_loss.item(), valid_loss.item()))
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'train_loss': f"{train_loss.item():.4f}", 
                    'valid_loss': f"{valid_loss.item():.4f}",
                    'patience': no_improve
                })
                
                # Early stopping
                if valid_loss.item() < best_loss:
                    best_loss = valid_loss.item()
                    best_model = copy.deepcopy(model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break
        
        # Load the best model
        if best_model is not None:
            model.load_state_dict(best_model)
        
        return model, np.array(losses)


class InputClusteredModel(BaseClusteredModel):
    """Clustered model based on input features"""
    
    def __init__(self, model_class, model_params, n_clusters=4, random_state=42):
        """
        Initialize the input clustered model
        
        Args:
            model_class: PyTorch model class
            model_params: Parameters for model initialization
            n_clusters: Number of clusters
            random_state: Random seed
        """
        super().__init__(n_clusters, random_state)
        self.model_class = model_class
        self.model_params = model_params
        self.scaler = StandardScaler()
        self.cluster_assignments = None
        self.stats = None
        
    def fit(self, X, y, valid_X=None, valid_y=None, stats=None):
        """
        Cluster based on input features and train models for each cluster
        
        Args:
            X: Input features (numpy array or tensor)
            y: Target values (numpy array or tensor)
            valid_X: Validation features
            valid_y: Validation targets
            stats: Dictionary with k_mean and k_std for denormalization
            
        Returns:
            self (for method chaining)
        """
        # Convert inputs to numpy if needed
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
            
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy()
        else:
            y_np = y
        
        # Store stats for later use
        self.stats = stats
        
        # Standardize features for clustering
        scaled_features = self.scaler.fit_transform(X_np)
        
        # Perform clustering
        print(f"Clustering input features into {self.n_clusters} clusters...")
        self.clusterer = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state,
            n_init=10
        )
        self.cluster_assignments = self.clusterer.fit_predict(scaled_features)
        
        # Print cluster distribution
        for i in range(self.n_clusters):
            count = np.sum(self.cluster_assignments == i)
            print(f"Cluster {i}: {count} samples ({count/len(self.cluster_assignments)*100:.2f}%)")
        
        # Train a model for each cluster
        for cluster_id in range(self.n_clusters):
            print(f"\nTraining model for Cluster {cluster_id}")
            
            # Get data for this cluster
            cluster_mask = self.cluster_assignments == cluster_id
            X_cluster = X_np[cluster_mask]
            y_cluster = y_np[cluster_mask]
            
            # Skip if cluster is too small
            if len(X_cluster) < 50:
                print(f"Cluster {cluster_id} has fewer than 50 samples, skipping")
                continue
                
            # Convert to tensors
            X_cluster_tensor = torch.FloatTensor(X_cluster).to(self.device)
            y_cluster_tensor = torch.FloatTensor(y_cluster).to(self.device)
            
            # Initialize model
            model = self.model_class(**self.model_params).to(self.device)
            
            # Train model
            trained_model, losses = self._train_model(
                model=model,
                X=X_cluster_tensor,
                y=y_cluster_tensor,
                valid_X=valid_X,
                valid_y=valid_y,
                stats=stats
            )
            
            # Store trained model
            self.cluster_models[cluster_id] = {
                'model': trained_model,
                'losses': losses
            }
        
        print("Completed training all cluster models")
        return self
    
    def predict(self, X):
        """
        Make predictions using cluster-specific models
        
        Args:
            X: Input features (numpy or tensor)
            
        Returns:
            Predictions, cluster assignments
        """
        # Convert input to numpy if needed
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
        
        # Standardize features
        scaled_features = self.scaler.transform(X_np)
        
        # Assign to clusters
        cluster_assignments = self.clusterer.predict(scaled_features)
        
        # Make predictions
        all_preds = []
        for i in range(len(X_np)):
            cluster_id = cluster_assignments[i]
            
            # If no model for this cluster, use the first available
            if cluster_id not in self.cluster_models:
                cluster_id = list(self.cluster_models.keys())[0]
                
            model = self.cluster_models[cluster_id]['model']
            
            # Get input as tensor
            X_i = torch.FloatTensor(X_np[i:i+1]).to(self.device)
            
            # Predict
            with torch.no_grad():
                pred = model(X_i)
                all_preds.append(pred)
        
        # Combine predictions
        preds = torch.cat(all_preds, dim=0)
        
        return preds, cluster_assignments
    
    def evaluate(self, X, y, stats=None):
        """
        Evaluate model performance
        
        Args:
            X: Input features
            y: Target values
            stats: Statistics for denormalization
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        preds, cluster_assignments = self.predict(X)
        
        # Convert to tensors if needed
        if not isinstance(preds, torch.Tensor):
            preds = torch.FloatTensor(preds).to(self.device)
        
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y).to(self.device)
        
        # Denormalize if stats provided
        if stats is not None:
            k_mean = stats['k_mean']
            k_std = stats['k_std']
            k_mean_tensor = torch.tensor(k_mean, dtype=torch.float32).to(self.device)
            k_std_tensor = torch.tensor(k_std, dtype=torch.float32).to(self.device)
            
            preds_orig = torch.exp(preds * k_std_tensor + k_mean_tensor)
            y_orig = torch.exp(y * k_std_tensor + k_mean_tensor)
        else:
            preds_orig = preds
            y_orig = y
        
        # Calculate overall MAE
        mae = torch.mean(torch.abs(preds_orig - y_orig)).item()
        
        # Calculate node-wise MAE
        node_mae = []
        for i in range(y_orig.shape[1]):
            node_mae.append(torch.mean(torch.abs(preds_orig[:, i] - y_orig[:, i])).item())
        
        # Calculate cluster-wise MAE
        cluster_mae = {}
        for cluster in np.unique(cluster_assignments):
            mask = cluster_assignments == cluster
            if np.sum(mask) > 0:
                cluster_mae[int(cluster)] = torch.mean(torch.abs(
                    preds_orig[mask] - y_orig[mask]
                )).item()
        
        return {
            'mae': mae,
            'node_mae': node_mae,
            'cluster_mae': cluster_mae
        }


class OutputClusteredModel(BaseClusteredModel):
    """Clustered model based on output shape functions"""
    
    def __init__(self, model_class, model_params, n_clusters=4, random_state=42, 
                 use_pca=True, n_components=5):
        """
        Initialize the output clustered model
        
        Args:
            model_class: PyTorch model class
            model_params: Parameters for model initialization
            n_clusters: Number of clusters
            random_state: Random seed
            use_pca: Whether to use PCA for dimension reduction
            n_components: Number of PCA components if used
        """
        super().__init__(n_clusters, random_state)
        self.model_class = model_class
        self.model_params = model_params
        self.use_pca = use_pca
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = None
        self.classifier = None
        self.shape_cluster_assignments = None
        self.stats = None
        
    def fit(self, X, y, valid_X=None, valid_y=None, stats=None):
        """
        Cluster based on output shape functions and train models
        
        Args:
            X: Input features (numpy array or tensor)
            y: Target shape functions (numpy array or tensor)
            valid_X: Validation features
            valid_y: Validation targets
            stats: Dictionary with k_mean and k_std for denormalization
            
        Returns:
            self (for method chaining)
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
            
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy()
        else:
            y_np = y
        
        # Store stats for later use
        self.stats = stats
        
        # Normalize shape functions for clustering
        print("Normalizing shape functions...")
        normalized_shapes = np.zeros_like(y_np)
        for i in range(y_np.shape[0]):
            max_val = np.max(y_np[i])
            if max_val > 0:
                normalized_shapes[i] = y_np[i] / max_val
        
        # Apply dimensionality reduction if requested
        if self.use_pca:
            print(f"Applying PCA to reduce dimensions to {self.n_components}...")
            self.pca = PCA(n_components=self.n_components)
            reduced_shapes = self.pca.fit_transform(normalized_shapes)
            clustering_input = reduced_shapes
            
            # Print explained variance
            explained_variance = self.pca.explained_variance_ratio_
            print(f"Explained variance: {np.sum(explained_variance):.2%}")
        else:
            clustering_input = normalized_shapes
        
        # Perform clustering on shape functions
        print(f"Clustering shape functions into {self.n_clusters} clusters...")
        self.clusterer = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state,
            n_init=10
        )
        self.shape_cluster_assignments = self.clusterer.fit_predict(clustering_input)
        
        # Print cluster distribution
        for i in range(self.n_clusters):
            count = np.sum(self.shape_cluster_assignments == i)
            print(f"Cluster {i}: {count} samples ({count/len(self.shape_cluster_assignments)*100:.2f}%)")
        
        # Train a classifier to predict shape clusters from input features
        print("Training a classifier to predict shape clusters from inputs...")
        self.classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state
        )
        self.classifier.fit(X_np, self.shape_cluster_assignments)
        
        # Print feature importance
        feature_importance = self.classifier.feature_importances_
        print("Feature importance for predicting shape clusters:")
        for i, importance in enumerate(feature_importance):
            print(f"Feature {i}: {importance:.4f}")
        
        # Train a model for each shape cluster
        for cluster_id in range(self.n_clusters):
            print(f"\nTraining model for Shape Cluster {cluster_id}")
            
            # Get data for this cluster
            cluster_mask = self.shape_cluster_assignments == cluster_id
            X_cluster = X_np[cluster_mask]
            y_cluster = y_np[cluster_mask]
            
            # Skip if cluster is too small
            if len(X_cluster) < 50:
                print(f"Cluster {cluster_id} has fewer than 50 samples, skipping")
                continue
                
            # Convert to tensors
            X_cluster_tensor = torch.FloatTensor(X_cluster).to(self.device)
            y_cluster_tensor = torch.FloatTensor(y_cluster).to(self.device)
            
            # Initialize model
            model = self.model_class(**self.model_params).to(self.device)
            
            # Train model
            trained_model, losses = self._train_model(
                model=model,
                X=X_cluster_tensor,
                y=y_cluster_tensor,
                valid_X=valid_X,
                valid_y=valid_y,
                stats=stats
            )
            
            # Store trained model
            self.cluster_models[cluster_id] = {
                'model': trained_model,
                'losses': losses
            }
        
        print("Completed training all shape cluster models")
        return self
    
    def predict(self, X):
        """
        Make predictions using shape-cluster-specific models
        
        Args:
            X: Input features (numpy or tensor)
            
        Returns:
            Predictions, cluster assignments
        """
        # Convert input to numpy if needed
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
        
        # Predict shape clusters using the classifier
        shape_clusters = self.classifier.predict(X_np)
        
        # Make predictions
        all_preds = []
        for i in range(len(X_np)):
            cluster_id = shape_clusters[i]
            
            # If no model for this cluster, use the first available
            if cluster_id not in self.cluster_models:
                cluster_id = list(self.cluster_models.keys())[0]
                
            model = self.cluster_models[cluster_id]['model']
            
            # Get input as tensor
            X_i = torch.FloatTensor(X_np[i:i+1]).to(self.device)
            
            # Predict
            with torch.no_grad():
                pred = model(X_i)
                all_preds.append(pred)
        
        # Combine predictions
        preds = torch.cat(all_preds, dim=0)
        
        return preds, shape_clusters
    
    def evaluate(self, X, y, stats=None):
        """
        Evaluate model performance
        
        Args:
            X: Input features
            y: Target values
            stats: Statistics for denormalization
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        preds, shape_clusters = self.predict(X)
        
        # Convert to tensors if needed
        if not isinstance(preds, torch.Tensor):
            preds = torch.FloatTensor(preds).to(self.device)
        
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y).to(self.device)
        
        # Denormalize if stats provided
        if stats is not None:
            k_mean = stats['k_mean']
            k_std = stats['k_std']
            k_mean_tensor = torch.tensor(k_mean, dtype=torch.float32).to(self.device)
            k_std_tensor = torch.tensor(k_std, dtype=torch.float32).to(self.device)
            
            preds_orig = torch.exp(preds * k_std_tensor + k_mean_tensor)
            y_orig = torch.exp(y * k_std_tensor + k_mean_tensor)
        else:
            preds_orig = preds
            y_orig = y
        
        # Calculate overall MAE
        mae = torch.mean(torch.abs(preds_orig - y_orig)).item()
        
        # Calculate node-wise MAE
        node_mae = []
        for i in range(y_orig.shape[1]):
            node_mae.append(torch.mean(torch.abs(preds_orig[:, i] - y_orig[:, i])).item())
        
        # Calculate cluster-wise MAE
        cluster_mae = {}
        for cluster in np.unique(shape_clusters):
            mask = shape_clusters == cluster
            if np.sum(mask) > 0:
                cluster_mae[int(cluster)] = torch.mean(torch.abs(
                    preds_orig[mask] - y_orig[mask]
                )).item()
        
        return {
            'mae': mae,
            'node_mae': node_mae,
            'cluster_mae': cluster_mae
        }


class ClusterModelVisualizer:
    """Visualizer for clustered models"""
    
    def __init__(self):
        """Initialize the visualizer"""
        pass
    
    def plot_clusters(self, model, X=None, feature_names=None):
        """
        Plot clusters distribution and centers
        
        Args:
            model: Trained clustered model
            X: Input features to show with clusters (optional)
            feature_names: Names of features (optional)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set default feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(X.shape[1])]
        
        # For InputClusteredModel
        if isinstance(model, InputClusteredModel) and X is not None:
            # Standardize features
            scaled_features = model.scaler.transform(X)
            
            # Get cluster assignments
            cluster_assignments = model.clusterer.predict(scaled_features)
            
            # Get cluster centers
            centers = model.clusterer.cluster_centers_
            centers_original = model.scaler.inverse_transform(centers)
            
            # Plot cluster centers
            plt.figure(figsize=(12, 8))
            for i, feature in enumerate(feature_names):
                plt.subplot(2, 2, i+1)
                plt.bar(range(model.n_clusters), centers_original[:, i])
                plt.xlabel('Cluster')
                plt.ylabel(feature)
                plt.title(f'Cluster Centers: {feature}')
                plt.xticks(range(model.n_clusters), 
                          [f'Cluster {i}' for i in range(model.n_clusters)])
                plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Plot cluster distribution
            plt.figure(figsize=(8, 8))
            counts = np.bincount(cluster_assignments)
            plt.pie(counts, 
                  labels=[f'Cluster {i}\n({count} samples)' for i, count in enumerate(counts)],
                  autopct='%1.1f%%')
            plt.title('Cluster Size Distribution')
            plt.show()
            
        # For OutputClusteredModel
        elif isinstance(model, OutputClusteredModel) and X is not None:
            # Predict shape clusters
            _, shape_clusters = model.predict(X)
            
            # Plot shape cluster distribution
            plt.figure(figsize=(8, 8))
            counts = np.bincount(shape_clusters)
            plt.pie(counts, 
                  labels=[f'Cluster {i}\n({count} samples)' for i, count in enumerate(counts)],
                  autopct='%1.1f%%')
            plt.title('Shape Cluster Size Distribution')
            plt.show()
            
            # If we have input feature names, show importance
            if hasattr(model, 'classifier') and hasattr(model.classifier, 'feature_importances_'):
                plt.figure(figsize=(10, 6))
                importance = model.classifier.feature_importances_
                plt.bar(feature_names, importance)
                plt.xlabel('Feature')
                plt.ylabel('Importance')
                plt.title('Feature Importance for Shape Cluster Prediction')
                plt.xticks(rotation=45)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.show()
    
    def plot_losses(self, model):
        """
        Plot training losses for all cluster models
        
        Args:
            model: Trained clustered model
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        for cluster_id, cluster_model in model.cluster_models.items():
            losses = cluster_model['losses']
            epochs = losses[:, 0]
            valid_losses = losses[:, 2]
            
            plt.plot(epochs, valid_losses, label=f'Cluster {cluster_id}')
            
            # Mark best point
            best_idx = np.argmin(valid_losses)
            plt.scatter(epochs[best_idx], valid_losses[best_idx], marker='o')
        
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Training Losses by Cluster')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()
    
    def plot_performance_comparison(self, baseline_metrics, clustered_metrics):
        """
        Compare performance of baseline and clustered model
        
        Args:
            baseline_metrics: Metrics from baseline model
            clustered_metrics: Metrics from clustered model
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Compare overall MAE
        plt.figure(figsize=(10, 6))
        baseline_mae = baseline_metrics['mae']
        clustered_mae = clustered_metrics['mae']
        improvement = (baseline_mae - clustered_mae) / baseline_mae * 100
        
        plt.bar(['Baseline', 'Clustered'], [baseline_mae, clustered_mae])
        plt.ylabel('Mean Absolute Error')
        plt.title(f'Overall Performance Comparison (Improvement: {improvement:.2f}%)')
        plt.grid(axis='y', alpha=0.3)
        
        # Add improvement arrow
        plt.annotate(f'{improvement:.2f}% improvement', 
                   xy=(1, clustered_mae),
                   xytext=(1, (baseline_mae + clustered_mae) / 2),
                   arrowprops=dict(arrowstyle='->'),
                   ha='center', va='center', fontsize=14)
        
        plt.show()
        
        # Compare node-wise MAE
        plt.figure(figsize=(12, 6))
        baseline_node_mae = baseline_metrics['node_mae']
        clustered_node_mae = clustered_metrics['node_mae']
        
        # Calculate improvement per node
        node_improvements = [(b - c) / b * 100 for b, c in zip(baseline_node_mae, clustered_node_mae)]
        
        plt.bar(range(len(node_improvements)), node_improvements,
              color=['green' if i > 0 else 'red' for i in node_improvements])
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add average line
        avg_improvement = np.mean(node_improvements)
        plt.axhline(y=avg_improvement, color='blue', linestyle='--')
        plt.text(len(node_improvements)-1, avg_improvement, f'Avg: {avg_improvement:.1f}%',
                ha='right', va='bottom')
        
        plt.xlabel('Vertical Level')
        plt.ylabel('Improvement (%)')
        plt.title('Node-wise Performance Improvement')
        plt.xticks(range(len(node_improvements)), 
                 [f'L{i+1}' for i in range(len(node_improvements))])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_shape_functions(self, model, y, normalize=True, labels=None):
        """
        Plot shape functions by cluster
        
        Args:
            model: Trained clustered model (OutputClusteredModel)
            y: Shape functions to plot
            normalize: Whether to normalize the shape functions
            labels: Cluster assignments (if known)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not isinstance(model, OutputClusteredModel):
            print("This method is only for OutputClusteredModel")
            return
        
        # Convert to numpy if needed
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy()
        else:
            y_np = y
        
        # Get shape cluster assignments if not provided
        if labels is None and hasattr(model, 'shape_cluster_assignments'):
            labels = model.shape_cluster_assignments
        
        # Normalize shape functions if requested
        if normalize:
            normalized_shapes = np.zeros_like(y_np)
            for i in range(y_np.shape[0]):
                max_val = np.max(y_np[i])
                if max_val > 0:
                    normalized_shapes[i] = y_np[i] / max_val
            shapes = normalized_shapes
        else:
            shapes = y_np
        
        # Set up plot
        plt.figure(figsize=(10, 8))
        sigma_levels = np.linspace(0, 1, shapes.shape[1])
        
        # Plot mean shape function for each cluster
        for cluster_id in range(model.n_clusters):
            mask = labels == cluster_id
            if np.sum(mask) > 0:
                cluster_shapes = shapes[mask]
                mean_shape = np.mean(cluster_shapes, axis=0)
                std_shape = np.std(cluster_shapes, axis=0)
                
                plt.plot(mean_shape, sigma_levels, 'o-', linewidth=2, 
                       label=f'Cluster {cluster_id} (n={np.sum(mask)})')
                plt.fill_betweenx(sigma_levels, 
                                mean_shape - std_shape,
                                mean_shape + std_shape,
                                alpha=0.2)
        
        # Add universal shape function
        z = np.linspace(0, 1, 100)
        z1 = z * (1-z)**2
        z1 = z1 / np.max(z1)
        # Note: z1[::-1] reverses the array to match the original paper's orientation
        plt.plot(z1[::-1], z, 'k--', linewidth=2, label='Universal')
        
        plt.xlabel('Normalized Diffusivity')
        plt.ylabel('Normalized Depth (σ)')
        plt.title('Mean Shape Functions by Cluster')
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Add annotations
        plt.annotate('Surface (σ=0)', xy=(0.5, 0.02), xytext=(0.6, 0.05),
                    arrowprops=dict(arrowstyle='->', color='black'), color='black')
        plt.annotate('Bottom (σ=1)', xy=(0.5, 0.98), xytext=(0.6, 0.95),
                    arrowprops=dict(arrowstyle='->', color='black'), color='black')
        
        plt.show()