import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel,
    VarianceThreshold, SelectPercentile
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FeatureSelector:
    """
    Advanced feature selection class for credit risk modeling
    """
    
    def __init__(self, data_folder='../outputs/features'):
        self.data_folder = data_folder
        self.data = {}
        self.selected_features = {}
        self.feature_scores = {}
        self.correlation_matrix = {}
        
    def load_feature_data(self):
        """Load feature engineering output data"""
        print("Loading feature data...")
        
        # Try to load final dataset first
        final_dataset_path = os.path.join(self.data_folder, 'final_dataset.parquet')
        if os.path.exists(final_dataset_path):
            self.data['final_dataset'] = pd.read_parquet(final_dataset_path)
            print(f"✓ final_dataset: {self.data['final_dataset'].shape}")
        else:
            print("⚠ final_dataset.parquet not found")
            return False
        
        # Load other feature sets if available
        feature_files = [
            'client_features.parquet',
            'contract_features.parquet'
        ]
        
        for file in feature_files:
            file_path = os.path.join(self.data_folder, file)
            if os.path.exists(file_path):
                name = file.replace('.parquet', '')
                self.data[name] = pd.read_parquet(file_path)
                print(f"✓ {name}: {self.data[name].shape}")
        
        print("Feature data loaded successfully!\n")
        return True
        
    def remove_constant_features(self, threshold=0.01):
        """Remove features with very low variance (quasi-constant)"""
        print(f"Removing quasi-constant features (variance threshold: {threshold})...")
        
        if 'final_dataset' not in self.data:
            print("No final dataset found!")
            return
        
        df = self.data['final_dataset'].copy()
        initial_features = len(df.columns)
        
        # Get numerical columns only
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate variance for each numerical feature
        variances = df[numerical_cols].var()
        
        # Find features with variance below threshold
        low_variance_features = variances[variances < threshold].index.tolist()
        
        if low_variance_features:
            print(f"  Removing {len(low_variance_features)} low variance features")
            df.drop(columns=low_variance_features, inplace=True)
            self.data['final_dataset'] = df
        else:
            print("  No low variance features found")
        
        final_features = len(df.columns)
        print(f"  Features reduced from {initial_features} to {final_features}")
        print("Quasi-constant features removed successfully!\n")
        
    def remove_correlated_features_advanced(self, threshold=0.95, method='pearson'):
        """
        Advanced correlation removal with multiple methods
        
        Parameters:
        - threshold: Correlation threshold (default: 0.95)
        - method: Correlation method ('pearson', 'spearman', 'kendall')
        """
        print(f"Removing correlated features (threshold: {threshold}, method: {method})...")
        
        if 'final_dataset' not in self.data:
            print("No final dataset found!")
            return
        
        df = self.data['final_dataset'].copy()
        initial_features = len(df.columns)
        
        # Get numerical columns only
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) <= 1:
            print("  Not enough numerical features for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr(method=method).abs()
        self.correlation_matrix['final_dataset'] = corr_matrix
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Strategy 1: Remove features with highest correlation count
        high_corr_features = []
        for col in upper_tri.columns:
            high_corr_count = (upper_tri[col] > threshold).sum()
            if high_corr_count > 0:
                high_corr_features.append((col, high_corr_count))
        
        # Sort by correlation count (highest first)
        high_corr_features.sort(key=lambda x: x[1], reverse=True)
        
        # Remove features with most correlations
        to_remove = []
        for feature, corr_count in high_corr_features:
            if feature not in to_remove:
                # Find all features correlated with this one
                correlated_with_feature = upper_tri.columns[upper_tri[feature] > threshold].tolist()
                # Add correlated features to removal list (but not the current one)
                to_remove.extend([f for f in correlated_with_feature if f not in to_remove])
        
        if to_remove:
            print(f"  Removing {len(to_remove)} highly correlated features")
            df.drop(columns=to_remove, inplace=True)
            self.data['final_dataset'] = df
            
            # Update correlation matrix
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numerical_cols) > 1:
                self.correlation_matrix['final_dataset'] = df[numerical_cols].corr(method=method).abs()
        else:
            print("  No highly correlated features found")
        
        final_features = len(df.columns)
        print(f"  Features reduced from {initial_features} to {final_features}")
        print("Correlated features removed successfully!\n")
        
    def statistical_feature_selection(self, target_col='EVER30MOB03', method='f_classif', k=100):
        """
        Statistical feature selection using various methods
        
        Parameters:
        - target_col: Target variable column name
        - method: Selection method ('f_classif', 'mutual_info_classif', 'chi2')
        - k: Number of top features to select
        """
        print(f"Performing statistical feature selection (method: {method}, k: {k})...")
        
        if 'final_dataset' not in self.data:
            print("No final dataset found!")
            return
        
        df = self.data['final_dataset'].copy()
        
        # Check if target column exists
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found!")
            # Try to find alternative target columns
            target_candidates = [col for col in df.columns if any(x in col.upper() for x in ['EVER30', 'OVER60', 'DPD90'])]
            if target_candidates:
                target_col = target_candidates[0]
                print(f"Using alternative target column: {target_col}")
            else:
                print("No suitable target column found!")
                return
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # Remove non-numerical features
        X = X.select_dtypes(include=[np.number])
        
        if len(X.columns) == 0:
            print("No numerical features found for selection!")
            return
        
        print(f"  Features available for selection: {len(X.columns)}")
        print(f"  Target distribution: {y.value_counts().to_dict()}")
        
        # Perform feature selection
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
        elif method == 'mutual_info_classif':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(X.columns)))
        else:
            print(f"Method {method} not supported, using f_classif")
            selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
        
        # Fit and transform
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names and scores
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('score', ascending=False)
        
        # Store results
        self.selected_features[method] = selected_features
        self.feature_scores[method] = feature_scores
        
        print(f"  Selected {len(selected_features)} features using {method}")
        print(f"  Top 10 features by score:")
        for i, (_, row) in enumerate(feature_scores.head(10).iterrows()):
            print(f"    {i+1:2d}. {row['feature']:<30} Score: {row['score']:>10.4f} p-value: {row['p_value']:>8.4f}")
        
        # Create selected dataset
        selected_df = df[selected_features + [target_col]].copy()
        self.data[f'selected_{method}'] = selected_df
        
        print(f"Statistical feature selection completed successfully!\n")
        
    def recursive_feature_elimination(self, target_col='EVER30MOB03', n_features=50, estimator='rf'):
        """
        Recursive Feature Elimination (RFE)
        
        Parameters:
        - target_col: Target variable column name
        - n_features: Number of features to select
        - estimator: Estimator type ('rf' for RandomForest, 'lr' for LogisticRegression)
        """
        print(f"Performing Recursive Feature Elimination (n_features: {n_features}, estimator: {estimator})...")
        
        if 'final_dataset' not in self.data:
            print("No final dataset found!")
            return
        
        df = self.data['final_dataset'].copy()
        
        # Check if target column exists
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found!")
            return
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # Remove non-numerical features
        X = X.select_dtypes(include=[np.number])
        
        if len(X.columns) == 0:
            print("No numerical features found for selection!")
            return
        
        # Scale features for better performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Choose estimator
        if estimator == 'rf':
            base_estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif estimator == 'lr':
            base_estimator = LogisticRegression(random_state=42, max_iter=1000)
        else:
            print(f"Estimator {estimator} not supported, using RandomForest")
            base_estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Perform RFE
        rfe = RFE(
            estimator=base_estimator,
            n_features_to_select=min(n_features, len(X.columns)),
            step=0.1  # Remove 10% of features at each step
        )
        
        rfe.fit(X_scaled, y)
        
        # Get selected features
        selected_features = X.columns[rfe.support_].tolist()
        feature_ranking = pd.DataFrame({
            'feature': X.columns,
            'ranking': rfe.ranking_,
            'selected': rfe.support_
        }).sort_values('ranking')
        
        # Store results
        self.selected_features[f'rfe_{estimator}'] = selected_features
        self.feature_scores[f'rfe_{estimator}'] = feature_ranking
        
        print(f"  Selected {len(selected_features)} features using RFE with {estimator}")
        print(f"  Top 10 features by ranking:")
        for i, (_, row) in enumerate(feature_ranking.head(10).iterrows()):
            status = "✓" if row['selected'] else "✗"
            print(f"    {i+1:2d}. {status} {row['feature']:<30} Ranking: {row['ranking']}")
        
        # Create selected dataset
        selected_df = df[selected_features + [target_col]].copy()
        self.data[f'selected_rfe_{estimator}'] = selected_df
        
        print(f"Recursive Feature Elimination completed successfully!\n")
        
    def lasso_feature_selection(self, target_col='EVER30MOB03', alpha=0.01):
        """
        Lasso-based feature selection using LogisticRegression with L1 penalty
        
        Parameters:
        - target_col: Target variable column name
        - alpha: L1 regularization strength (smaller = more features)
        """
        print(f"Performing Lasso-based feature selection (alpha: {alpha})...")
        
        if 'final_dataset' not in self.data:
            print("No final dataset found!")
            return
        
        df = self.data['final_dataset'].copy()
        
        # Check if target column exists
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found!")
            return
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # Remove non-numerical features
        X = X.select_dtypes(include=[np.number])
        
        if len(X.columns) == 0:
            print("No numerical features found for selection!")
            return
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Fit Lasso logistic regression
        lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1/alpha, random_state=42, max_iter=1000)
        lasso.fit(X_scaled, y)
        
        # Get feature importance (absolute coefficients)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': np.abs(lasso.coef_[0]),
            'selected': np.abs(lasso.coef_[0]) > 0
        }).sort_values('coefficient', ascending=False)
        
        # Get selected features
        selected_features = feature_importance[feature_importance['selected']]['feature'].tolist()
        
        # Store results
        self.selected_features['lasso'] = selected_features
        self.feature_scores['lasso'] = feature_importance
        
        print(f"  Selected {len(selected_features)} features using Lasso")
        print(f"  Top 10 features by coefficient magnitude:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            status = "✓" if row['selected'] else "✗"
            print(f"    {i+1:2d}. {status} {row['feature']:<30} Coef: {row['coefficient']:>8.4f}")
        
        # Create selected dataset
        selected_df = df[selected_features + [target_col]].copy()
        self.data['selected_lasso'] = selected_df
        
        print(f"Lasso-based feature selection completed successfully!\n")
        
    def pca_feature_reduction(self, target_col='EVER30MOB03', n_components=0.95):
        """
        PCA-based feature reduction
        
        Parameters:
        - target_col: Target variable column name
        - n_components: Number of components or explained variance ratio
        """
        print(f"Performing PCA feature reduction (n_components: {n_components})...")
        
        if 'final_dataset' not in self.data:
            print("No final dataset found!")
            return
        
        df = self.data['final_dataset'].copy()
        
        # Check if target column exists
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found!")
            return
        
        # Prepare features (exclude target)
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].fillna(0)
        
        # Remove non-numerical features
        X = X.select_dtypes(include=[np.number])
        
        if len(X.columns) == 0:
            print("No numerical features found for PCA!")
            return
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        if isinstance(n_components, float):
            pca = PCA(n_components=n_components)
        else:
            pca = PCA(n_components=min(n_components, len(X.columns)))
        
        X_pca = pca.fit_transform(X_scaled)
        
        # Get explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"  Original features: {len(X.columns)}")
        print(f"  PCA components: {len(pca.components_)}")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        print(f"  Cumulative variance by component:")
        for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
            print(f"    Component {i+1:2d}: {var_ratio:>6.4f} (Cumulative: {cum_var:>6.4f})")
        
        # Create feature names for PCA components
        pca_feature_names = [f'PCA_{i+1}' for i in range(X_pca.shape[1])]
        
        # Create PCA dataset
        pca_df = pd.DataFrame(X_pca, columns=pca_feature_names, index=df.index)
        pca_df[target_col] = df[target_col]
        
        # Store results
        self.data['pca_reduced'] = pca_df
        self.feature_scores['pca'] = pd.DataFrame({
            'component': pca_feature_names,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance
        })
        
        print(f"PCA feature reduction completed successfully!\n")
        
    def create_feature_selection_summary(self):
        """Create a summary of all feature selection methods"""
        print("=" * 60)
        print("FEATURE SELECTION SUMMARY")
        print("=" * 60)
        
        if 'final_dataset' in self.data:
            print(f"\nOriginal dataset:")
            print(f"  Shape: {self.data['final_dataset'].shape}")
            print(f"  Features: {len(self.data['final_dataset'].columns)}")
        
        print(f"\nFeature selection results:")
        for method, features in self.selected_features.items():
            print(f"  {method.upper()}: {len(features)} features selected")
            
            # Show dataset info if available
            dataset_name = f'selected_{method}'
            if dataset_name in self.data:
                df = self.data[dataset_name]
                print(f"    Dataset: {df.shape}")
        
        # Show PCA results if available
        if 'pca_reduced' in self.data:
            df = self.data['pca_reduced']
            print(f"  PCA: {df.shape[1]-1} components (plus target)")
        
        print("\n" + "=" * 60)
        
    def save_selected_features(self, output_folder='../outputs/selected_features'):
        """Save all selected feature datasets"""
        print("Saving selected features...")
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Save selected datasets
        for name, df in self.data.items():
            if name.startswith('selected_') or name == 'pca_reduced':
                output_path = os.path.join(output_folder, f'{name}.parquet')
                df.to_parquet(output_path, index=False)
                print(f"✓ {name}: saved to {output_path}")
        
        # Save feature scores
        for method, scores in self.feature_scores.items():
            output_path = os.path.join(output_folder, f'{method}_scores.csv')
            scores.to_csv(output_path, index=False)
            print(f"✓ {method}_scores: saved to {output_path}")
        
        # Save correlation matrix if available
        if 'final_dataset' in self.correlation_matrix:
            corr_matrix = self.correlation_matrix['final_dataset']
            output_path = os.path.join(output_folder, 'correlation_matrix.csv')
            corr_matrix.to_csv(output_path)
            print(f"✓ correlation_matrix: saved to {output_path}")
        
        print("All selected features saved successfully!\n")
        
    def run_complete_feature_selection(self, target_col='EVER30MOB03'):
        """Run the complete feature selection pipeline"""
        print("Starting complete feature selection pipeline...\n")
        
        # Step 1: Load data
        if not self.load_feature_data():
            print("Failed to load feature data!")
            return
        
        # Step 2: Remove constant features
        self.remove_constant_features()
        
        # Step 3: Remove correlated features
        self.remove_correlated_features_advanced(threshold=0.95, method='pearson')
        
        # Step 4: Statistical feature selection
        self.statistical_feature_selection(target_col=target_col, method='f_classif', k=100)
        self.statistical_feature_selection(target_col=target_col, method='mutual_info_classif', k=100)
        
        # Step 5: Recursive feature elimination
        self.recursive_feature_elimination(target_col=target_col, n_features=50, estimator='rf')
        self.recursive_feature_elimination(target_col=target_col, n_features=50, estimator='lr')
        
        # Step 6: Lasso feature selection
        self.lasso_feature_selection(target_col=target_col, alpha=0.01)
        
        # Step 7: PCA feature reduction
        self.pca_feature_reduction(target_col=target_col, n_components=0.95)
        
        # Step 8: Create summary
        self.create_feature_selection_summary()
        
        # Step 9: Save results
        self.save_selected_features()
        
        print("Complete feature selection pipeline finished successfully!")

def main():
    """Main function to run the feature selection pipeline"""
    selector = FeatureSelector()
    selector.run_complete_feature_selection()

if __name__ == "__main__":
    main()
