"""
Example usage of the FeatureSelector class

This script demonstrates different ways to use the feature selection pipeline
"""

from feature_selection import FeatureSelector
import pandas as pd
import numpy as np

def example_basic_usage():
    """Example 1: Basic usage - run complete pipeline"""
    print("=== Example 1: Basic Usage ===")
    
    selector = FeatureSelector()
    selector.run_complete_feature_selection()
    
    print("Basic feature selection pipeline completed!\n")

def example_step_by_step():
    """Example 2: Step-by-step feature selection"""
    print("=== Example 2: Step-by-Step Feature Selection ===")
    
    selector = FeatureSelector()
    
    # Step 1: Load data
    if not selector.load_feature_data():
        print("Failed to load data!")
        return
    
    # Step 2: Remove constant features
    selector.remove_constant_features(threshold=0.01)
    
    # Step 3: Remove correlated features with different thresholds
    print("Removing correlated features with different thresholds...")
    selector.remove_correlated_features_advanced(threshold=0.90, method='pearson')
    
    # Step 4: Statistical feature selection with different methods
    print("Performing statistical feature selection...")
    selector.statistical_feature_selection(target_col='EVER30MOB03', method='f_classif', k=50)
    selector.statistical_feature_selection(target_col='OVER60MOB06', method='mutual_info_classif', k=50)
    
    # Step 5: Recursive feature elimination
    print("Performing recursive feature elimination...")
    selector.recursive_feature_elimination(target_col='EVER30MOB03', n_features=30, estimator='rf')
    
    # Step 6: Lasso feature selection
    print("Performing Lasso feature selection...")
    selector.lasso_feature_selection(target_col='EVER30MOB03', alpha=0.005)
    
    # Step 7: Create summary and save
    selector.create_feature_selection_summary()
    selector.save_selected_features(output_folder='../outputs/custom_selected_features')
    
    print("Step-by-step feature selection completed!\n")

def example_custom_correlation_removal():
    """Example 3: Custom correlation removal strategies"""
    print("=== Example 3: Custom Correlation Removal ===")
    
    selector = FeatureSelector()
    if not selector.load_feature_data():
        print("Failed to load data!")
        return
    
    # Remove constant features first
    selector.remove_constant_features(threshold=0.01)
    
    # Try different correlation methods and thresholds
    methods = ['pearson', 'spearman']
    thresholds = [0.90, 0.85, 0.80]
    
    for method in methods:
        for threshold in thresholds:
            print(f"\nTrying {method} correlation with threshold {threshold}")
            selector.remove_correlated_features_advanced(threshold=threshold, method=method)
            
            # Check how many features remain
            if 'final_dataset' in selector.data:
                remaining_features = len(selector.data['final_dataset'].columns)
                print(f"  Remaining features: {remaining_features}")
                
                # Stop if we have a reasonable number of features
                if remaining_features <= 200:
                    print(f"  Target reached with {method} and threshold {threshold}")
                    break
    
    print("Custom correlation removal completed!\n")

def example_target_variable_analysis():
    """Example 4: Analyze different target variables"""
    print("=== Example 4: Target Variable Analysis ===")
    
    selector = FeatureSelector()
    if not selector.load_feature_data():
        print("Failed to load data!")
        return
    
    # Remove constant and correlated features first
    selector.remove_constant_features(threshold=0.01)
    selector.remove_correlated_features_advanced(threshold=0.95, method='pearson')
    
    # Find available target variables
    if 'final_dataset' in selector.data:
        df = selector.data['final_dataset']
        target_candidates = [col for col in df.columns if any(x in col.upper() for x in ['EVER30', 'OVER60', 'DPD90'])]
        
        print(f"Available target variables: {target_candidates}")
        
        # Analyze each target variable
        for target in target_candidates:
            print(f"\n--- Analyzing target: {target} ---")
            
            # Check target distribution
            target_dist = df[target].value_counts().to_dict()
            print(f"Target distribution: {target_dist}")
            
            # Perform feature selection for this target
            selector.statistical_feature_selection(target_col=target, method='f_classif', k=30)
            selector.statistical_feature_selection(target_col=target, method='mutual_info_classif', k=30)
            
            # Show top features for each method
            for method in ['f_classif', 'mutual_info_classif']:
                if method in selector.feature_scores:
                    scores = selector.feature_scores[method]
                    print(f"\nTop 5 features using {method}:")
                    for i, (_, row) in enumerate(scores.head(5).iterrows()):
                        print(f"  {i+1}. {row['feature']} (Score: {row['score']:.4f})")
    
    print("Target variable analysis completed!\n")

def example_feature_selection_comparison():
    """Example 5: Compare different feature selection methods"""
    print("=== Example 5: Feature Selection Method Comparison ===")
    
    selector = FeatureSelector()
    if not selector.load_feature_data():
        print("Failed to load data!")
        return
    
    # Remove constant and correlated features first
    selector.remove_constant_features(threshold=0.01)
    selector.remove_correlated_features_advanced(threshold=0.95, method='pearson')
    
    # Use EVER30MOB03 as target
    target_col = 'EVER30MOB03'
    
    # Apply different feature selection methods
    methods_config = [
        ('f_classif', {'k': 50}),
        ('mutual_info_classif', {'k': 50}),
        ('rfe_rf', {'n_features': 30, 'estimator': 'rf'}),
        ('rfe_lr', {'n_features': 30, 'estimator': 'lr'}),
        ('lasso', {'alpha': 0.01})
    ]
    
    results_comparison = {}
    
    for method_name, config in methods_config:
        print(f"\n--- Testing {method_name} ---")
        
        try:
            if method_name == 'f_classif':
                selector.statistical_feature_selection(target_col=target_col, method='f_classif', k=config['k'])
                selected_features = selector.selected_features.get('f_classif', [])
            elif method_name == 'mutual_info_classif':
                selector.statistical_feature_selection(target_col=target_col, method='mutual_info_classif', k=config['k'])
                selected_features = selector.selected_features.get('mutual_info_classif', [])
            elif method_name == 'rfe_rf':
                selector.recursive_feature_elimination(target_col=target_col, n_features=config['n_features'], estimator='rf')
                selected_features = selector.selected_features.get('rfe_rf', [])
            elif method_name == 'rfe_lr':
                selector.recursive_feature_elimination(target_col=target_col, n_features=config['n_features'], estimator='lr')
                selected_features = selector.selected_features.get('rfe_lr', [])
            elif method_name == 'lasso':
                selector.lasso_feature_selection(target_col=target_col, alpha=config['alpha'])
                selected_features = selector.selected_features.get('lasso', [])
            
            results_comparison[method_name] = {
                'n_features': len(selected_features),
                'features': selected_features[:10]  # First 10 features
            }
            
            print(f"  Selected {len(selected_features)} features")
            print(f"  Top 5 features: {selected_features[:5]}")
            
        except Exception as e:
            print(f"  Error with {method_name}: {e}")
            results_comparison[method_name] = {'n_features': 0, 'features': []}
    
    # Compare results
    print(f"\n{'='*60}")
    print("FEATURE SELECTION METHOD COMPARISON")
    print(f"{'='*60}")
    
    for method, result in results_comparison.items():
        print(f"\n{method.upper()}:")
        print(f"  Features selected: {result['n_features']}")
        if result['features']:
            print(f"  Sample features: {result['features'][:3]}")
    
    print(f"\n{'='*60}")
    print("Feature selection method comparison completed!\n")

def example_pca_analysis():
    """Example 6: PCA analysis with different components"""
    print("=== Example 6: PCA Analysis ===")
    
    selector = FeatureSelector()
    if not selector.load_feature_data():
        print("Failed to load data!")
        return
    
    # Remove constant and correlated features first
    selector.remove_constant_features(threshold=0.01)
    selector.remove_correlated_features_advanced(threshold=0.95, method='pearson')
    
    # Try different numbers of PCA components
    target_col = 'EVER30MOB03'
    component_options = [0.80, 0.85, 0.90, 0.95, 0.99]
    
    pca_results = {}
    
    for var_ratio in component_options:
        print(f"\n--- PCA with {var_ratio*100:.0f}% explained variance ---")
        
        try:
            selector.pca_feature_reduction(target_col=target_col, n_components=var_ratio)
            
            if 'pca_reduced' in selector.data:
                df = selector.data['pca_reduced']
                n_components = df.shape[1] - 1  # Exclude target column
                
                pca_results[var_ratio] = {
                    'n_components': n_components,
                    'explained_variance': var_ratio,
                    'dataset_shape': df.shape
                }
                
                print(f"  Components: {n_components}")
                print(f"  Dataset shape: {df.shape}")
                
        except Exception as e:
            print(f"  Error with {var_ratio}: {e}")
            pca_results[var_ratio] = {'n_components': 0, 'explained_variance': var_ratio, 'dataset_shape': (0, 0)}
    
    # Show PCA comparison
    print(f"\n{'='*60}")
    print("PCA COMPONENT ANALYSIS")
    print(f"{'='*60}")
    
    for var_ratio, result in pca_results.items():
        print(f"\n{var_ratio*100:.0f}% explained variance:")
        print(f"  Components: {result['n_components']}")
        print(f"  Dataset shape: {result['dataset_shape']}")
    
    print(f"\n{'='*60}")
    print("PCA analysis completed!\n")

if __name__ == "__main__":
    print("Feature Selection Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_step_by_step()
    example_custom_correlation_removal()
    example_target_variable_analysis()
    example_feature_selection_comparison()
    example_pca_analysis()
    
    print("All feature selection examples completed!")
