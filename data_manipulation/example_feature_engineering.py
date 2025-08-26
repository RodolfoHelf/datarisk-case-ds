"""
Example usage of the FeatureEngineer class

This script demonstrates different ways to use the feature engineering pipeline
"""

from feature_engineering import FeatureEngineer
import numpy as np

def example_basic_usage():
    """Example 1: Basic usage - run complete pipeline"""
    print("=== Example 1: Basic Usage ===")
    
    engineer = FeatureEngineer()
    engineer.run_feature_engineering()
    
    print("Basic feature engineering pipeline completed!\n")

def example_step_by_step():
    """Example 2: Step-by-step feature creation"""
    print("=== Example 2: Step-by-Step Feature Creation ===")
    
    engineer = FeatureEngineer()
    
    # Step 1: Load processed data
    print("Loading processed data...")
    engineer.load_processed_data()
    
    # Step 2: Create specific feature types
    print("Creating temporal features...")
    engineer.create_temporal_features()
    
    print("Creating lag features...")
    engineer.create_lag_features()
    
    print("Creating target features...")
    engineer.create_target_features()
    
    print("Creating expanding features...")
    engineer.create_expanding_features()
    
    # Step 3: Encode and scale
    print("Encoding categorical features...")
    engineer.encode_categorical_features()
    
    print("Scaling numerical features...")
    engineer.scale_numerical_features()
    
    # Step 4: Remove correlations and create aggregates
    print("Removing correlated features...")
    engineer.remove_correlated_features(threshold=0.90)  # More aggressive correlation removal
    
    print("Creating aggregated features...")
    engineer.create_aggregated_features()
    
    # Step 5: Merge and save
    print("Merging all features...")
    engineer.merge_all_features()
    
    print("Saving features...")
    engineer.save_features(output_folder='../outputs/custom_features')
    
    # Step 6: Show summary
    engineer.get_feature_summary()
    
    print("Step-by-step feature creation completed!\n")

def example_custom_target_features():
    """Example 3: Custom target feature creation"""
    print("=== Example 3: Custom Target Features ===")
    
    engineer = FeatureEngineer()
    engineer.load_processed_data()
    
    # Access the historico_parcelas data
    if 'historico_parcelas' in engineer.data:
        df = engineer.data['historico_parcelas'].copy()
        
        # Create custom delinquency features
        print("Creating custom delinquency features...")
        
        # Custom 30-day delinquency (different from standard)
        df['custom_30_delinquency'] = (df['dias_atraso'] > 30).astype(int)
        
        # Custom 90-day delinquency (different from standard)
        df['custom_90_delinquency'] = (df['dias_atraso'] > 90).astype(int)
        
        # Payment ratio (actual vs expected)
        df['payment_ratio'] = df['valor_pago_parcela'] / df['valor_previsto_parcela']
        df['payment_ratio'] = df['payment_ratio'].fillna(1.0)  # Fill NaN with 1.0
        
        # Payment efficiency (1 = perfect, 0 = no payment)
        df['payment_efficiency'] = np.where(df['dias_atraso'] <= 0, 1.0, 
                                          np.maximum(0, 1 - (df['dias_atraso'] / 90)))
        
        # Update the data
        engineer.data['historico_parcelas'] = df
        
        print("Custom target features created successfully!")
        print(f"Features added: custom_30_delinquency, custom_90_delinquency, payment_ratio, payment_efficiency")
    
    print()

def example_feature_analysis():
    """Example 4: Analyze created features"""
    print("=== Example 4: Feature Analysis ===")
    
    engineer = FeatureEngineer()
    engineer.load_processed_data()
    
    # Create some basic features first
    engineer.create_temporal_features()
    engineer.create_target_features()
    
    # Analyze temporal features
    if 'base_cadastral' in engineer.data:
        df = engineer.data['base_cadastral']
        
        print("Temporal features analysis:")
        if 'idade' in df.columns:
            print(f"- Age range: {df['idade'].min():.1f} to {df['idade'].max():.1f}")
            print(f"- Age mean: {df['idade'].mean():.1f}")
            print(f"- Age groups distribution:")
            if 'idade_grupo' in df.columns:
                print(df['idade_grupo'].value_counts().sort_index())
    
    # Analyze target features
    if 'historico_parcelas' in engineer.data:
        df = engineer.data['historico_parcelas']
        
        print("\nTarget features analysis:")
        if 'EVER30MOB03' in df.columns:
            print(f"- EVER30MOB03: {df['EVER30MOB03'].value_counts().to_dict()}")
        if 'OVER60MOB06' in df.columns:
            print(f"- OVER60MOB06: {df['OVER60MOB06'].value_counts().to_dict()}")
        if 'DPD90' in df.columns:
            print(f"- DPD90: {df['DPD90'].value_counts().to_dict()}")
    
    print()

def example_feature_selection():
    """Example 5: Feature selection and importance"""
    print("=== Example 5: Feature Selection ===")
    
    engineer = FeatureEngineer()
    engineer.load_processed_data()
    
    # Create features first
    engineer.create_temporal_features()
    engineer.create_target_features()
    engineer.encode_categorical_features()
    engineer.scale_numerical_features()
    
    # Merge data for analysis
    engineer.create_aggregated_features()
    engineer.merge_all_features()
    
    if 'final_dataset' in engineer.features:
        final_df = engineer.features['final_dataset']
        
        print(f"Final dataset shape: {final_df.shape}")
        print(f"Total features: {len(final_df.columns)}")
        
        # Show feature types
        print("\nFeature types:")
        print(final_df.dtypes.value_counts())
        
        # Show numerical features
        numerical_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\nNumerical features: {len(numerical_cols)}")
        
        # Show categorical features
        categorical_cols = final_df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Categorical features: {len(categorical_cols)}")
        
        # Show boolean features
        boolean_cols = final_df.select_dtypes(include=['bool']).columns.tolist()
        print(f"Boolean features: {len(boolean_cols)}")
        
        # Memory usage
        print(f"\nMemory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print()

if __name__ == "__main__":
    print("Feature Engineering Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_step_by_step()
    example_custom_target_features()
    example_feature_analysis()
    example_feature_selection()
    
    print("All examples completed!")
