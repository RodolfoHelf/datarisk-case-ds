import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering class for credit risk modeling
    """
    
    def __init__(self, data_folder='../outputs/processed_data'):
        self.data_folder = data_folder
        self.data = {}
        self.features = {}
        self.scalers = {}
        self.encoders = {}
        
    def load_processed_data(self):
        """Load processed data files"""
        print("Loading processed data...")
        
        # Load all processed data files
        files = [
            'base_cadastral_processed.parquet',
            'base_submissao_processed.parquet', 
            'historico_emprestimos_processed.parquet',
            'historico_parcelas_processed.parquet'
        ]
        
        for file in files:
            name = file.replace('_processed.parquet', '')
            file_path = os.path.join(self.data_folder, file)
            if os.path.exists(file_path):
                self.data[name] = pd.read_parquet(file_path)
                print(f"✓ {name}: {self.data[name].shape}")
            else:
                print(f"⚠ {file} not found")
        
        print("Data loaded successfully!\n")
        
    def create_temporal_features(self):
        """Create temporal features from date columns"""
        print("Creating temporal features...")
        
        # base_cadastral temporal features
        if 'base_cadastral' in self.data:
            df = self.data['base_cadastral'].copy()
            
            # Age from birth date
            df['idade'] = (pd.Timestamp.now() - df['data_nascimento']).dt.days / 365.25
            
            # Age groups
            df['idade_grupo'] = pd.cut(df['idade'], 
                                     bins=[0, 25, 35, 45, 55, 65, 100], 
                                     labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
            
            # Birth year and month
            df['ano_nascimento'] = df['data_nascimento'].dt.year
            df['mes_nascimento'] = df['data_nascimento'].dt.month
            df['trimestre_nascimento'] = df['data_nascimento'].dt.quarter
            
            self.data['base_cadastral'] = df
            print(f"✓ base_cadastral: added temporal features")
        
        # base_submissao temporal features
        if 'base_submissao' in self.data:
            df = self.data['base_submissao'].copy()
            
            # Application date features
            df['ano_solicitacao'] = df['data_solicitacao'].dt.year
            df['mes_solicitacao'] = df['data_solicitacao'].dt.month
            df['trimestre_solicitacao'] = df['data_solicitacao'].dt.quarter
            df['dia_semana_solicitacao_num'] = df['data_solicitacao'].dt.dayofweek
            df['mes_solicitacao_sin'] = np.sin(2 * np.pi * df['mes_solicitacao'] / 12)
            df['mes_solicitacao_cos'] = np.cos(2 * np.pi * df['mes_solicitacao'] / 12)
            
            # Season features
            df['estacao'] = pd.cut(df['mes_solicitacao'], 
                                 bins=[0, 3, 6, 9, 12], 
                                 labels=['Inverno', 'Primavera', 'Verão', 'Outono'])
            
            self.data['base_submissao'] = df
            print(f"✓ base_submissao: added temporal features")
        
        # historico_emprestimos temporal features
        if 'historico_emprestimos' in self.data:
            df = self.data['historico_emprestimos'].copy()
            
            # Contract duration features
            df['duracao_contrato_dias'] = (df['data_encerramento'] - df['data_decisao']).dt.days
            df['duracao_contrato_meses'] = df['duracao_contrato_dias'] / 30.44
            
            # Time between decision and first payment
            df['tempo_ate_primeiro_pagamento'] = (df['data_primeiro_vencimento'] - df['data_decisao']).dt.days
            
            # Decision date features
            df['ano_decisao'] = df['data_decisao'].dt.year
            df['mes_decisao'] = df['data_decisao'].dt.month
            df['trimestre_decisao'] = df['data_decisao'].dt.quarter
            
            # Month and quarter cyclical features
            df['mes_decisao_sin'] = np.sin(2 * np.pi * df['mes_decisao'] / 12)
            df['mes_decisao_cos'] = np.cos(2 * np.pi * df['mes_decisao'] / 12)
            df['trimestre_decisao_sin'] = np.sin(2 * np.pi * df['trimestre_decisao'] / 4)
            df['trimestre_decisao_cos'] = np.cos(2 * np.pi * df['trimestre_decisao'] / 4)
            
            self.data['historico_emprestimos'] = df
            print(f"✓ historico_emprestimos: added temporal features")
        
        # historico_parcelas temporal features
        if 'historico_parcelas' in self.data:
            df = self.data['historico_parcelas'].copy()
            
            # Payment timing features
            df['dias_atraso'] = (df['data_real_pagamento'] - df['data_prevista_pagamento']).dt.days
            df['atraso_categoria'] = pd.cut(df['dias_atraso'], 
                                          bins=[-np.inf, 0, 30, 60, 90, np.inf], 
                                          labels=['Antecipado', 'No prazo', '30+ dias', '60+ dias', '90+ dias'])
            
            # Payment date features
            df['ano_pagamento'] = df['data_real_pagamento'].dt.year
            df['mes_pagamento'] = df['data_real_pagamento'].dt.month
            df['trimestre_pagamento'] = df['data_real_pagamento'].dt.quarter
            
            # Month cyclical features
            df['mes_pagamento_sin'] = np.sin(2 * np.pi * df['mes_pagamento'] / 12)
            df['mes_pagamento_cos'] = np.cos(2 * np.pi * df['mes_pagamento'] / 12)
            
            self.data['historico_parcelas'] = df
            print(f"✓ historico_parcelas: added temporal features")
        
        print("Temporal features created successfully!\n")
        
    def create_lag_features(self):
        """Create lag features for time series analysis"""
        print("Creating lag features...")
        
        if 'historico_parcelas' in self.data:
            df = self.data['historico_parcelas'].copy()
            
            # Sort by contract, client, and payment date
            df = df.sort_values(['id_contrato', 'id_cliente', 'data_real_pagamento'])
            
            # Lag features for payment amounts
            df['valor_previsto_lag1'] = df.groupby('id_contrato')['valor_previsto_parcela'].shift(1)
            df['valor_previsto_lag2'] = df.groupby('id_contrato')['valor_previsto_parcela'].shift(2)
            df['valor_previsto_lag3'] = df.groupby('id_contrato')['valor_previsto_parcela'].shift(3)
            
            df['valor_pago_lag1'] = df.groupby('id_contrato')['valor_pago_parcela'].shift(1)
            df['valor_pago_lag2'] = df.groupby('id_contrato')['valor_pago_parcela'].shift(2)
            df['valor_pago_lag3'] = df.groupby('id_contrato')['valor_pago_parcela'].shift(3)
            
            # Lag features for days late
            df['dias_atraso_lag1'] = df.groupby('id_contrato')['dias_atraso'].shift(1)
            df['dias_atraso_lag2'] = df.groupby('id_contrato')['dias_atraso'].shift(2)
            df['dias_atraso_lag3'] = df.groupby('id_contrato')['dias_atraso'].shift(3)
            
            # Rolling averages
            df['valor_previsto_media_3'] = df.groupby('id_contrato')['valor_previsto_parcela'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
            df['valor_pago_media_3'] = df.groupby('id_contrato')['valor_pago_parcela'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
            df['dias_atraso_media_3'] = df.groupby('id_contrato')['dias_atraso'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
            
            self.data['historico_parcelas'] = df
            print(f"✓ historico_parcelas: added lag features")
        
        print("Lag features created successfully!\n")
        
    def create_target_features(self):
        """Create target features using EVER30MOB03, OVER60MOB06 and DPD90"""
        print("Creating target features...")
        
        if 'historico_parcelas' in self.data:
            df = self.data['historico_parcelas'].copy()
            
            # EVER30MOB03: Ever 30+ days past due in last 3 months
            df['EVER30MOB03'] = df.groupby('id_contrato')['dias_atraso'].rolling(3, min_periods=1).apply(
                lambda x: (x > 30).any(), raw=True
            ).reset_index(0, drop=True).astype(int)
            
            # OVER60MOB06: Over 60 days past due in last 6 months
            df['OVER60MOB06'] = df.groupby('id_contrato')['dias_atraso'].rolling(6, min_periods=1).apply(
                lambda x: (x > 60).any(), raw=True
            ).reset_index(0, drop=True).astype(int)
            
            # DPD90: Days past due (90+ days)
            df['DPD90'] = (df['dias_atraso'] > 90).astype(int)
            
            # Additional target features
            df['EVER60MOB03'] = df.groupby('id_contrato')['dias_atraso'].rolling(3, min_periods=1).apply(
                lambda x: (x > 60).any(), raw=True
            ).reset_index(0, drop=True).astype(int)
            
            df['EVER90MOB06'] = df.groupby('id_contrato')['dias_atraso'].rolling(6, min_periods=1).apply(
                lambda x: (x > 90).any(), raw=True
            ).reset_index(0, drop=True).astype(int)
            
            # Severity of delinquency
            df['severidade_atraso'] = pd.cut(df['dias_atraso'], 
                                           bins=[-np.inf, 0, 1, 30, 60, 90, np.inf], 
                                           labels=[0, 1, 2, 3, 4, 5])
            
            self.data['historico_parcelas'] = df
            print(f"✓ historico_parcelas: added target features")
        
        print("Target features created successfully!\n")
        
    def create_expanding_features(self):
        """Create expanding window features"""
        print("Creating expanding features...")
        
        if 'historico_parcelas' in self.data:
            df = self.data['historico_parcelas'].copy()
            
            # Expanding statistics for each contract
            df['valor_previsto_expanding_mean'] = df.groupby('id_contrato')['valor_previsto_parcela'].expanding().mean().reset_index(0, drop=True)
            df['valor_previsto_expanding_std'] = df.groupby('id_contrato')['valor_previsto_parcela'].expanding().std().reset_index(0, drop=True)
            df['valor_previsto_expanding_min'] = df.groupby('id_contrato')['valor_previsto_parcela'].expanding().min().reset_index(0, drop=True)
            df['valor_previsto_expanding_max'] = df.groupby('id_contrato')['valor_previsto_parcela'].expanding().max().reset_index(0, drop=True)
            
            df['valor_pago_expanding_mean'] = df.groupby('id_contrato')['valor_pago_parcela'].expanding().mean().reset_index(0, drop=True)
            df['valor_pago_expanding_std'] = df.groupby('id_contrato')['valor_pago_parcela'].expanding().std().reset_index(0, drop=True)
            
            df['dias_atraso_expanding_mean'] = df.groupby('id_contrato')['dias_atraso'].expanding().mean().reset_index(0, drop=True)
            df['dias_atraso_expanding_std'] = df.groupby('id_contrato')['dias_atraso'].expanding().std().reset_index(0, drop=True)
            
            # Expanding counts
            df['parcelas_pagas_expanding'] = df.groupby('id_contrato')['valor_pago_parcela'].expanding().count().reset_index(0, drop=True)
            df['parcelas_atrasadas_expanding'] = df.groupby('id_contrato')['dias_atraso'].expanding().apply(
                lambda x: (x > 0).sum(), raw=True
            ).reset_index(0, drop=True)
            
            self.data['historico_parcelas'] = df
            print(f"✓ historico_parcelas: added expanding features")
        
        print("Expanding features created successfully!\n")
        
    def encode_categorical_features(self):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        # base_cadastral encoding
        if 'base_cadastral' in self.data:
            df = self.data['base_cadastral'].copy()
            
            categorical_cols = ['tipo_renda', 'ocupacao', 'tipo_organizacao', 'nivel_educacao', 
                               'estado_civil', 'tipo_moradia', 'idade_grupo']
            
            for col in categorical_cols:
                if col in df.columns and df[col].dtype == 'object':
                    # Label encoding for ordinal categories
                    if col in ['nivel_educacao', 'idade_grupo']:
                        le = LabelEncoder()
                        df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
                        self.encoders[f'{col}_le'] = le
                    else:
                        # One-hot encoding for nominal categories
                        dummies = pd.get_dummies(df[col].fillna('Unknown'), prefix=col)
                        df = pd.concat([df, dummies], axis=1)
                        df.drop(columns=[col], inplace=True)
            
            self.data['base_cadastral'] = df
            print(f"✓ base_cadastral: encoded categorical features")
        
        # base_submissao encoding
        if 'base_submissao' in self.data:
            df = self.data['base_submissao'].copy()
            
            # Encode day of week
            if 'dia_semana_solicitacao' in df.columns:
                day_mapping = {'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2, 'THURSDAY': 3, 
                              'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6}
                df['dia_semana_solicitacao_encoded'] = df['dia_semana_solicitacao'].map(day_mapping)
            
            # Encode season
            if 'estacao' in df.columns:
                season_mapping = {'Inverno': 0, 'Primavera': 1, 'Verão': 2, 'Outono': 3}
                df['estacao_encoded'] = df['estacao'].map(season_mapping)
            
            self.data['base_submissao'] = df
            print(f"✓ base_submissao: encoded categorical features")
        
        # historico_emprestimos encoding
        if 'historico_emprestimos' in self.data:
            df = self.data['historico_emprestimos'].copy()
            
            categorical_cols = ['status_contrato', 'tipo_pagamento', 'finalidade_emprestimo', 
                               'tipo_cliente', 'faixa_rendimento', 'tipo_portfolio', 'tipo_produto', 
                               'categoria_bem', 'setor_vendedor', 'canal_venda', 'motivo_recusa', 
                               'acompanhantes_cliente']
            
            for col in categorical_cols:
                if col in df.columns and df[col].dtype == 'object':
                    # One-hot encoding for most categorical variables
                    dummies = pd.get_dummies(df[col].fillna('Unknown'), prefix=col)
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(columns=[col], inplace=True)
            
            self.data['historico_emprestimos'] = df
            print(f"✓ historico_emprestimos: encoded categorical features")
        
        # historico_parcelas encoding
        if 'historico_parcelas' in self.data:
            df = self.data['historico_parcelas'].copy()
            
            # Encode atraso_categoria
            if 'atraso_categoria' in df.columns:
                atraso_mapping = {'Antecipado': 0, 'No prazo': 1, '30+ dias': 2, '60+ dias': 3, '90+ dias': 4}
                df['atraso_categoria_encoded'] = df['atraso_categoria'].map(atraso_mapping)
            
            # Encode versao_parcela as categorical
            if 'versao_parcela' in df.columns:
                df['versao_parcela_encoded'] = df['versao_parcela'].astype('category').cat.codes
            
            self.data['historico_parcelas'] = df
            print(f"✓ historico_parcelas: encoded categorical features")
        
        print("Categorical features encoded successfully!\n")
        
    def scale_numerical_features(self):
        """Scale numerical features"""
        print("Scaling numerical features...")
        
        # base_cadastral scaling
        if 'base_cadastral' in self.data:
            df = self.data['base_cadastral'].copy()
            
            numerical_cols = ['qtd_filhos', 'qtd_membros_familia', 'renda_anual', 'nota_regiao_cliente', 
                             'nota_regiao_cliente_cidade', 'idade']
            
            # Standard scaling for most numerical features
            scaler = StandardScaler()
            df_scaled = df[numerical_cols].copy()
            df_scaled_scaled = scaler.fit_transform(df_scaled)
            df_scaled = pd.DataFrame(df_scaled_scaled, columns=df_scaled.columns, index=df_scaled.index)
            
            # Add scaled columns
            for col in numerical_cols:
                df[f'{col}_scaled'] = df_scaled[col]
            
            self.scalers['base_cadastral'] = scaler
            self.data['base_cadastral'] = df
            print(f"✓ base_cadastral: scaled numerical features")
        
        # base_submissao scaling
        if 'base_submissao' in self.data:
            df = self.data['base_submissao'].copy()
            
            numerical_cols = ['hora_solicitacao', 'valor_credito', 'valor_bem', 'valor_parcela']
            
            # Min-Max scaling for financial values (0-1 range)
            scaler = MinMaxScaler()
            df_scaled = df[numerical_cols].copy()
            df_scaled_scaled = scaler.fit_transform(df_scaled)
            df_scaled = pd.DataFrame(df_scaled_scaled, columns=df_scaled.columns, index=df_scaled.index)
            
            # Add scaled columns
            for col in numerical_cols:
                df[f'{col}_scaled'] = df_scaled[col]
            
            self.scalers['base_submissao'] = scaler
            self.data['base_submissao'] = df
            print(f"✓ base_submissao: scaled numerical features")
        
        # historico_emprestimos scaling
        if 'historico_emprestimos' in self.data:
            df = self.data['historico_emprestimos'].copy()
            
            numerical_cols = ['valor_solicitado', 'valor_credito', 'valor_bem', 'valor_parcela', 
                             'valor_entrada', 'percentual_entrada', 'qtd_parcelas_planejadas', 
                             'taxa_juros_padrao', 'taxa_juros_promocional', 'area_venda', 
                             'hora_solicitacao', 'duracao_contrato_dias', 'duracao_contrato_meses', 
                             'tempo_ate_primeiro_pagamento']
            
            # Remove columns that don't exist
            numerical_cols = [col for col in numerical_cols if col in df.columns]
            
            # Standard scaling
            scaler = StandardScaler()
            df_scaled = df[numerical_cols].copy()
            df_scaled_scaled = scaler.fit_transform(df_scaled)
            df_scaled = pd.DataFrame(df_scaled_scaled, columns=df_scaled.columns, index=df_scaled.index)
            
            # Add scaled columns
            for col in numerical_cols:
                df[f'{col}_scaled'] = df_scaled[col]
            
            self.scalers['historico_emprestimos'] = scaler
            self.data['historico_emprestimos'] = df
            print(f"✓ historico_emprestimos: scaled numerical features")
        
        # historico_parcelas scaling
        if 'historico_parcelas' in self.data:
            df = self.data['historico_parcelas'].copy()
            
            numerical_cols = ['numero_parcela', 'valor_previsto_parcela', 'valor_pago_parcela', 'dias_atraso']
            
            # Remove columns that don't exist
            numerical_cols = [col for col in numerical_cols if col in df.columns]
            
            # Standard scaling
            scaler = StandardScaler()
            df_scaled = df[numerical_cols].copy()
            df_scaled_scaled = scaler.fit_transform(df_scaled)
            df_scaled = pd.DataFrame(df_scaled_scaled, columns=df_scaled.columns, index=df_scaled.index)
            
            # Add scaled columns
            for col in numerical_cols:
                df[f'{col}_scaled'] = df_scaled[col]
            
            self.scalers['historico_parcelas'] = scaler
            self.data['historico_parcelas'] = df
            print(f"✓ historico_parcelas: scaled numerical features")
        
        print("Numerical features scaled successfully!\n")
        
    def remove_correlated_features(self, threshold=0.95):
        """Remove highly correlated features"""
        print(f"Removing correlated features (threshold: {threshold})...")
        
        for name, df in self.data.items():
            # Get numerical columns only
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) > 1:
                # Calculate correlation matrix
                corr_matrix = df[numerical_cols].corr().abs()
                
                # Find highly correlated pairs
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
                
                if to_drop:
                    print(f"  {name}: removing {len(to_drop)} correlated features")
                    df.drop(columns=to_drop, inplace=True)
                    self.data[name] = df
                else:
                    print(f"  {name}: no highly correlated features found")
        
        print("Correlated features removed successfully!\n")
        
    def create_aggregated_features(self):
        """Create aggregated features by client and contract"""
        print("Creating aggregated features...")
        
        # Aggregate by client from historico_parcelas
        if 'historico_parcelas' in self.data:
            df_parcelas = self.data['historico_parcelas']
            
            # Client-level aggregations
            client_features = df_parcelas.groupby('id_cliente').agg({
                'dias_atraso': ['mean', 'std', 'max', 'min', 'count'],
                'valor_previsto_parcela': ['mean', 'std', 'sum'],
                'valor_pago_parcela': ['mean', 'std', 'sum'],
                'EVER30MOB03': 'max',
                'OVER60MOB06': 'max',
                'DPD90': 'max'
            }).reset_index()
            
            # Flatten column names
            client_features.columns = ['id_cliente'] + [f'cliente_{col[0]}_{col[1]}' for col in client_features.columns[1:]]
            
            # Contract-level aggregations
            contract_features = df_parcelas.groupby('id_contrato').agg({
                'dias_atraso': ['mean', 'std', 'max', 'min', 'count'],
                'valor_previsto_parcela': ['mean', 'std', 'sum'],
                'valor_pago_parcela': ['mean', 'std', 'sum'],
                'EVER30MOB03': 'max',
                'OVER60MOB06': 'max',
                'DPD90': 'max'
            }).reset_index()
            
            # Flatten column names
            contract_features.columns = ['id_contrato'] + [f'contrato_{col[0]}_{col[1]}' for col in contract_features.columns[1:]]
            
            # Store aggregated features
            self.features['client_features'] = client_features
            self.features['contract_features'] = contract_features
            
            print(f"✓ Created client features: {client_features.shape}")
            print(f"✓ Created contract features: {contract_features.shape}")
        
        print("Aggregated features created successfully!\n")
        
    def merge_all_features(self):
        """Merge all features into a single dataset"""
        print("Merging all features...")
        
        # Start with base_cadastral
        if 'base_cadastral' in self.data:
            final_df = self.data['base_cadastral'].copy()
            
            # Merge with base_submissao
            if 'base_submissao' in self.data:
                final_df = final_df.merge(
                    self.data['base_submissao'], 
                    on='id_cliente', 
                    how='left'
                )
            
            # Merge with client features
            if 'client_features' in self.features:
                final_df = final_df.merge(
                    self.features['client_features'], 
                    on='id_cliente', 
                    how='left'
                )
            
            # Merge with historico_emprestimos (latest contract per client)
            if 'historico_emprestimos' in self.data:
                latest_contracts = self.data['historico_emprestimos'].sort_values('data_decisao').groupby('id_cliente').tail(1)
                final_df = final_df.merge(
                    latest_contracts, 
                    on='id_cliente', 
                    how='left',
                    suffixes=('', '_contrato')
                )
            
            # Merge with contract features
            if 'contract_features' in self.features and 'id_contrato' in final_df.columns:
                final_df = final_df.merge(
                    self.features['contract_features'], 
                    on='id_contrato', 
                    how='left'
                )
            
                    # Fill missing values - handle categorical and numerical columns separately
        for col in final_df.columns:
            if final_df[col].dtype.name == 'category':
                # For categorical columns, fill with mode (most frequent value) or 'Unknown'
                if final_df[col].notna().sum() > 0:
                    mode_value = final_df[col].mode().iloc[0] if len(final_df[col].mode()) > 0 else 'Unknown'
                    final_df[col] = final_df[col].fillna(mode_value)
                else:
                    final_df[col] = final_df[col].fillna('Unknown')
            elif final_df[col].dtype == 'object':
                # For object columns, fill with 'Unknown'
                final_df[col] = final_df[col].fillna('Unknown')
            elif final_df[col].dtype == 'bool':
                # For boolean columns, fill with False
                final_df[col] = final_df[col].fillna(False)
            else:
                # For numerical columns, fill with 0
                final_df[col] = final_df[col].fillna(0)
        
        # Ensure data type consistency for parquet saving
        for col in final_df.columns:
            if final_df[col].dtype == 'bool':
                # Convert boolean columns to int8 to avoid mixed types
                final_df[col] = final_df[col].astype('int8')
            elif final_df[col].dtype.name == 'category':
                # Convert categorical to string to avoid mixed types
                final_df[col] = final_df[col].astype('string')
            elif final_df[col].dtype == 'object':
                # Check if object column contains mixed types and convert appropriately
                unique_types = set([type(x) for x in final_df[col].dropna().unique()])
                if len(unique_types) > 1:
                    # Convert to string to handle mixed types
                    final_df[col] = final_df[col].astype('string')
                elif bool in unique_types:
                    # If it contains boolean values, convert to int8
                    final_df[col] = final_df[col].astype('int8')
                else:
                    # Keep as string for other object types
                    final_df[col] = final_df[col].astype('string')
        
        # Store the final dataset
        self.features['final_dataset'] = final_df
        print(f"✓ Final dataset created: {final_df.shape}")
        
        # Show feature summary
        print(f"\nFinal dataset columns: {len(final_df.columns)}")
        print(f"Final dataset rows: {len(final_df)}")
        print(f"Memory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("Feature merging completed successfully!\n")
        
    def save_features(self, output_folder='../outputs/features'):
        """Save all features to files"""
        print("Saving features...")
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Save individual feature sets
        for name, df in self.features.items():
            output_path = os.path.join(output_folder, f'{name}.parquet')
            df.to_parquet(output_path, index=False)
            print(f"✓ {name}: saved to {output_path}")
        
        # Save scalers and encoders
        import pickle
        scalers_path = os.path.join(output_folder, 'scalers.pkl')
        encoders_path = os.path.join(output_folder, 'encoders.pkl')
        
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.encoders, f)
        
        print(f"✓ Scalers saved to {scalers_path}")
        print(f"✓ Encoders saved to {encoders_path}")
        
        print("All features saved successfully!\n")
        
    def get_feature_summary(self):
        """Get summary of all features"""
        print("=" * 60)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 60)
        
        for name, df in self.data.items():
            print(f"\n{name.upper()}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        for name, df in self.features.items():
            print(f"\n{name.upper()}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n" + "=" * 60)
        
    def run_feature_engineering(self):
        """Run the complete feature engineering pipeline"""
        print("Starting feature engineering pipeline...\n")
        
        # Step 1: Load data
        self.load_processed_data()
        
        # Step 2: Create temporal features
        self.create_temporal_features()
        
        # Step 3: Create lag features
        self.create_lag_features()
        
        # Step 4: Create target features
        self.create_target_features()
        
        # Step 5: Create expanding features
        self.create_expanding_features()
        
        # Step 6: Encode categorical features
        self.encode_categorical_features()
        
        # Step 7: Scale numerical features
        self.scale_numerical_features()
        
        # Step 8: Remove correlated features
        self.remove_correlated_features()
        
        # Step 9: Create aggregated features
        self.create_aggregated_features()
        
        # Step 10: Merge all features
        self.merge_all_features()
        
        # Step 11: Save features
        self.save_features()
        
        # Step 12: Show summary
        self.get_feature_summary()
        
        print("Feature engineering pipeline completed successfully!")

def main():
    """Main function to run the feature engineering pipeline"""
    engineer = FeatureEngineer()
    engineer.run_feature_engineering()

if __name__ == "__main__":
    main()
