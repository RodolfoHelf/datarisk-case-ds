import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Data processor class to clean and prepare all data files
    """
    
    def __init__(self, data_folder='../data'):
        self.data_folder = data_folder
        self.processed_data = {}
        
    def read_all_files(self):
        """Read all parquet files from the data folder"""
        print("Reading all data files...")
        
        # Read base_cadastral
        self.processed_data['base_cadastral'] = pd.read_parquet(
            os.path.join(self.data_folder, 'base_cadastral.parquet')
        )
        print(f"✓ base_cadastral: {self.processed_data['base_cadastral'].shape}")
        
        # Read base_submissao
        self.processed_data['base_submissao'] = pd.read_parquet(
            os.path.join(self.data_folder, 'base_submissao.parquet')
        )
        print(f"✓ base_submissao: {self.processed_data['base_submissao'].shape}")
        
        # Read historico_emprestimos
        self.processed_data['historico_emprestimos'] = pd.read_parquet(
            os.path.join(self.data_folder, 'historico_emprestimos.parquet')
        )
        print(f"✓ historico_emprestimos: {self.processed_data['historico_emprestimos'].shape}")
        
        # Read historico_parcelas
        self.processed_data['historico_parcelas'] = pd.read_parquet(
            os.path.join(self.data_folder, 'historico_parcelas.parquet')
        )
        print(f"✓ historico_parcelas: {self.processed_data['historico_parcelas'].shape}")
        
        print("All files loaded successfully!\n")
        
    def handle_missing_values(self):
        """Handle missing values in all datasets"""
        print("Handling missing values...")
        
        # base_cadastral - fill missing ocupacao with 'Unknown'
        if 'base_cadastral' in self.processed_data:
            df = self.processed_data['base_cadastral']
            df['ocupacao'] = df['ocupacao'].fillna('Unknown')
            print(f"✓ base_cadastral: filled {df['ocupacao'].isnull().sum()} missing ocupacao values")
        
        # base_submissao - fill missing values
        if 'base_submissao' in self.processed_data:
            df = self.processed_data['base_submissao']
            df['valor_bem'] = df['valor_bem'].fillna(df['valor_credito'])  # Use credit value as fallback
            df['valor_parcela'] = df['valor_parcela'].fillna(df['valor_credito'] / 12)  # Estimate monthly payment
            print(f"✓ base_submissao: filled missing valor_bem and valor_parcela values")
        
        # historico_emprestimos - fill missing values
        if 'historico_emprestimos' in self.processed_data:
            df = self.processed_data['historico_emprestimos']
            
            # Fill missing dates with reasonable defaults
            df['data_liberacao'] = df['data_liberacao'].fillna(df['data_decisao'])
            
            # Convert data_decisao to datetime first
            df['data_decisao'] = pd.to_datetime(df['data_decisao'])
            
            # Fill missing first payment date (1 month after decision)
            df['data_primeiro_vencimento'] = df['data_primeiro_vencimento'].fillna(
                df['data_decisao'] + pd.DateOffset(months=1)
            )
            
            # Fill missing last payment date (use 12 months as default)
            df['data_ultimo_vencimento_original'] = df['data_ultimo_vencimento_original'].fillna(
                df['data_decisao'] + pd.DateOffset(months=12)
            )
            
            df['data_ultimo_vencimento'] = df['data_ultimo_vencimento'].fillna(df['data_ultimo_vencimento_original'])
            df['data_encerramento'] = df['data_encerramento'].fillna(df['data_ultimo_vencimento'])
            
            # Fill missing financial values
            df['valor_bem'] = df['valor_bem'].fillna(df['valor_credito'])
            df['valor_parcela'] = df['valor_parcela'].fillna(df['valor_credito'] / df['qtd_parcelas_planejadas'].fillna(12))
            df['valor_entrada'] = df['valor_entrada'].fillna(0)
            df['percentual_entrada'] = df['percentual_entrada'].fillna(0)
            df['qtd_parcelas_planejadas'] = df['qtd_parcelas_planejadas'].fillna(12)
            df['taxa_juros_padrao'] = df['taxa_juros_padrao'].fillna(0.02)  # 2% default
            df['taxa_juros_promocional'] = df['taxa_juros_promocional'].fillna(0.02)
            df['acompanhantes_cliente'] = df['acompanhantes_cliente'].fillna('None')
            df['flag_seguro_contratado'] = df['flag_seguro_contratado'].fillna(0)
            
            print(f"✓ historico_emprestimos: filled missing values in dates, financial fields, and flags")
        
        # historico_parcelas - fill missing values
        if 'historico_parcelas' in self.processed_data:
            df = self.processed_data['historico_parcelas']
            df['data_real_pagamento'] = df['data_real_pagamento'].fillna(df['data_prevista_pagamento'])
            df['valor_pago_parcela'] = df['valor_pago_parcela'].fillna(df['valor_previsto_parcela'])
            print(f"✓ historico_parcelas: filled missing payment dates and amounts")
        
        print("Missing values handled successfully!\n")
        
    def convert_data_types(self):
        """Convert data types according to specifications"""
        print("Converting data types...")
        
        # base_cadastral
        if 'base_cadastral' in self.processed_data:
            df = self.processed_data['base_cadastral']
            
            # Convert id_cliente to object (string)
            df['id_cliente'] = df['id_cliente'].astype('object')
            
            # Convert sexo to boolean (F=False, M=True)
            df['sexo'] = (df['sexo'] == 'M').astype(bool)
            
            # Convert data_nascimento to datetime
            df['data_nascimento'] = pd.to_datetime(df['data_nascimento'])
            
            # Convert boolean columns
            df['possui_carro'] = (df['possui_carro'] == 'Y').astype(bool)
            df['possui_imovel'] = (df['possui_imovel'] == 'Y').astype(bool)
            
            print(f"✓ base_cadastral: converted data types")
        
        # base_submissao
        if 'base_submissao' in self.processed_data:
            df = self.processed_data['base_submissao']
            
            # Convert id_cliente to object
            df['id_cliente'] = df['id_cliente'].astype('object')
            
            # Convert data_solicitacao to datetime
            df['data_solicitacao'] = pd.to_datetime(df['data_solicitacao'])
            
            # Convert tipo_contrato to boolean (Cash loans=False, Revolving loans=True)
            df['tipo_contrato'] = (df['tipo_contrato'] == 'Revolving loans').astype(bool)
            
            print(f"✓ base_submissao: converted data types")
        
        # historico_emprestimos
        if 'historico_emprestimos' in self.processed_data:
            df = self.processed_data['historico_emprestimos']
            
            # Convert id_cliente and id_contrato to object
            df['id_cliente'] = df['id_cliente'].astype('object')
            df['id_contrato'] = df['id_contrato'].astype('object')
            
            # Convert date columns to datetime
            date_columns = [
                'data_decisao', 'data_liberacao', 'data_primeiro_vencimento',
                'data_ultimo_vencimento_original', 'data_ultimo_vencimento', 'data_encerramento'
            ]
            for col in date_columns:
                df[col] = pd.to_datetime(df[col])
            
            # Convert boolean columns
            df['flag_ultima_solicitacao_contrato'] = (df['flag_ultima_solicitacao_contrato'] == 'Y').astype(bool)
            df['flag_ultima_solicitacao_dia'] = df['flag_ultima_solicitacao_dia'].astype(bool)
            
            # Convert tipo_contrato to boolean (Cash loans=False, Revolving loans=True, Consumer loans=True, XNA=False)
            df['tipo_contrato'] = df['tipo_contrato'].isin(['Revolving loans', 'Consumer loans']).astype(bool)
            
            print(f"✓ historico_emprestimos: converted data types")
        
        # historico_parcelas
        if 'historico_parcelas' in self.processed_data:
            df = self.processed_data['historico_parcelas']
            
            # Convert id_cliente and id_contrato to object
            df['id_cliente'] = df['id_cliente'].astype('object')
            df['id_contrato'] = df['id_contrato'].astype('object')
            
            # Convert versao_parcela to object (categorical)
            df['versao_parcela'] = df['versao_parcela'].astype('object')
            
            # Convert date columns to datetime
            df['data_prevista_pagamento'] = pd.to_datetime(df['data_prevista_pagamento'])
            df['data_real_pagamento'] = pd.to_datetime(df['data_real_pagamento'])
            
            print(f"✓ historico_parcelas: converted data types")
        
        print("Data types converted successfully!\n")
        
    def drop_duplicates(self):
        """Drop duplicate rows from all datasets"""
        print("Dropping duplicates...")
        
        for name, df in self.processed_data.items():
            initial_rows = len(df)
            df.drop_duplicates(inplace=True)
            final_rows = len(df)
            duplicates_removed = initial_rows - final_rows
            
            if duplicates_removed > 0:
                print(f"✓ {name}: removed {duplicates_removed} duplicate rows")
            else:
                print(f"✓ {name}: no duplicates found")
        
        print("Duplicates removed successfully!\n")
        
    def save_processed_data(self, output_folder='../outputs/processed_data'):
        """Save processed data to parquet files"""
        print("Saving processed data...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        for name, df in self.processed_data.items():
            output_path = os.path.join(output_folder, f'{name}_processed.parquet')
            df.to_parquet(output_path, index=False)
            print(f"✓ {name}: saved to {output_path}")
        
        print("All processed data saved successfully!\n")
        
    def get_summary_stats(self):
        """Get summary statistics for all processed datasets"""
        print("=" * 60)
        print("PROCESSED DATA SUMMARY")
        print("=" * 60)
        
        for name, df in self.processed_data.items():
            print(f"\n{name.upper()}:")
            print(f"  Shape: {df.shape}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"  Missing values: {df.isnull().sum().sum()}")
            print(f"  Duplicate rows: {df.duplicated().sum()}")
            
            # Show data types
            print(f"  Data types:")
            for col, dtype in df.dtypes.items():
                print(f"    {col}: {dtype}")
        
        print("\n" + "=" * 60)
        
    def process_all(self):
        """Run the complete data processing pipeline"""
        print("Starting data processing pipeline...\n")
        
        # Step 1: Read all files
        self.read_all_files()
        
        # Step 2: Handle missing values
        self.handle_missing_values()
        
        # Step 3: Convert data types
        self.convert_data_types()
        
        # Step 4: Drop duplicates
        self.drop_duplicates()
        
        # Step 5: Save processed data
        self.save_processed_data()
        
        # Step 6: Show summary
        self.get_summary_stats()
        
        print("Data processing pipeline completed successfully!")

def main():
    """Main function to run the data processor"""
    processor = DataProcessor()
    processor.process_all()

if __name__ == "__main__":
    main()
