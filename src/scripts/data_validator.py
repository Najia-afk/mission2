import pandas as pd
import numpy as np

class DataValidator:
    """A comprehensive class for data validation, quality assessment and metadata generation."""
    
    def __init__(self):
        """Initialize the DataValidator."""
        pass
        
    def generate_quality_report(self, df, dataset_name):
        """Generate a comprehensive data quality report."""
        report = pd.DataFrame({
            'Dataset': dataset_name,
            'Total Rows': df.shape[0],
            'Total Columns': df.shape[1],
            'Duplicate Rows': df.duplicated().sum(),
            'Missing Values (%)': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100).round(2)
        }, index=[0])
        return report
    
    def analyze_column(self, col):
        """Analyze a single column for type, fill rate, and null values."""
        col_data = col.dropna()
        col_dtypes = col.dtypes
        
        if col_data.empty:
            col_type = 'NaN'
            fill_percentage = 0.0
            nan_percentage = 100.0
            bad_null_percentage = 0.0
        else:
            type_counts = col_data.apply(lambda x: type(x).__name__).value_counts(normalize=True) * 100
            if len(type_counts) == 1:
                if type_counts.index[0] != 'NaN':
                    max_length = col_data.apply(lambda x: len(str(x))).max()
                    col_type = f"{type_counts.index[0]}({max_length})"
                else:
                    col_type = type_counts.index[0]
            else:
                # If multiple types are present, compute errorType with percentages
                error_type_details = ', '.join([f"{t}: {p:.2f}%" for t, p in type_counts.items()])
                col_type = f"errorType({error_type_details})"
            
            fill_percentage = col_data.size / col.size * 100
            nan_percentage = col.isna().sum() / col.size * 100
            
            # Check for other forms of null values
            bad_null_count = col.isin(['', 'None', 'NULL', 'null']).sum()
            bad_null_percentage = bad_null_count / col.size * 100
    
        return {
            'Column Name': col.name,
            'Dtype': col_dtypes,
            'Type': col_type,
            'Fill Percentage': fill_percentage,
            'NaN Percentage': nan_percentage,
            'Bad Null Percentage': bad_null_percentage
        }
    
    def analyze_dataframe(self, df):
        """Generate detailed metadata for each column in a DataFrame."""
        columns_info = []
        num_rows = len(df)
        for col_name in df.columns:
            col_info = self.analyze_column(df[col_name])
            columns_info.append(col_info)
        df_info = pd.DataFrame(columns_info)
        return df_info, num_rows
    
    def create_metadata_dfs(self, dfs):
        """Create metadata DataFrames for multiple input DataFrames."""
        metadata_dfs = {}
        for df_name, df in dfs.items():
            metadata_df, num_rows = self.analyze_dataframe(df)
            metadata_dfs[f'metadata_{df_name} {df.shape}'] = metadata_df
        return metadata_dfs
        
    def display_metadata_dfs(self, metadata_dfs):
        """Display metadata DataFrames in a formatted way."""
        for name, metadata_df in metadata_dfs.items():
            print(f"Metadata for {name}:")
            print(metadata_df)
            print("\n")
    
    def combine_metadata(self, metadata_dfs, output_path=None):
        """Combine multiple metadata DataFrames and optionally save to CSV."""
        combined_metadata = pd.concat(metadata_dfs.values(), keys=metadata_dfs.keys()).reset_index(level=0).rename(columns={'level_0': 'DataFrame'})
        
        if output_path:
            combined_metadata.to_csv(output_path, index=False)
            print(f"Combined metadata saved to {output_path}")
            
        return combined_metadata
            
    def analyze_missing_values(self, df):
        """Analyze missing values by column."""
        missing = pd.DataFrame(df.isnull().sum(), columns=['Missing Count'])
        missing['Missing Percentage'] = (missing['Missing Count'] / len(df) * 100).round(2)
        missing.reset_index(inplace=True)
        missing.rename(columns={'index': 'Column'}, inplace=True)
        return missing.sort_values('Missing Percentage', ascending=False)
    
    def check_duplicates(self, dfs, ignore_fields={}, mandatory_fields={}):
        """
        Checks for duplicate rows in the DataFrames and if no raw duplicates are found, 
        checks for composite key duplicates.
        
        Parameters:
        - dfs (dict): Dictionary of DataFrames to check.
        - ignore_fields (dict): Dictionary where keys are DataFrame names and values are lists of columns to ignore.
        - mandatory_fields (dict): Dictionary where keys are DataFrame names and values are lists of columns for composite keys.
        
        Returns:
        - result (dict): Dictionary with number of raw duplicate rows and composite key duplicate rows.
        """
        result = {}
        
        for df_name, df in dfs.items():
            if df_name in ignore_fields:
                # Drop the specified columns to ignore
                df_to_check = df.drop(columns=ignore_fields[df_name], errors='ignore')
            else:
                df_to_check = df
            
            # Find raw duplicates
            duplicate_rows = df_to_check.duplicated(keep=False)
            num_raw_duplicates = duplicate_rows.sum()
            
            # Check for composite key duplicates if no raw duplicates are found
            num_composite_key_duplicates = 0
            if num_raw_duplicates == 0 and df_name in mandatory_fields:
                composite_key_columns = mandatory_fields[df_name]
                if set(composite_key_columns).issubset(df.columns):
                    composite_key_duplicates = df.duplicated(subset=composite_key_columns, keep=False)
                    num_composite_key_duplicates = composite_key_duplicates.sum()
            
            # Add to result
            result[df_name] = (num_raw_duplicates, num_composite_key_duplicates)
            
            print(f"DataFrame '{df_name}': {num_raw_duplicates} raw duplicate rows found, {num_composite_key_duplicates} composite key duplicate rows found")
        
        return result
    
    def check_country_consistency(self, df, country_metadata):
        """Check consistency between country codes and names."""
        issues = []
        df_countries = df['CountryCode'].unique()
        metadata_countries = country_metadata['CountryCode'].unique()
        
        for country in df_countries:
            if country not in metadata_countries:
                issues.append({
                    'CountryCode': country,
                    'Issue': 'In data but missing from metadata'
                })
        return issues