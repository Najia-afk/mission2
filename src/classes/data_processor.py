import pandas as pd
import numpy as np

class EdStatsProcessor:
    """
    Processes World Bank EdStats data and extracts meaningful features.
    """
    
    def __init__(self, data=None, series=None, country=None):
        """
        Initialize the processor with datasets.
        
        Parameters:
        - data: DataFrame containing the educational data
        - series: DataFrame containing indicator metadata
        - country: DataFrame containing country information
        """
        self.data = data
        self.series = series
        self.country = country
    
    def extract_latest_values(self, year_columns=None):
        """
        Extract the latest available values for each indicator and country.
        
        Parameters:
        - year_columns: List of column names representing years
        
        Returns:
        - DataFrame with latest values and list of indicators with no data
        """
        if self.data is None or self.series is None:
            raise ValueError("Data or series information not provided")
        
        if year_columns is None:
            year_columns = [col for col in self.data.columns 
                           if col.isdigit() and int(col) >= 1990 and int(col) <= 2036]
        
        unique_indicators = self.series['Series Code'].unique()
        indicators_no_data = []
        combined_latest_values = pd.DataFrame(index=self.data['Country Code'].unique())

        for indicator_code in unique_indicators:
            df_indicator = self.data[self.data['Indicator Code'] == indicator_code]

            if df_indicator[year_columns].isnull().all().all():
                indicators_no_data.append(indicator_code)
                continue

            heatmap_data = df_indicator.pivot_table(
                values=year_columns,
                index=['Country Code'],
                aggfunc='mean'
            )

            present_year_columns = [col for col in year_columns if col in heatmap_data.columns]

            if not present_year_columns:
                indicators_no_data.append(indicator_code)
                continue

            latest_values = heatmap_data.apply(lambda row: row.dropna().values[-1] 
                                               if not row.dropna().empty else None, axis=1)
            combined_latest_values[indicator_code] = latest_values

        return combined_latest_values, indicators_no_data
    
    def calculate_statistics(self, latest_values=None):
        """
        Calculate statistics (mean, median, std dev) for each indicator.
        
        Parameters:
        - latest_values: DataFrame with latest values (if None, uses self.latest_values)
        
        Returns:
        - DataFrame with statistics for each indicator
        """
        if latest_values is None:
            raise ValueError("Latest values not provided")
        
        statistics = []

        for indicator_code in latest_values.columns:
            values = latest_values[indicator_code].dropna()

            if values.empty:
                continue

            mean_val = values.mean()
            median_val = values.median()
            std_val = values.std()
            
            indicator_name = self.series[
                self.series['Series Code'] == indicator_code
            ]['Indicator Name'].values[0]

            statistics.append({
                'Indicator Name': indicator_name,
                'Indicator Code': indicator_code,
                'Mean': mean_val,
                'Median': median_val,
                'Std Dev': std_val,
                'Count': len(values)
            })

        return pd.DataFrame(statistics)