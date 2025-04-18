import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class EdStatsVisualizer:
    """
    A class to visualize World Bank EdStats data and analyze educational indicators across countries.
    """
    
    def __init__(self, data_reduced=None, series_reduced=None, country_reduced=None):
        """
        Initialize the visualizer with the reduced datasets.
        
        Parameters:
        - data_reduced: DataFrame containing the educational data
        - series_reduced: DataFrame containing the indicator descriptions
        - country_reduced: DataFrame containing country information
        """
        self.data_reduced = data_reduced
        self.series_reduced = series_reduced
        self.country_reduced = country_reduced
        self.latest_values = None
        self.indicators_no_data = []
        self.statistics_df = None
    
    def extract_latest_values(self, year_columns=None):
        """
        Extract the latest available values for each indicator and country.
        
        Parameters:
        - year_columns: List of column names representing years (e.g., ['2000', '2001', ...])
        
        Returns:
        - DataFrame with countries as index and indicators as columns
        """
        if self.data_reduced is None or self.series_reduced is None:
            raise ValueError("Data or series information not provided")
        
        if year_columns is None:
            # Try to detect year columns automatically
            year_columns = [col for col in self.data_reduced.columns 
                           if col.isdigit() and int(col) >= 1990 and int(col) <= 2036]
        
        unique_indicators = self.series_reduced['Series Code'].unique()
        self.indicators_no_data = []
        combined_latest_values = pd.DataFrame(index=self.data_reduced['Country Code'].unique())

        for indicator_code in unique_indicators:
            df_indicator = self.data_reduced[self.data_reduced['Indicator Code'] == indicator_code]

            if df_indicator[year_columns].isnull().all().all():
                self.indicators_no_data.append(indicator_code)
                continue

            # Create pivot table with countries as index and years as columns
            heatmap_data = df_indicator.pivot_table(
                values=year_columns,
                index=['Country Code'],
                aggfunc='mean'
            )

            present_year_columns = [col for col in year_columns if col in heatmap_data.columns]

            if not present_year_columns:
                self.indicators_no_data.append(indicator_code)
                continue

            # Extract the latest non-null value for each country
            latest_values = heatmap_data.apply(lambda row: row.dropna().values[-1] 
                                              if not row.dropna().empty else None, axis=1)
            combined_latest_values[indicator_code] = latest_values

        self.latest_values = combined_latest_values
        return combined_latest_values
    
    def calculate_statistics(self):
        """
        Calculate statistics (mean, median, std dev) for each indicator.
        
        Returns:
        - DataFrame with statistics for each indicator
        """
        if self.latest_values is None:
            raise ValueError("Latest values not extracted. Run extract_latest_values first.")
        
        statistics = []

        for indicator_code in self.latest_values.columns:
            values = self.latest_values[indicator_code].dropna()

            if values.empty:
                continue

            mean_val = values.mean()
            median_val = values.median()
            std_val = values.std()
            
            # Get the indicator name from the series info
            indicator_name = self.series_reduced[
                self.series_reduced['Series Code'] == indicator_code
            ]['Indicator Name'].values[0]

            statistics.append({
                'Indicator Name': indicator_name,
                'Indicator Code': indicator_code,
                'Mean': mean_val,
                'Median': median_val,
                'Std Dev': std_val,
                'Count': len(values)
            })

        self.statistics_df = pd.DataFrame(statistics)
        return self.statistics_df
        
    def select_top_countries(self, top_n=5):
        """
        Select top n countries for each income group based on indicator values.
        
        Parameters:
        - top_n: Number of top countries to select per income group
        
        Returns:
        - tuple: (Dictionary of top countries, DataFrame with enriched statistics)
        """
        if self.latest_values is None or self.statistics_df is None:
            raise ValueError("Latest values or statistics not calculated")
        
        if self.country_reduced is None:
            raise ValueError("Country information not provided")
        
        top_countries_dict = {}
        enriched_statistics = self.statistics_df.copy()
        income_groups = self.country_reduced['Income Group'].unique()

        for index, row in self.statistics_df.iterrows():
            indicator_code = row['Indicator Code']
            top_countries_dict[indicator_code] = {}
            for income_group in income_groups:
                countries_in_group = self.country_reduced[
                    self.country_reduced['Income Group'] == income_group
                ]['Country Code']
                countries_in_group = countries_in_group[
                    countries_in_group.isin(self.latest_values.index)
                ]
                
                latest_values_in_group = self.latest_values.loc[countries_in_group].dropna(how='all')
                if not latest_values_in_group.empty and indicator_code in latest_values_in_group.columns:
                    top_countries_in_group = latest_values_in_group[indicator_code].nlargest(top_n).index.tolist()
                    top_countries_dict[indicator_code][income_group] = top_countries_in_group
                    
                    # Create a list to hold new rows
                    new_rows = []
                    for position, country_code in enumerate(top_countries_in_group, start=1):
                        new_rows.append({
                            'Indicator Name': row['Indicator Name'],
                            'Indicator Code': indicator_code,
                            'Income Group': income_group,
                            'Country Code': country_code,
                            'Country Rank': position,
                            'Value': latest_values_in_group.loc[country_code, indicator_code]
                        })
                    
                    # Convert the list of new rows to a DataFrame and concatenate it to enriched_statistics
                    new_rows_df = pd.DataFrame.from_records(new_rows)
                    enriched_statistics = pd.concat([enriched_statistics, new_rows_df], ignore_index=True)

        return top_countries_dict, enriched_statistics
    
    def create_distribution_plots_with_selector(self):
        """
        Create an interactive distribution plot with a dropdown selector for all indicators.
        
        Returns:
        - A combined Plotly figure with dropdown selector for indicators
        """
        if self.latest_values is None or self.statistics_df is None:
            raise ValueError("Latest values or statistics not calculated")
        
        # Create dictionaries to store figures and data
        indicator_data = {}
        
        # First collect all the data we need
        for index, row in self.statistics_df.iterrows():
            indicator_code = row['Indicator Code']
            indicator_name = row['Indicator Name']
            mean_val = row['Mean']
            median_val = row['Median']
            std_val = row['Std Dev']

            values = self.latest_values[indicator_code].dropna()
            
            indicator_data[indicator_code] = {
                'name': indicator_name,
                'values': values,
                'mean': mean_val,
                'median': median_val,
                'std_dev': std_val
            }
        
        # Get the first indicator for initial display
        first_code = list(indicator_data.keys())[0]
        first_data = indicator_data[first_code]
        
        # Create subplot structure with violin on top and histogram below
        fig = make_subplots(rows=2, cols=1, row_heights=[0.3, 0.7], 
                            vertical_spacing=0.05, shared_xaxes=True)
        
        # Add traces for each indicator
        first_indicator = True
        for code, data in indicator_data.items():
            values = data['values']
            
            # Add violin plot in the top row
            fig.add_trace(
                go.Violin(
                    x=values,
                    orientation='h',
                    side='positive',
                    width=3,
                    points=False,
                    line_color='#1f77b4',
                    fillcolor='#1f77b4',
                    opacity=0.6,
                    visible=first_indicator
                ),
                row=1, col=1
            )
            
            # Add histogram in the bottom row
            fig.add_trace(
                go.Histogram(
                    x=values,
                    opacity=0.7,
                    marker_color='#1f77b4',
                    visible=first_indicator
                ),
                row=2, col=1
            )
            
            first_indicator = False
        
        # Prepare the dropdown menu
        dropdown_buttons = []
        
        # For each indicator, we need to set which traces should be visible
        for i, (code, data) in enumerate(indicator_data.items()):
            # Each indicator has 2 traces: violin and histogram
            visibility_list = [False] * (2 * len(indicator_data))
            # Make this indicator's traces visible
            visibility_list[i*2] = True     # Violin
            visibility_list[i*2+1] = True   # Histogram
            
            button = dict(
                method="update",
                label=data['name'],
                args=[
                    {"visible": visibility_list},
                    {
                        "title": f"Distribution of {data['name']}",
                        "shapes": [
                            # Mean line (row 2 - histogram)
                            dict(
                                type='line', 
                                x0=data['mean'], x1=data['mean'], 
                                y0=0, y1=1, 
                                yref='y2 domain',
                                line=dict(color='red', width=2, dash='dash')
                            ),
                            # Median line (row 2 - histogram)
                            dict(
                                type='line', 
                                x0=data['median'], x1=data['median'], 
                                y0=0, y1=1, 
                                yref='y2 domain',
                                line=dict(color='green', width=2)
                            ),
                            # +1 StdDev line (row 2 - histogram)
                            dict(
                                type='line', 
                                x0=data['mean'] + data['std_dev'], 
                                x1=data['mean'] + data['std_dev'], 
                                y0=0, y1=1, 
                                yref='y2 domain',
                                line=dict(color='blue', width=1, dash='dot')
                            ),
                            # -1 StdDev line (row 2 - histogram)
                            dict(
                                type='line', 
                                x0=data['mean'] - data['std_dev'], 
                                x1=data['mean'] - data['std_dev'], 
                                y0=0, y1=1, 
                                yref='y2 domain',
                                line=dict(color='blue', width=1, dash='dot')
                            )
                        ],
                        # Add annotations for the statistical lines
                        "annotations": [
                            # Mean annotation
                            dict(
                                x=data['mean'],
                                y=-0.15,  # Position below x-axis
                                xref='x2',
                                yref='paper',
                                text=f"Mean: {data['mean']:.2f}",
                                showarrow=False,
                                font=dict(size=10, color='red')
                            ),
                            # Median annotation
                            dict(
                                x=data['median'],
                                y=-0.22,  # Further below
                                xref='x2',
                                yref='paper',
                                text=f"Median: {data['median']:.2f}",
                                showarrow=False,
                                font=dict(size=10, color='green')
                            ),
                            # +1 StdDev annotation
                            dict(
                                x=data['mean'] + data['std_dev'],
                                y=-0.15,
                                xref='x2',
                                yref='paper',
                                text=f"+1 StdDev: {data['mean'] + data['std_dev']:.2f}",
                                showarrow=False,
                                font=dict(size=10, color='blue')
                            ),
                            # -1 StdDev annotation
                            dict(
                                x=data['mean'] - data['std_dev'],
                                y=-0.22,
                                xref='x2',
                                yref='paper',
                                text=f"-1 StdDev: {data['mean'] - data['std_dev']:.2f}",
                                showarrow=False,
                                font=dict(size=10, color='blue')
                            )
                        ]
                    }
                ]
            )
            dropdown_buttons.append(button)
        
        # Add initial shapes for the first indicator
        shapes = [
            # Mean line
            dict(
                type='line', 
                x0=first_data['mean'], x1=first_data['mean'], 
                y0=0, y1=1, 
                yref='y2 domain',
                line=dict(color='red', width=2, dash='dash')
            ),
            # Median line
            dict(
                type='line', 
                x0=first_data['median'], x1=first_data['median'], 
                y0=0, y1=1, 
                yref='y2 domain',
                line=dict(color='green', width=2)
            ),
            # +1 StdDev line
            dict(
                type='line', 
                x0=first_data['mean'] + first_data['std_dev'], 
                x1=first_data['mean'] + first_data['std_dev'], 
                y0=0, y1=1, 
                yref='y2 domain',
                line=dict(color='blue', width=1, dash='dot')
            ),
            # -1 StdDev line
            dict(
                type='line', 
                x0=first_data['mean'] - first_data['std_dev'], 
                x1=first_data['mean'] - first_data['std_dev'], 
                y0=0, y1=1, 
                yref='y2 domain',
                line=dict(color='blue', width=1, dash='dot')
            )
        ]
        
        # Add initial annotations
        annotations = [
            # Mean annotation
            dict(
                x=first_data['mean'],
                y=-0.15,  # Position below x-axis
                xref='x2',
                yref='paper',
                text=f"Mean: {first_data['mean']:.2f}",
                showarrow=False,
                font=dict(size=10, color='red')
            ),
            # Median annotation
            dict(
                x=first_data['median'],
                y=-0.22,  # Further below
                xref='x2',
                yref='paper',
                text=f"Median: {first_data['median']:.2f}",
                showarrow=False,
                font=dict(size=10, color='green')
            ),
            # +1 StdDev annotation
            dict(
                x=first_data['mean'] + first_data['std_dev'],
                y=-0.15,
                xref='x2',
                yref='paper',
                text=f"+1 StdDev: {first_data['mean'] + first_data['std_dev']:.2f}",
                showarrow=False,
                font=dict(size=10, color='blue')
            ),
            # -1 StdDev annotation
            dict(
                x=first_data['mean'] - first_data['std_dev'],
                y=-0.22,
                xref='x2',
                yref='paper',
                text=f"-1 StdDev: {first_data['mean'] - first_data['std_dev']:.2f}",
                showarrow=False,
                font=dict(size=10, color='blue')
            )
        ]
        # Update layout
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ],
            title=f"Distribution of {first_data['name']}",
            shapes=shapes,
            annotations=annotations,
            showlegend=False
        )
        
        # Update xaxis properties
        fig.update_xaxes(title="Value", row=2, col=1)
        
        # Update yaxis properties
        fig.update_yaxes(title="Density", row=1, col=1)
        fig.update_yaxes(title="Count", row=2, col=1)
        
        # Hide x-axis title for violin plot
        fig.update_xaxes(title="", row=1, col=1)
        
                
        return fig