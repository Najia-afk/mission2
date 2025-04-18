import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class HeatmapVisualizer:
    """
    Creates heatmap visualizations for World Bank EdStats data.
    """
    
    @staticmethod
    def create_heatmap(latest_values, series_data, country_data=None, indicator_subset=None, normalize=True):
        """
        Create a heatmap of the latest values for selected indicators across countries.
        
        Parameters:
        - latest_values: DataFrame with latest values
        - series_data: DataFrame with indicator metadata
        - country_data: DataFrame with country information
        - indicator_subset: List of indicator codes to include
        - normalize: Whether to normalize values for better visualization
        
        Returns:
        - Plotly figure object
        """
        if latest_values is None:
            raise ValueError("Latest values not provided")
            
        # If no subset specified, use all indicators
        if indicator_subset is None:
            indicator_subset = list(latest_values.columns)
        
        # Filter to only include the specified indicators
        heatmap_data = latest_values[indicator_subset].copy()
        
        # Remove rows with all NaN values
        heatmap_data = heatmap_data.dropna(how='all')
        
        # Remove columns with all NaN values
        heatmap_data = heatmap_data.dropna(axis=1, how='all')
        
        # Get indicator names for better labels
        indicator_names = {}
        for code in heatmap_data.columns:
            name = series_data[series_data['Series Code'] == code]['Indicator Name'].values
            indicator_names[code] = name[0] if len(name) > 0 else code
        
        # Create shortened names for display
        short_indicator_names = {}
        for code, name in indicator_names.items():
            if len(name) > 5:
                short_indicator_names[code] = name[:4] + "..."
            else:
                short_indicator_names[code] = name
        
        # Rename columns to use shortened indicator names
        column_names = [f"{short_indicator_names.get(col, col)} ({col})" for col in heatmap_data.columns]
        
        # Add country names if available
        if country_data is not None and 'Country Code' in country_data.columns and 'Short Name' in country_data.columns:
            country_names = country_data.set_index('Country Code')['Short Name'].to_dict()
            country_labels = [f"{idx} ({country_names.get(idx, 'Unknown')})" 
                        if idx in country_names else idx 
                        for idx in heatmap_data.index]
        else:
            country_labels = heatmap_data.index
        
        # Normalize data for better visualization
        if normalize:
            normalized_data = heatmap_data.copy()
            for col in normalized_data.columns:
                min_val = normalized_data[col].min()
                max_val = normalized_data[col].max()
                if not pd.isna(min_val) and not pd.isna(max_val) and (max_val > min_val):
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
            plot_data = normalized_data
            color_title = "Normalized Value"
        else:
            plot_data = heatmap_data
            color_title = "Value"
        
        # Create hover text with full information
        hover_text = []
        for idx in heatmap_data.index:
            country_name = country_names.get(idx, idx) if country_data is not None else idx
            row_text = []
            for col in heatmap_data.columns:
                indicator_full_name = indicator_names.get(col, col)
                value = heatmap_data.loc[idx, col]
                if pd.isna(value):
                    row_text.append(f"Country: {country_name}<br>Indicator: {indicator_full_name}<br>Value: No data")
                else:
                    if normalize:
                        normalized_value = plot_data.loc[idx, col]
                        row_text.append(
                            f"Country: {country_name}<br>"
                            f"Indicator: {indicator_full_name}<br>"
                            f"Code: {col}<br>"
                            f"Value: {value:.4g}<br>"
                            f"Normalized: {normalized_value:.4g}"
                        )
                    else:
                        row_text.append(
                            f"Country: {country_name}<br>"
                            f"Indicator: {indicator_full_name}<br>"
                            f"Code: {col}<br>"
                            f"Value: {value:.4g}"
                        )
            hover_text.append(row_text)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=plot_data.values,
            x=column_names,
            y=country_labels,
            colorscale='viridis',
            text=hover_text,
            hoverinfo='text',
            colorbar=dict(title=color_title)
        ))
        
        # Calculate dynamic size based on data dimensions
        width = min(1800, max(900, len(heatmap_data.columns) * 100))  # Width based on columns
        height = min(1200, max(600, len(heatmap_data) * 20))  # Height based on rows
        
        # Update layout for better readability
        fig.update_layout(
            autosize=False,
            width=width,
            height=height,
            margin=dict(
                l=200,  # Left margin for country names
                r=50,
                b=50,
                t=120,  # Top margin for indicator names
                pad=4
            ),
            xaxis=dict(
                side='top',
                tickangle=45,
                tickfont=dict(size=10),
                constrain='domain'  # Constrain xaxis to fit all labels
            ),
            yaxis=dict(
                title=dict(text='Countries', font=dict(size=14)),
                tickfont=dict(size=10),
                automargin=True  # Automatically adjust margin to fit labels
            ),
            title=dict(
                text="Heatmap of Education Indicators by Country",
                font=dict(size=18),
                y=0.98
            )
        )
        
        # Add income group filtering if available
        if country_data is not None and 'Income Group' in country_data.columns:
            # Get unique income groups - handle potential mix of types
            try:
                income_groups = country_data['Income Group'].dropna().unique().tolist()
                # Try to convert to strings to avoid comparison issues
                income_groups = [str(group) for group in income_groups]
                income_groups.sort()  # Sort after conversion to strings
            except Exception:
                # If sorting fails, just use them as-is
                income_groups = country_data['Income Group'].dropna().unique().tolist()
            
            # Create a dictionary to map country codes to income groups
            country_to_income = country_data.set_index('Country Code')['Income Group'].to_dict()
            
            # Add income group info to heatmap data
            income_group_data = {}
            for group in income_groups:
                # Convert group to string for safer comparison
                group_str = str(group)
                # Get countries in this income group
                countries_in_group = [code for code in heatmap_data.index 
                                    if code in country_to_income and str(country_to_income[code]) == group_str]
                
                if not countries_in_group:  # Skip if no countries in this group
                    continue
                    
                # Filtered data for this income group
                income_group_data[group_str] = heatmap_data.loc[countries_in_group]
            
            # Create dropdown for income group selection
            dropdown_buttons = [
                {'label': 'All Income Groups', 'method': 'update', 
                'args': [{'z': [plot_data.values], 'y': [country_labels]}]}
            ]
            
            for group in income_groups:
                group_str = str(group)
                if group_str not in income_group_data:  # Skip if no data for this group
                    continue
                    
                # Filter countries for this income group
                countries_in_group = [code for code in heatmap_data.index 
                                    if code in country_to_income and str(country_to_income[code]) == group_str]
                
                # Get country labels for this group
                group_country_labels = [f"{idx} ({country_names.get(idx, 'Unknown')})" 
                                if idx in country_names else idx 
                                for idx in countries_in_group]
                
                # Normalize data for this group if needed
                if normalize:
                    group_data_norm = income_group_data[group_str].copy()
                    for col in group_data_norm.columns:
                        min_val = group_data_norm[col].min()
                        max_val = group_data_norm[col].max()
                        if not pd.isna(min_val) and not pd.isna(max_val) and (max_val > min_val):
                            group_data_norm[col] = (group_data_norm[col] - min_val) / (max_val - min_val)
                    group_plot_data = group_data_norm
                else:
                    group_plot_data = income_group_data[group_str]
                    
                # Create hover text for this group
                group_hover_text = []
                for idx in countries_in_group:
                    country_name = country_names.get(idx, idx) if country_data is not None else idx
                    row_text = []
                    for col in heatmap_data.columns:
                        indicator_full_name = indicator_names.get(col, col)
                        value = heatmap_data.loc[idx, col]
                        if pd.isna(value):
                            row_text.append(f"Country: {country_name}<br>Indicator: {indicator_full_name}<br>Value: No data")
                        else:
                            if normalize:
                                normalized_value = group_plot_data.loc[idx, col]
                                row_text.append(
                                    f"Country: {country_name}<br>"
                                    f"Indicator: {indicator_full_name}<br>"
                                    f"Code: {col}<br>"
                                    f"Value: {value:.4g}<br>"
                                    f"Normalized: {normalized_value:.4g}"
                                )
                            else:
                                row_text.append(
                                    f"Country: {country_name}<br>"
                                    f"Indicator: {indicator_full_name}<br>"
                                    f"Code: {col}<br>"
                                    f"Value: {value:.4g}"
                                )
                    group_hover_text.append(row_text)
                
                # Add button for this income group
                dropdown_buttons.append(
                    {'label': f'{group}', 'method': 'update', 
                    'args': [{'z': [group_plot_data.values], 
                            'y': [group_country_labels],
                            'text': [group_hover_text]},
                            {'title': f"Heatmap of Education Indicators by Country - {group} Income Group"}]}
                )
            
            # Add dropdown menu to layout
            fig.update_layout(
                updatemenus=[{
                    'buttons': dropdown_buttons,
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.55,
                    'xanchor': 'left',
                    'y': 1.25,
                    'yanchor': 'top',
                    'bgcolor': 'white',
                    'font': {'color': 'black'}
                }],
                annotations=[{
                    'text': 'Income Group:',
                    'x': 0.5,
                    'y': 1.25,
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False
                }]
            )
        
        # Add toggle for normalization
        if normalize:
            # Create a new list for updatemenus
            updatemenus = []
            
            # Add existing menus if there are any
            if hasattr(fig.layout, 'updatemenus') and fig.layout.updatemenus:
                updatemenus = list(fig.layout.updatemenus)
            
            # Create non-normalized version of data for toggle
            raw_data_hover = []
            for idx in heatmap_data.index:
                country_name = country_names.get(idx, idx) if country_data is not None else idx
                row_text = []
                for col in heatmap_data.columns:
                    indicator_full_name = indicator_names.get(col, col)
                    value = heatmap_data.loc[idx, col]
                    if pd.isna(value):
                        row_text.append(f"Country: {country_name}<br>Indicator: {indicator_full_name}<br>Value: No data")
                    else:
                        row_text.append(
                            f"Country: {country_name}<br>"
                            f"Indicator: {indicator_full_name}<br>"
                            f"Code: {col}<br>"
                            f"Value: {value:.4g}"
                        )
                raw_data_hover.append(row_text)
            
            # Add normalization toggle
            updatemenus.append({
                'buttons': [
                    {'label': 'Normalized', 'method': 'update', 
                    'args': [{'z': [plot_data.values], 'text': [hover_text]}, 
                            {'coloraxis.colorbar.title.text': 'Normalized Value'}]},
                    {'label': 'Raw Values', 'method': 'update', 
                    'args': [{'z': [heatmap_data.values], 'text': [raw_data_hover]}, 
                            {'coloraxis.colorbar.title.text': 'Raw Value'}]}
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.5,
                'xanchor': 'left',
                'y': 1.20,
                'yanchor': 'top',
                'bgcolor': 'white',
                'font': {'color': 'black'}
            })
            
            # Update the layout with the combined menus
            fig.update_layout(updatemenus=updatemenus)
        
        # Add ability to zoom/pan
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=False),
                type='category',
                side='top',
                tickangle=45
            )
        )
        
        return fig


class ComparisonVisualizer:
    """
    Creates comparison visualizations for World Bank EdStats data.
    """
    
    @staticmethod
    def create_income_group_comparison(enriched_stats, indicator_code=None, series_data=None, 
                                    income_groups=None, top_n=5, country_data=None):
        """
        Create a chart comparing indicator values across top countries in different income groups.
        
        Parameters:
        - enriched_stats: DataFrame with enriched statistics
        - indicator_code: The indicator code to visualize (if None, provides dropdown)
        - series_data: DataFrame with indicator metadata
        - income_groups: List of income groups to include
        - top_n: Number of top countries to display per income group
        - country_data: DataFrame with country information
        
        Returns:
        - Plotly figure object
        """
        # Get all unique indicator codes
        available_indicators = enriched_stats['Indicator Code'].unique()
        
        if not available_indicators.size:
            raise ValueError("No indicators found in enriched statistics")
        
        # Use first indicator if none specified
        if indicator_code is None:
            indicator_code = available_indicators[0]
        elif indicator_code not in available_indicators:
            raise ValueError(f"Indicator code {indicator_code} not found in data")
        
        # Filter for the specific indicator
        indicator_data = enriched_stats[enriched_stats['Indicator Code'] == indicator_code]
        
        if indicator_data.empty:
            raise ValueError(f"No data found for indicator {indicator_code}")
        
        # Get indicator name
        indicator_name = indicator_code
        if series_data is not None and 'Series Code' in series_data.columns:
            name_match = series_data[series_data['Series Code'] == indicator_code]['Indicator Name'].values
            if len(name_match) > 0:
                indicator_name = name_match[0]
        
        # Add country names if available
        if country_data is not None and 'Country Code' in country_data.columns and 'Short Name' in country_data.columns:
            country_names = country_data.set_index('Country Code')['Short Name'].to_dict()
        else:
            country_names = {}
        
        # Get unique income groups from all data for consistent ordering
        all_income_groups = sorted([str(g) for g in enriched_stats['Income Group'].unique()])
        
        # Filter by income groups if specified
        if income_groups:
            filtered_groups = [str(g) for g in income_groups if str(g) in all_income_groups]
            if not filtered_groups:
                raise ValueError(f"None of the specified income groups {income_groups} are in the data")
            available_income_groups = filtered_groups
        else:
            available_income_groups = all_income_groups
        
        # Create initial figure with the first indicator
        fig = go.Figure()
        
        # Get color sequence for consistent colors
        colorway = px.colors.qualitative.Plotly
        
        # For each income group, get top countries
        for i, group in enumerate(available_income_groups):
            # Convert group to string to avoid concatenation issues
            group_str = str(group)
            
            group_data = indicator_data[indicator_data['Income Group'] == group]
            
            # Skip if no data for this group
            if group_data.empty:
                continue
                
            # Get top N countries
            top_countries = group_data.nlargest(top_n, 'Value')
            
            # Get X labels (with country names if available)
            x_labels = []
            for code in top_countries['Country Code']:
                if code in country_names:
                    x_labels.append(f"{code} ({country_names[code]})")
                else:
                    x_labels.append(code)
            
            # Add to chart - use group_str for string concatenation
            fig.add_trace(go.Bar(
                x=x_labels,
                y=top_countries['Value'],
                name=group_str,  # Use string version for name too
                text=top_countries['Value'].round(2),
                textposition='auto',
                marker_color=colorway[i % len(colorway)],
                hovertemplate=f"Country: %{{x}}<br>Value: %{{y:.2f}}<br>Income Group: {group_str}<extra></extra>"
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Top {top_n} Countries by {indicator_name} for Each Income Group',
            xaxis_title='Country',
            yaxis_title='Value',
            barmode='group',
            height=600,
            legend_title='Income Group',
            # Add more margin for x-axis labels
            margin=dict(b=100, l=50, r=50, t=120)
        )
        
        # If more than one indicator is available, add a dropdown to select
        if len(available_indicators) > 1:
            dropdown_buttons = []
            
            for ind_code in available_indicators:
                # Get indicator name
                ind_name = ind_code
                if series_data is not None and 'Series Code' in series_data.columns:
                    name_match = series_data[series_data['Series Code'] == ind_code]['Indicator Name'].values
                    if len(name_match) > 0:
                        ind_name = name_match[0]
                
                # Filter data for this indicator
                ind_data = enriched_stats[enriched_stats['Indicator Code'] == ind_code]
                
                # Create data for each income group
                updated_traces = []
                
                for i, group in enumerate(available_income_groups):
                    group_str = str(group)
                    group_data = ind_data[ind_data['Income Group'] == group]
                    
                    # If no data for this group, add empty trace
                    if group_data.empty:
                        updated_traces.append({
                            'x': [],
                            'y': [],
                            'name': group_str,
                            'marker': {'color': colorway[i % len(colorway)]},
                        })
                        continue
                    
                    # Get top N countries
                    top_countries = group_data.nlargest(top_n, 'Value')
                    
                    # Get X labels with country names
                    x_labels = []
                    for code in top_countries['Country Code']:
                        if code in country_names:
                            x_labels.append(f"{code} ({country_names[code]})")
                        else:
                            x_labels.append(code)
                    
                    # Add trace data
                    updated_traces.append({
                        'x': x_labels,
                        'y': top_countries['Value'].tolist(),
                        'name': group_str,
                        'text': top_countries['Value'].round(2).tolist(),
                        'marker': {'color': colorway[i % len(colorway)]},
                        'hovertemplate': f"Country: %{{x}}<br>Value: %{{y:.2f}}<br>Income Group: {group_str}<extra></extra>"
                    })
                
                # Add button for this indicator
                dropdown_buttons.append({
                    'method': 'update',
                    'label': ind_name[:30] + ('...' if len(ind_name) > 30 else ''),  # Truncate long names
                    'args': [
                        {'x': [trace['x'] for trace in updated_traces],
                        'y': [trace['y'] for trace in updated_traces],
                        'text': [trace.get('text', []) for trace in updated_traces],
                        'hovertemplate': [trace.get('hovertemplate', '') for trace in updated_traces]},
                        {'title': f'Top {top_n} Countries by {ind_name} for Each Income Group'}
                    ]
                })
            
            # Add dropdown menu
            fig.update_layout(
                updatemenus=[{
                    'buttons': dropdown_buttons,
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': 1.15,
                    'yanchor': 'top',
                    'bgcolor': 'white',
                    'font': {'color': 'black'}
                }])
            
            # Add slider for number of top countries
            slider_steps = []
            for n in range(1, 11):
                updated_slider_traces = []
                
                # Recalculate data for each income group with new top_n
                for i, group in enumerate(available_income_groups):
                    group_str = str(group)
                    group_data = indicator_data[indicator_data['Income Group'] == group]
                    
                    # If no data for this group, add empty trace
                    if group_data.empty:
                        updated_slider_traces.append({
                            'x': [],
                            'y': []
                        })
                        continue
                    
                    # Get top N countries with the new n value
                    top_countries = group_data.nlargest(n, 'Value')
                    
                    # Get X labels with country names
                    x_labels = []
                    for code in top_countries['Country Code']:
                        if code in country_names:
                            x_labels.append(f"{code} ({country_names[code]})")
                        else:
                            x_labels.append(code)
                    
                    # Add trace data
                    updated_slider_traces.append({
                        'x': x_labels,
                        'y': top_countries['Value'].tolist(),
                    })
                
                # Add step for this n value
                slider_steps.append({
                    'method': 'update',
                    'label': str(n),
                    'args': [
                        {'x': [trace['x'] for trace in updated_slider_traces],
                        'y': [trace['y'] for trace in updated_slider_traces]},
                        {'title': f'Top {n} Countries by {indicator_name} for Each Income Group'}
                    ]
                })
            
            # Add slider to layout
            fig.update_layout(
                sliders=[{
                    'active': top_n - 1,  # Convert top_n to zero-based index
                    'currentvalue': {"prefix": "Top N: "},
                    'pad': {"t": 50, "b": 10},
                    'steps': slider_steps,
                    'x': 0.5,
                    'xanchor': 'center',
                    'y': -0.05,
                    'yanchor': 'top'
                }]
            )
        
        return fig