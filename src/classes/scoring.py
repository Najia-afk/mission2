import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math  

class EdStatsScorer:
    """
    A class to calculate and visualize country scores based on weighted educational indicators.
    """
    
    def __init__(self, country_data=None, visualizer=None):
        """
        Initialize the EdStatsScorer with country data.
        
        Parameters:
        - country_data: DataFrame containing country metadata (Code, Name, Income Group)
        - visualizer: Reference to the EdStatsVisualizer instance for accessing data
        """
        self.country_data = country_data
        self.visualizer = visualizer  # Add this line
        self.statistics_df = None
        self.country_scores = None
        self.merged_scores_df = None

        # Define the default KPI weights
        self.weights = {
            'IT.NET.USER.P2': 0.15,      # Internet users (per 100 people)
            'IT.CMP.PCMP.P2': 0.15,      # Personal computers (per 100 people)
            'UIS.MS.56.T': 0.10,         # Total number of inbound internationally mobile students
            'UIS.MENFR.56': 0.05,        # Net flow ratio of internationally mobile students
            'UIS.MSEP.56': 0.05,         # Inbound mobility rate
            'PRJ.POP.ALL.2.MF': 0.03,    # Population projections (secondary)
            'PRJ.POP.ALL.3.MF': 0.03,    # Population projections (higher)
            'PRJ.POP.ALL.4.MF': 0.04,    # Population projections (post-secondary)
            'SE.SEC.ENRR': 0.05,         # Gross enrollment ratio (secondary)
            'SE.TER.ENRR': 0.05,         # Gross enrollment ratio (tertiary)
            'SE.TER.GRAD': 0.05,         # Tertiary education graduates
            'SE.XPD.SECO.ZS': 0.05,      # Expenditure on secondary education (% of GDP)
            'SE.XPD.TERT.ZS': 0.05,      # Expenditure on tertiary education (% of GDP)
            'SE.XPD.TOTL.GD.ZS': 0.05,   # Total expenditure on education (% of GDP)
            'SL.UEM.NEET.ZS': 0.05,      # Share of youth not in employment, education, or training
            'SE.SEC.ENRL.VO.ZS': 0.05    # Percentage of secondary students enrolled in vocational programs
        }
        
        # Define KPIs where lower values are better
        self.lower_is_better = {'SL.UEM.NEET.ZS': True}
        
        # Define KPI categories for visualization
        self.categories = {
            'Internet & Device Access (30%)': ['IT.NET.USER.P2', 'IT.CMP.PCMP.P2'],
            'Language Proficiency (20%)': ['UIS.MS.56.T', 'UIS.MENFR.56', 'UIS.MSEP.56'],
            'Socio-economic Context (40%)': ['PRJ.POP.ALL.2.MF', 'PRJ.POP.ALL.3.MF', 'PRJ.POP.ALL.4.MF', 
                                           'SE.SEC.ENRR', 'SE.TER.ENRR', 'SE.TER.GRAD', 
                                           'SE.XPD.SECO.ZS', 'SE.XPD.TERT.ZS', 'SE.XPD.TOTL.GD.ZS'],
            'Future Generation Needs (10%)': ['SL.UEM.NEET.ZS', 'SE.SEC.ENRL.VO.ZS']
        }
        
        # KPI descriptions for visualization and documentation
        self.kpi_descriptions = {
            'IT.NET.USER.P2': 'Internet users (per 100 people)',
            'IT.CMP.PCMP.P2': 'Personal computers (per 100 people)',
            'UIS.MS.56.T': 'Total inbound internationally mobile students',
            'UIS.MENFR.56': 'Net flow ratio of internationally mobile students',
            'UIS.MSEP.56': 'Inbound mobility rate',
            'PRJ.POP.ALL.2.MF': 'Population projection (Lower Secondary)',
            'PRJ.POP.ALL.3.MF': 'Population projection (Upper Secondary)',
            'PRJ.POP.ALL.4.MF': 'Population projection (Post Secondary)',
            'SE.SEC.ENRR': 'Gross enrolment ratio (secondary)',
            'SE.TER.ENRR': 'Gross enrolment ratio (tertiary)',
            'SE.TER.GRAD': 'Graduates from tertiary education',
            'SE.XPD.SECO.ZS': 'Expenditure on secondary education',
            'SE.XPD.TERT.ZS': 'Expenditure on tertiary education',
            'SE.XPD.TOTL.GD.ZS': 'Government expenditure on education as % of GDP',
            'SL.UEM.NEET.ZS': 'Share of youth not in education, employment or training',
            'SE.SEC.ENRL.VO.ZS': 'Percentage of secondary students in vocational programmes'
        }
        
        # Color scheme for visualizations
        self.color_scheme = px.colors.qualitative.Plotly
    
    def set_weights(self, weights):
        """
        Update the weights for KPIs.
        
        Parameters:
        - weights: Dictionary mapping indicator codes to weights (must sum to 1)
        """
        if sum(weights.values()) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.weights = weights
    
    def set_lower_is_better(self, indicators):
        """
        Update the list of indicators where lower values are better.
        
        Parameters:
        - indicators: Dictionary mapping indicator codes to boolean (True if lower is better)
        """
        self.lower_is_better = indicators
    
    def rank_to_score(self, rank):
        """
        Convert a rank to a score.
        
        Parameters:
        - rank: The rank value
        
        Returns:
        - score: Converted score
        """
        if pd.isna(rank) or rank > 20:  # If rank is NaN or greater than 20, return 0
            return 0
        if rank == 1:
            return 10
        elif 1 < rank <= 10:
            return 11 - rank
        else:
            return 0
    
    def adjust_rank_if_lower_is_better(self, rank, kpi, max_rank=20):
        """
        Adjust the rank if the KPI is one where lower values are better.
        
        Parameters:
        - rank: The rank value
        - kpi: The indicator code
        - max_rank: The maximum rank to consider
        
        Returns:
        - adjusted_rank: The possibly adjusted rank
        """
        if kpi in self.lower_is_better and self.lower_is_better[kpi]:
            if not pd.isna(rank) and rank <= max_rank:
                return max_rank - rank + 1
        return rank
    
    def calculate_scores(self, statistics_df, normalize=True):
        """
        Calculate scores for each country based on ranks and weights.
        
        Parameters:
        - statistics_df: DataFrame with statistics including country ranks
        - normalize: If True, normalize scores to 0-10 scale
        
        Returns:
        - country_scores_df: DataFrame with country scores
        """
        self.statistics_df = statistics_df
        country_scores = {}
        
        # Calculate raw scores
        for _, row in statistics_df.iterrows():
            country_code = row['Country Code']
            if pd.isnull(country_code):
                continue
            
            if country_code not in country_scores:
                country_scores[country_code] = 0
            
            for kpi, weight in self.weights.items():
                if kpi == row['Indicator Code']:
                    rank = row['Country Rank']
                    adjusted_rank = self.adjust_rank_if_lower_is_better(rank, kpi)
                    score = self.rank_to_score(adjusted_rank)
                    weighted_score = score * weight
                    country_scores[country_code] += weighted_score
        
        # Normalize if requested
        if normalize:
            max_possible_score = 10  # Perfect score in every indicator
            for country_code in country_scores:
                country_scores[country_code] = (country_scores[country_code] / max_possible_score) * 10
        
        self.country_scores = country_scores
        
        # Convert to DataFrame
        country_scores_df = pd.DataFrame(country_scores.items(), columns=['Country Code', 'Score'])
        
        # Merge with country metadata if available
        if self.country_data is not None:
            self.merged_scores_df = pd.merge(
                country_scores_df, 
                self.country_data[['Country Code', 'Income Group', 'Short Name']], 
                on='Country Code'
            )
        else:
            self.merged_scores_df = country_scores_df
        
        return self.merged_scores_df
    
    def get_top_countries_by_income_group(self, top_n=5):
        """
        Get the top N countries for each income group.
        
        Parameters:
        - top_n: Number of top countries to return per income group
        
        Returns:
        - top_per_income_group: Dictionary with income groups as keys and DataFrames as values
        """
        if self.merged_scores_df is None:
            raise ValueError("Scores not calculated. Call calculate_scores first.")
        
        if 'Income Group' not in self.merged_scores_df.columns:
            raise ValueError("Income Group not available in scores data.")
        
        top_per_income_group = {}
        income_groups = self.merged_scores_df['Income Group'].unique()
        
        for income_group in income_groups:
            group_data = self.merged_scores_df[self.merged_scores_df['Income Group'] == income_group]
            top_per_income_group[income_group] = group_data.nlargest(top_n, 'Score')
        
        return top_per_income_group
    
    
    def calculate_category_scores(self, country_code):
        """
        Calculate scores for each category for a given country.
        
        Parameters:
        - country_code: The country code to calculate scores for
        
        Returns:
        - category_scores: Dictionary mapping category names to scores on a 0-5 scale
        """
        if self.statistics_df is None:
            raise ValueError("Statistics data not available. Call calculate_scores first.")
            
        country_data = self.statistics_df[self.statistics_df['Country Code'] == country_code]
        
        # Calculate category scores
        category_scores = {}
        for category, indicators in self.categories.items():
            category_score = 0
            total_weight = 0
            
            for indicator in indicators:
                indicator_data = country_data[country_data['Indicator Code'] == indicator]
                
                if not indicator_data.empty:
                    row = indicator_data.iloc[0]
                    rank = row['Country Rank']
                    
                    if not pd.isna(rank):
                        adjusted_rank = self.adjust_rank_if_lower_is_better(rank, indicator)
                        score = self.rank_to_score(adjusted_rank)
                        weight = self.weights.get(indicator, 0)
                        
                        category_score += score * weight
                        total_weight += weight
            
            # Normalize score to 0-5 scale
            if total_weight > 0:
                # Using the same scale as the overall score (0-5)
                category_scores[category] = (category_score / total_weight)
            else:
                category_scores[category] = 0
        
        return category_scores
    
    def create_interactive_radar_chart(self, top_n=10):
        """
        Create an interactive radar chart with dropdown to select different countries.
        
        Parameters:
        - top_n: Number of top countries to include in the dropdown
        
        Returns:
        - fig: Plotly figure with the interactive radar chart and score breakdown
        """
        if self.statistics_df is None or self.merged_scores_df is None:
            raise ValueError("Statistics data or scores not available. Call calculate_scores first.")
            
        # Get top N countries by score
        top_countries = self.merged_scores_df.nlargest(top_n, 'Score')
        
        # Calculate category scores for all top countries
        all_category_scores = {}
        all_detailed_scores = {}  # Store detailed breakdown for each country
        categories_list = []
        
        for _, country_row in top_countries.iterrows():
            country_code = country_row['Country Code']
            country_name = country_row['Short Name']
            income_group = country_row['Income Group']
            
            # Calculate category scores using the common method
            all_category_scores[country_code] = self.calculate_category_scores(country_code)
            
            if not categories_list and all_category_scores[country_code]:
                categories_list = list(all_category_scores[country_code].keys())
            
            # Calculate detailed breakdown for each category
            detailed_scores = {}
            for category, indicators in self.categories.items():
                indicator_scores = {}
                for indicator in indicators:
                    indicator_data = self.statistics_df[
                        (self.statistics_df['Country Code'] == country_code) & 
                        (self.statistics_df['Indicator Code'] == indicator)
                    ]
                    
                    if not indicator_data.empty:
                        row = indicator_data.iloc[0]
                        rank = row['Country Rank']
                        
                        if not pd.isna(rank):
                            adjusted_rank = self.adjust_rank_if_lower_is_better(rank, indicator)
                            score = self.rank_to_score(adjusted_rank)
                            weight = self.weights.get(indicator, 0)
                            
                            indicator_scores[indicator] = {
                                'rank': int(rank),
                                'score': score * weight,
                                'raw_score': score,
                                'weight': weight * 100,  # Convert to percentage
                                'name': row['Indicator Name'] if 'Indicator Name' in row else indicator
                            }
                
                detailed_scores[category] = indicator_scores
                
            all_detailed_scores[country_code] = {
                'categories': all_category_scores[country_code],
                'details': detailed_scores,
                'income_group': income_group
            }
        
        # Simplify category names for better display
        simplified_categories = [cat.split('(')[0].strip() for cat in categories_list]
        category_letters = ['A', 'B', 'C', 'D']  # Assuming 4 categories
        
        # Create figure with subplots (radar chart on left, text on right)
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            specs=[[{"type": "polar"}, {"type": "xy"}]],
            horizontal_spacing=0.05
        )
        
        # Add a trace for each country (only first one visible)
        first_visible = True
        for _, country_row in top_countries.iterrows():
            country_code = country_row['Country Code']
            country_name = country_row['Short Name']
            
            # Get scores for this country
            scores = all_category_scores[country_code]
            values = [scores[cat] for cat in categories_list]
            
            # Add radar chart
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=simplified_categories,
                    fill='toself',
                    name=country_name,
                    visible=first_visible
                ),
                row=1, col=1
            )
            
            # Create score breakdown text
            country_details = all_detailed_scores[country_code]
            text_details = f"<b>{country_name} ({country_code})</b><br>"
            text_details += f"Income Group: {country_details['income_group']}<br>"
            text_details += f"Overall Score: {country_row['Score']:.2f}<br><br>"
            
            # Add category scores with their KPIs
            for i, (category, letter) in enumerate(zip(categories_list, category_letters)):
                cat_score = country_details['categories'][category]
                
                # Handle category names that may not have a colon
                if ':' in category:
                    category_display_name = category.split(':')[1].strip()
                else:
                    category_display_name = category
                    
                # Add category header with score
                text_details += f"<b>{letter}: {category_display_name}</b> - {cat_score:.2f}<br>"
                
                # Add individual indicator scores
                indicator_details = country_details['details'][category]
                if indicator_details:
                    for indicator, details in indicator_details.items():
                        indicator_name = details.get('name', indicator)
                        short_name = indicator_name[:30] + '...' if len(indicator_name) > 30 else indicator_name
                        text_details += f"  • {short_name} ({details['weight']:.1f}%)<br>"
                        text_details += f"    Rank: {details['rank']}, Score: {details['raw_score']:.1f}, Weighted: {details['score']:.2f}<br>"
                else:
                    text_details += f"  • No data available for this category<br>"
                
                text_details += "<br>"
            
            # Add the text annotation (only first one visible initially)
            fig.add_trace(
                go.Scatter(
                    x=[0.5],
                    y=[0],
                    mode='text',
                    text=text_details,
                    textposition="middle center",
                    hoverinfo='none',
                    visible=first_visible,
                    showlegend=False
                ),
                row=1, col=2
            )
            
            first_visible = False  # Only first country visible initially
        
        # Create dropdown menu
        dropdown_buttons = []
        for i, (_, country_row) in enumerate(top_countries.iterrows()):
            country_code = country_row['Country Code']
            country_name = country_row['Short Name']
            score = country_row['Score']
            
            # Create visibility list - only selected country visible
            # We have 2 traces per country (radar + text)
            visibility = [False] * (2 * len(top_countries))
            visibility[i*2] = True    # Radar chart
            visibility[i*2 + 1] = True  # Text details
            
            dropdown_buttons.append(
                dict(
                    method="update",
                    label=f"{country_name} (Score: {score:.2f})",
                    args=[
                        {"visible": visibility},
                        {"title": f"KPI Category Performance for {country_name} ({country_code})"}
                    ]
                )
            )
        
        # Add dropdown menu to layout
        fig.update_layout(
            showlegend=False,
            updatemenus=[
                dict(
                    active=0,
                    buttons=dropdown_buttons,
                    direction="down",
                    showactive=True,
                    x=0.5,
                    xanchor="center",
                    y=1.01,
                    yanchor="top"
                )
            ],
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickfont=dict(size=10),
                    tickangle=45
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            title=dict(
                text=f"KPI Category Performance for {top_countries.iloc[0]['Short Name']} ({top_countries.iloc[0]['Country Code']})",
                x=0.5,
                xanchor="center"
            ),
            height=850,

            margin=dict(t=120, b=50, l=80, r=80)
        )
        
        # Hide axes for the text part
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)
        
        
        return fig
    
    def create_income_group_dashboard(self, top_n=5):
        """
        Create a dashboard showing top countries by score for each income group.
        
        Parameters:
        - top_n: Number of top countries to display per income group
        
        Returns:
        - fig: Plotly figure with the dashboard
        """
        if self.merged_scores_df is None:
            raise ValueError("Scores not calculated. Call calculate_scores first.")
        
        if 'Income Group' not in self.merged_scores_df.columns:
            raise ValueError("Income Group not available in scores data.")
        
        # Get top countries by income group
        top_per_income_group = self.get_top_countries_by_income_group(top_n)
        income_groups = list(top_per_income_group.keys())
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, 
            cols=len(income_groups),
            subplot_titles=[f"{group}" for group in income_groups],
            horizontal_spacing=0.05
        )
        
        # Add bar charts for each income group
        for i, income_group in enumerate(income_groups):
            group_data = top_per_income_group[income_group].sort_values(by='Score', ascending=True)
            
            # Define colors based on score rank
            colors = []
            for j in range(len(group_data)):
                color_idx = j % len(self.color_scheme)
                colors.append(self.color_scheme[color_idx])
            
            # Create horizontal bar chart
            fig.add_trace(
                go.Bar(
                    x=group_data['Score'],
                    y=group_data['Short Name'],
                    marker_color=colors,
                    text=group_data['Score'].round(2),
                    textposition='outside',
                    orientation='h',
                    hovertemplate=(
                        "<b>%{y}</b><br>" +
                        "Score: %{x:.2f}<br>" +
                        "Country Code: %{customdata}<extra></extra>"
                    ),
                    customdata=group_data['Country Code'],
                    name=income_group
                ),
                row=1, col=i+1
            )
            
            # Update layout for each subplot
            fig.update_xaxes(title_text="Score", row=1, col=i+1)
        
        # Update overall layout
        fig.update_layout(
            title_text=f"Top {top_n} Countries by Score for Each Income Group",
            height=500,
            showlegend=False,
            margin=dict(t=80, b=40, l=20, r=20)
        )
        
        return fig

    def create_kpi_weight_visualization(self):
        """
        Create an interactive visualization of the KPI weights by category.
        
        Returns:
        - fig: Plotly figure with nested donut chart showing KPI weights
        """
        # Color schemes for categories
        category_colors = {
            'Internet & Device Access (30%)': '#4CAF50',  # Green
            'Language Proficiency (20%)': '#2196F3',      # Blue
            'Socio-economic Context (40%)': '#FF9800',    # Orange
            'Future Generation Needs (10%)': '#F44336'    # Red
        }
        
        # Calculate total weights for each group
        group_weights = {group: sum(self.weights[kpi] for kpi in kpis) 
                        for group, kpis in self.categories.items()}
        
        # Prepare data for outer donut (categories)
        category_labels = list(group_weights.keys())
        category_values = list(group_weights.values())
        
        # Prepare data for inner donut (all KPIs)
        all_kpi_labels = []
        all_kpi_values = []
        all_kpi_colors = []
        all_kpi_categories = []  # For category name
        all_kpi_descriptions = []  # For descriptions
        all_kpi_parents = []  # Track parent category for each KPI
        
        # Create empty dictionaries for legend items
        legend_labels = {}
        legend_colors = {}
        
        # Process all KPIs across all categories in a fixed order
        for category_idx, (category, kpis) in enumerate(self.categories.items()):
            base_color = category_colors[category]
            
            # Generate colors for KPIs in this category
            for i, kpi in enumerate(kpis):
                weight = self.weights[kpi]
                description = self.kpi_descriptions.get(kpi, kpi)
                
                # Format for inner donut
                all_kpi_labels.append(f"{kpi}")  # Use code for slice
                all_kpi_values.append(weight)
                all_kpi_categories.append(category)
                all_kpi_descriptions.append(description)
                all_kpi_parents.append(category)  # Link to parent category
                
                # Create gradient color based on position within category
                if len(kpis) > 1:
                    # Create a shade that varies from lighter to darker
                    brightness = 1.3 - (0.6 * i / (len(kpis) - 1))
                    
                    r = int(base_color[1:3], 16)
                    g = int(base_color[3:5], 16)
                    b = int(base_color[5:7], 16)
                    
                    r = min(255, max(0, int(r * brightness)))
                    g = min(255, max(0, int(g * brightness)))
                    b = min(255, max(0, int(b * brightness)))
                    
                    color = f'#{r:02x}{g:02x}{b:02x}'
                else:
                    color = base_color
                
                all_kpi_colors.append(color)
                
                # Store for legend
                legend_labels[kpi] = f"{description} ({weight*100:.0f}%)"
                legend_colors[kpi] = color
        
        # Create figure with two donut charts
        fig = go.Figure()
        
        # Add inner donut (categories) - this will be the inner one
        fig.add_trace(go.Pie(
            labels=category_labels,
            values=category_values,
            hole=0.90,
            textinfo='label+percent',
            textposition='inside',
            textfont=dict(size=12, color='white'),
            marker=dict(colors=[category_colors[cat] for cat in category_labels]),
            domain=dict(x=[0, 0.45], y=[0, 1]),  # Position on left half of figure
            sort=False,
            name='Categories',
            direction='clockwise',  # Ensure consistent direction
            rotation=90  # Start at the top
        ))
        
        # Add outer donut (all KPIs) - this will be the outer one
        fig.add_trace(go.Pie(
            labels=all_kpi_labels,
            values=all_kpi_values,
            hole=0.65,
            textinfo='none',  # No text on the slices
            hovertemplate="<b>%{label}</b><br>Description: %{text}<br>Category: %{customdata}<br>Weight: %{percent}<extra></extra>",
            text=all_kpi_descriptions,  # Use text for descriptions
            marker=dict(colors=all_kpi_colors),
            customdata=all_kpi_categories,  # Use customdata for categories
            domain=dict(x=[0, 0.45], y=[0, 1]),  # Position on left half of figure
            sort=False,
            direction='clockwise',  # Ensure consistent direction
            rotation=90  # Start at the top
        ))
        
        
        # Add legend to the right side using annotations grouped by category
        x_legend = 0.55  # Legend x position (start of right half)
        y_legend_start = 0.95  # Starting y position for legend
        y_step = 0.035  # Vertical spacing between legend items
        
        # Calculate legend positions without gaps
        y_positions = {}
        current_y = y_legend_start
            
        # First pass: Set category headers and their KPI positions
        for category, kpis in self.categories.items():
            y_positions[category] = {'header': current_y}
            current_y -= y_step
            
            for i, kpi in enumerate(kpis):
                y_positions[category][kpi] = current_y
                current_y -= y_step
            
            # Add a small space between categories
            current_y -= y_step/2

        # Add category headers and KPIs using the calculated positions
        for category, kpis in self.categories.items():
            header_y = y_positions[category]['header']
            
            # Add category header
            fig.add_annotation(
                text=f"<b>{category}</b>",
                x=x_legend,
                y=header_y,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(
                    color=category_colors[category],
                    size=12
                ),
                align="left",
                xanchor="left"
            )
            
            # Add KPI items under the category header
            for j, kpi in enumerate(kpis):
                item_y = y_positions[category][kpi]
                
                # Add colored box as a marker
                fig.add_shape(
                    type="rect",
                    x0=x_legend - 0.03,
                    y0=item_y - 0.01,
                    x1=x_legend - 0.01,
                    y1=item_y + 0.01,
                    fillcolor=legend_colors[kpi],
                    line_color=legend_colors[kpi],
                    xref="paper",
                    yref="paper"
                )
                
                # Add KPI label
                fig.add_annotation(
                    text=f"{kpi}: {legend_labels[kpi]}",
                    x=x_legend + 0.01,
                    y=item_y,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(
                        color="black",
                        size=10
                    ),
                    align="left",
                    xanchor="left"
                )
        
        # Update layout
        fig.update_layout(
            title="KPI Selection and Weighting Structure",
            title_x=0.5,  # Center the title
            showlegend=False,
            width=1000,
            height=700,
            margin=dict(t=80, b=40, l=20, r=20)
        )
        
        return fig
    
    def create_kpi_trend_charts(self, top_n=5, country_code=None, years=None):
        """
        Create interactive trend charts showing KPI values over time using Plotly.
        
        Parameters:
        - top_n: Number of top countries per income group to include in dropdown
        - country_code: Optional specific country code to display first
        - years: List of years to include (defaults to all available years)
        
        Returns:
        - fig: Plotly figure with interactive trend charts
        """
        if self.merged_scores_df is None:
            raise ValueError("Scores not calculated. Call calculate_scores first.")
        
        # Check if visualizer and data are available
        if not hasattr(self, 'visualizer') or not hasattr(self.visualizer, 'data_reduced'):
            raise ValueError("Visualizer with data_reduced attribute not available.")
        
        # Get the data frame with yearly values
        edstats_data_reduced = self.visualizer.data_reduced
        
        # Get metadata for indicator names
        metadata_df = self.visualizer.series_reduced if hasattr(self.visualizer, 'series_reduced') else None
        
        # Define years to use (all available year columns)
        if years is None:
            years = [col for col in edstats_data_reduced.columns if col.isdigit()]
            years.sort()
        
        # Function to determine if KPI is percentage-based
        def is_percentage_based(kpi_code, metadata_df):
            if metadata_df is None:
                return False
                
            indicator_name = metadata_df[metadata_df['Series Code'] == kpi_code]['Indicator Name'].values
            if len(indicator_name) > 0:
                return '%' in indicator_name[0] or '100' in indicator_name[0]
            return False
        
        # Get top countries for dropdown options
        top_per_income_group = self.get_top_countries_by_income_group(top_n)
        
        # Flatten the top countries into a single list
        all_top_countries = []
        for income_group, countries_df in top_per_income_group.items():
            for _, row in countries_df.iterrows():
                all_top_countries.append({
                    'country_code': row['Country Code'],
                    'country_name': row['Short Name'],
                    'income_group': row['Income Group'],
                    'score': row['Score']
                })
        
        # Sort by score
        all_top_countries = sorted(all_top_countries, key=lambda x: x['score'], reverse=True)
        
        # If specific country_code provided, move it to the front
        if country_code:
            for i, country in enumerate(all_top_countries):
                if country['country_code'] == country_code:
                    all_top_countries.insert(0, all_top_countries.pop(i))
                    break
        
        # Create subplots (2x2 grid)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[category.split('(')[0].strip() for category in self.categories.keys()],
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Create traces for each country (only first one visible)
        for country_idx, country_info in enumerate(all_top_countries):
            country_code = country_info['country_code']
            country_name = country_info['country_name']
            income_group = country_info['income_group']
            
            is_visible = country_idx == 0  # Only first country visible initially
            
            # Process each category
            for cat_idx, (category_name, indicators) in enumerate(self.categories.items()):
                row, col = cat_idx // 2 + 1, cat_idx % 2 + 1
                
                # Process indicators in this category
                for indicator in indicators:
                    kpi_data = edstats_data_reduced[
                        (edstats_data_reduced['Country Code'] == country_code) & 
                        (edstats_data_reduced['Indicator Code'] == indicator)
                    ]
                    
                    if kpi_data.empty:
                        continue
                        
                    # Extract values for available years
                    available_years = [y for y in years if y in kpi_data.columns]
                    if not available_years:
                        continue
                        
                    kpi_values = kpi_data[available_years].values[0]
                    valid_indices = ~pd.isna(kpi_values)
                    valid_years = [int(y) for y, valid in zip(available_years, valid_indices) if valid]
                    valid_values = kpi_values[valid_indices].tolist()
                    
                    if not valid_years:
                        continue
                    
                    # Get indicator name
                    if metadata_df is not None:
                        indicator_name = metadata_df[metadata_df['Series Code'] == indicator]['Indicator Name'].values
                        kpi_name = indicator_name[0] if len(indicator_name) > 0 else indicator
                    else:
                        kpi_name = indicator
                    
                    # Shorten name if too long
                    if len(kpi_name) > 30:
                        kpi_name = kpi_name[:27] + "..."
                    
                    # Use percentage yaxis for percentage-based KPIs
                    yaxis = "y" if is_percentage_based(indicator, metadata_df) else "y2"
                    
                    # Add line for this indicator
                    fig.add_trace(
                        go.Scatter(
                            x=valid_years,
                            y=valid_values,
                            mode='lines+markers',
                            name=kpi_name,
                            legendgroup=f"{country_code}_{indicator}",
                            legendgrouptitle_text=category_name.split('(')[0].strip(),
                            yaxis=yaxis,
                            visible=is_visible,
                            line=dict(width=2),
                            marker=dict(
                                size=6,
                                symbol='circle' if is_percentage_based(indicator, metadata_df) else 'x'
                            ),
                            hovertemplate=(
                                f"<b>{kpi_name}</b><br>" +
                                "Year: %{x}<br>" +
                                "Value: %{y:.2f}<extra></extra>"
                            )
                        ),
                        row=row, col=col
                    )
        
        # Update axes for each subplot
        for cat_idx, category_name in enumerate(self.categories.keys()):
            row, col = cat_idx // 2 + 1, cat_idx % 2 + 1
            
            # Update primary y-axis (percentage-based)
            fig.update_yaxes(
                title_text="Percentage-Based KPIs (%)",
                color="blue",
                rangemode="tozero",
                row=row, col=col
            )
            
            # Update secondary y-axis (non-percentage based)
            fig.update_yaxes(
                title_text="Other KPIs",
                color="green",
                rangemode="tozero",
                overlaying="y",
                side="right",
                secondary_y=True,
                row=row, col=col
            )
            
            # Update x-axis
            fig.update_xaxes(
                title_text="Year",
                tickangle=45,
                row=row, col=col
            )
        
        # Create dropdown menu for country selection
        dropdown_buttons = []
        for i, country_info in enumerate(all_top_countries):
            visibility = []
            for j in range(len(all_top_countries)):
                # Set visibility for all traces of this country
                # Each country has multiple traces (one per indicator)
                is_visible = (j == i)
                
                # Count how many traces this country has
                trace_count = 0
                for indicators in self.categories.values():
                    for indicator in indicators:
                        kpi_data = edstats_data_reduced[
                            (edstats_data_reduced['Country Code'] == all_top_countries[j]['country_code']) & 
                            (edstats_data_reduced['Indicator Code'] == indicator)
                        ]
                        
                        if not kpi_data.empty:
                            # Extract values for available years
                            available_years = [y for y in years if y in kpi_data.columns]
                            if available_years:
                                kpi_values = kpi_data[available_years].values[0]
                                if not pd.isna(kpi_values).all():
                                    trace_count += 1
                
                # Add visibility for all traces of this country
                visibility.extend([is_visible] * trace_count)
            
            # Add button for this country
            dropdown_buttons.append(
                dict(
                    method="update",
                    label=f"{country_info['country_name']} ({country_info['country_code']}) - {country_info['income_group']}",
                    args=[
                        {"visible": visibility},
                        {"title": f"KPI Trends for {country_info['country_name']} ({country_info['country_code']}) - {country_info['income_group']}"}
                    ]
                )
            )
        
        # Update the legend settings in the update_layout call:
        fig.update_layout(
            title=f"KPI Trends over Years - {all_top_countries[0]['country_name']} ({all_top_countries[0]['country_code']})",
            height=800,
  
            showlegend=True,
            legend=dict(
                orientation="h",           # Keep horizontal orientation 
                yanchor="bottom",
                y=-0.3,                    # Move it lower to give more space (was -0.2)
                xanchor="center",
                x=0.5,
                font=dict(size=9),         # Reduce font size
                itemsizing="constant",     # Make legend items consistent size
                itemwidth=40,              # Control width of legend items
                tracegroupgap=5            # Reduce gap between legend groups
            ),
            updatemenus=[
                dict(
                    active=0,
                    buttons=dropdown_buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.5,
                    xanchor="center",
                    y=1.15,
                    yanchor="top"
                )
            ],
            margin=dict(t=150, l=50, r=50, b=150)  # Increase bottom margin for legend (was b=100)
        )
                
        # Add grid lines for better readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig