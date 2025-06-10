import json
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
from typing import Dict, List, Union
import plotly.express as px

class SpecificationsValueAnalyzer:
    """Analyze product specifications to extract key-value pairs and their frequencies"""
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        
    def _parse_specification(self, spec_str: str) -> List[Dict[str, str]]:
        """Extract key-value pairs from specification"""
        try:
            if isinstance(spec_str, str):
                spec_dict = json.loads(spec_str.replace("=>", ":"))
            else:
                spec_dict = spec_str
            return spec_dict['product_specification']
        except:
            return []
            
    def get_top_values(self, top_keys: int = 10, top_values: int = 5) -> pd.DataFrame:
        """Create DataFrame of top values for most common keys"""
        # Collect all key-value pairs
        key_value_pairs = {}
        
        for spec in self.df['product_specifications']:
            for item in self._parse_specification(spec):
                if 'key' in item and 'value' in item:
                    key = item['key']
                    value = item['value']
                    if key not in key_value_pairs:
                        key_value_pairs[key] = []
                    key_value_pairs[key].append(value)
        
        # Count frequencies and create result DataFrame
        results = []
        for key, values in key_value_pairs.items():
            value_counts = Counter(values)
            total_count = len(values)
            top_vals = value_counts.most_common(top_values)
            
            for value, count in top_vals:
                results.append({
                    'key': key,
                    'value': value,
                    'count': count,
                    'percentage': round(count/total_count * 100, 2),
                    'total_occurrences': total_count
                })
        
        # Create DataFrame and sort
        result_df = pd.DataFrame(results)
        result_df = (result_df
                    .sort_values(['total_occurrences', 'count'], ascending=[False, False])
                    .groupby('key')
                    .head(top_values)
                    .reset_index(drop=True))
        
        # Keep only top_keys unique keys
        top_keys_list = (result_df.groupby('key')['total_occurrences']
                        .first()
                        .nlargest(top_keys)
                        .index
                        .tolist())
        
        return result_df[result_df['key'].isin(top_keys_list)]
    
    def create_radial_icicle_chart(self, top_keys: int = 10, top_values: int = 5) -> go.Figure:
        """
        Create a radial icicle (sunburst) chart of the most common specification keys and values
        using plotly express.
        """

        # Get the data
        data_df = self.get_top_values(top_keys=top_keys, top_values=top_values)
        
        # Create a dataframe for plotly express with the right structure
        # We need to transform our data into a format suitable for px.sunburst
        sunburst_data = []
        
        for _, row in data_df.iterrows():
            sunburst_data.append({
                'level1': 'Specifications',
                'level2': row['key'],
                'level3': row['value'],
                'count': row['count']
            })
        
        sunburst_df = pd.DataFrame(sunburst_data)
        
        # Create the sunburst chart with plotly express
        fig = px.sunburst(
            sunburst_df,
            path=['level1', 'level2', 'level3'],  # Define the hierarchy
            values='count',                        # Size of segments
            title="Product Specifications Hierarchy",
            branchvalues='total',                 # Use 'total' for better circle proportion
            width=900,
            height=900
        )
        
        # Update layout for better appearance
        fig.update_layout(
            margin=dict(t=50, l=0, r=0, b=0),
            uniformtext=dict(minsize=10, mode='hide')
        )
        
        return fig