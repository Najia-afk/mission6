import json
import pandas as pd
from collections import Counter
from typing import Dict, List, Union

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