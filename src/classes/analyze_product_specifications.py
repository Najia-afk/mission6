import json
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple

class SpecificationsAnalyzer:   
    """Analyze product specifications to extract keys and their frequencies"""
    def __init__(self, dataframe: pd.DataFrame):
        """Initialize with dataframe containing product_specifications column"""
        self.df = dataframe
        self.key_counts = None
        
    def _parse_specification(self, spec_str: str) -> List[str]:
        """Extract keys from a single product specification"""
        try:
            if isinstance(spec_str, str):
                spec_dict = json.loads(spec_str.replace("=>", ":"))
            else:
                spec_dict = spec_str
                
            return [item['key'] for item in spec_dict['product_specification'] 
                   if 'key' in item]
        except:
            return []
            
    def analyze_keys(self) -> Counter:
        """Count frequency of each specification key"""
        all_keys = []
        for spec in self.df['product_specifications']:
            keys = self._parse_specification(spec)
            all_keys.extend(keys)
        
        self.key_counts = Counter(all_keys)
        return self.key_counts
    
    def get_summary(self) -> Dict:
        """Get summary statistics of specifications"""
        if not self.key_counts:
            self.analyze_keys()
            
        return {
            'total_unique_keys': len(self.key_counts),
            'most_common_keys': self.key_counts.most_common(10),
            'total_products': len(self.df),
            'products_with_specs': sum(1 for spec in self.df['product_specifications'] 
                                     if self._parse_specification(spec))
        }