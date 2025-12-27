import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from typing import Dict, List, Union

class CategoryTreeAnalyzer:
    """Analyze product category trees to visualize hierarchical category structures"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        
    def _parse_category_tree(self, category_tree: str) -> List[str]:
        """Extract category levels from category tree string"""
        if not isinstance(category_tree, str):
            return []
            
        # Split by ">>" to get category levels
        categories = [cat.strip() for cat in category_tree.split(">>")]
        return categories
    
    def _preprocess_category_tree(self, cat_tree: str) -> List[str]:
        """Preprocess a category tree string and return its categories."""
        if not isinstance(cat_tree, str):
            return []
            
        # Remove brackets if present
        if cat_tree.startswith('[') and cat_tree.endswith(']'):
            cat_tree = cat_tree[1:-1].strip('"\'')
                
        return self._parse_category_tree(cat_tree)
    
    def get_category_counts(self, max_depth: int = 4) -> pd.DataFrame:
        """
        Count occurrences of categories at each level.
        
        Args:
            max_depth: Maximum category depth to analyze
            
        Returns:
            DataFrame with category levels and counts
        """
        # Initialize category data
        category_data = []
        
        # Process each product's category tree
        for cat_tree in self.df['product_category_tree']:
            categories = self._preprocess_category_tree(cat_tree)
            
            # Skip empty categories
            if not categories:
                continue
            
            # Only consider up to max_depth levels
            categories = categories[:max_depth]
            
            # Create an entry for this category path, regardless of its depth
            entry = {}
            for j in range(len(categories)):
                entry[f'level{j+1}'] = categories[j]
            
            entry['count'] = 1
            category_data.append(entry)
        
        # Convert to DataFrame and aggregate counts
        if not category_data:
            return pd.DataFrame()
            
        df_categories = pd.DataFrame(category_data)
        
        # Group by all level columns and count occurrences
        level_cols = [col for col in df_categories.columns if col.startswith('level')]
        df_counts = df_categories.groupby(level_cols).size().reset_index(name='count')
        
        return df_counts
    
    def create_radial_category_chart(self, max_depth: int = None) -> go.Figure:
        """
        Create a radial icicle (sunburst) chart of the product category hierarchy using Graph Objects.
        
        Args:
            max_depth: Maximum category depth to visualize (None for unlimited)
            
        Returns:
            Plotly Figure object with the radial category chart
        """
        # Process category trees
        labels = []
        parents = []
        values = []
        ids = []
        
        # Process each product's category tree
        for cat_tree in self.df['product_category_tree']:
            categories = self._preprocess_category_tree(cat_tree)
            
            # Skip empty categories
            if not categories:
                continue
            
            # Limit to max_depth if specified
            if max_depth is not None:
                categories = categories[:max_depth]
            
            # Add hierarchical entries
            for i in range(len(categories)):
                # Create node ID
                if i == 0:
                    node_id = categories[0]
                    parent = ""  # Root node
                else:
                    parent = " >> ".join(categories[:i])
                    node_id = " >> ".join(categories[:i+1])
                
                # Add to lists if this node hasn't been seen before
                if node_id not in ids:
                    ids.append(node_id)
                    labels.append(categories[i])
                    parents.append(parent)
                    values.append(1)
                else:
                    # Increment value for existing node
                    idx = ids.index(node_id)
                    values[idx] += 1
        
        # Create figure using go.Sunburst
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total"
        ))
        
        # Update layout
        fig.update_layout(
            title="Product Category Hierarchy",
            margin=dict(t=50, l=0, r=0, b=0),
            width=1000,
            height=1000
        )
        
        return fig