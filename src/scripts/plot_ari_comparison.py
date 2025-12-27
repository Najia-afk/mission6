def ari_comparison(ari_scores):
    """
    Creates a bar chart comparing ARI scores of different embedding methods.
    
    Args:
        ari_scores (dict): Dictionary of method names and their ARI scores
        
    Returns:
        plotly.graph_objects.Figure: The comparison visualization
    """
    import plotly.graph_objects as go
    
    # Sort scores for better visualization
    sorted_scores = sorted(ari_scores.items(), key=lambda x: x[1], reverse=True)
    methods, scores = zip(*sorted_scores)
    
    # Define colors based on performance
    colors = []
    for score in scores:
        if score >= 0.6:
            colors.append('#2ca02c')      # Green for excellent
        elif score >= 0.4:
            colors.append('#ff7f0e')      # Orange for good  
        elif score >= 0.2:
            colors.append('#d62728')      # Red for moderate
        else:
            colors.append('#7f7f7f')      # Gray for poor
    
    # Create the bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=methods,
        y=scores,
        marker_color=colors,
        text=[f'{score:.4f}' for score in scores],
        textposition='auto',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{x}</b><br>ARI Score: %{y:.4f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(text="Embedding Methods Performance Comparison (ARI Scores)", 
                  x=0.5, font=dict(size=16, color='#2c3e50')),
        xaxis=dict(title=dict(text="Embedding Method", font=dict(size=14)),
                  tickfont=dict(size=12), tickangle=45),
        yaxis=dict(title=dict(text="Adjusted Rand Index", font=dict(size=14)),
                  tickfont=dict(size=12), range=[0, max(scores) * 1.1]),
        height=500, width=800, template='plotly_white', 
        showlegend=False, margin=dict(t=60, b=100, l=80, r=40)
    )
    
    # Add performance threshold lines
    fig.add_hline(y=0.6, line_dash="dash", line_color="green", 
                 annotation_text="Excellent (≥0.6)", annotation_position="top right")
    fig.add_hline(y=0.4, line_dash="dash", line_color="orange",
                 annotation_text="Good (≥0.4)", annotation_position="top right") 
    fig.add_hline(y=0.2, line_dash="dash", line_color="red",
                 annotation_text="Moderate (≥0.2)", annotation_position="top right")
    
    return fig
