import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_binary_pies(df, binary_vars):
    
    cols = 2
    rows = len(binary_vars) // cols + len(binary_vars) % cols
    specs = [[{'type': 'domain'} for _ in range(cols)] for _ in range(rows)]
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=binary_vars,
                        specs=specs, )

    for i, var in enumerate(binary_vars):
        fig.add_trace(
            go.Pie(labels=df[var].unique(), values=df[var].fillna('NaN').value_counts(), name=var),
            row=(i // cols) + 1, col=(i % cols) + 1)
        
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                    marker=dict(line=dict(color='#000000', width=2)))
    fig.update_layout(height=800, showlegend=True, plot_bgcolor='rgba(240,240,245,1)')

    fig.update_layout(paper_bgcolor='WhiteSmoke', 
                    title='Binary features repartition',
                    title_font=dict(size=30),
                    title_x=0.5)
        
    fig.show()
    
    
    
