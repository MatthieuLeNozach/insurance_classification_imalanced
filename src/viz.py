import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

COLORS = ['#FED9CA', '#CAE1D4', '#F3E1EB', '#D6E2FE', '#FEE2D6']
COLORS2 = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6', '#c2f0c2']
COLORS = px.colors.qualitative.Set3


def plot_binary_pies(df, binary_vars, col_names):

    
    cols = 2
    rows = len(binary_vars) // cols + len(binary_vars) % cols
    specs = [[{'type': 'domain'} for _ in range(cols)] for _ in range(rows)]
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=col_names,
                        specs=specs)

    for i, var in enumerate(binary_vars):
        fig.add_trace(
            go.Pie(labels=df[var].unique(), values=df[var].fillna('NaN').value_counts(), name=var,
                marker=dict(colors=COLORS, line=dict(color='#000000', width=2)),
                rotation=90),  # Rotate the pie chart
            row=(i // cols) + 1, col=(i % cols) + 1)
        
    fig.update_traces(hoverinfo='value', textinfo='label+percent', textfont_size=20,
                    marker=dict(line=dict(color='#000000', width=2)))
    fig.update_layout(height=800, showlegend=True, plot_bgcolor='rgba(240,240,245,1)')

    fig.update_layout(paper_bgcolor='WhiteSmoke', 
                    title='Binary Features: Class Repartition',
                    title_font=dict(size=30),
                    title_x=0.5)
        
    fig.show()
    
    
def plot_target_pie(df, target):
    fig = make_subplots(rows=1, cols=1)

    # Get the value counts
    value_counts = df[target[0]].fillna('NaN').value_counts()

    fig.add_trace(
        go.Pie(labels=value_counts.index, values=value_counts.values, name=target[0],
            marker=dict(colors=COLORS2, line=dict(color='#000000', width=2)),
            rotation=90),) # Rotate the pie chart

    fig.update_traces(hoverinfo='value', textinfo='label+percent', textfont_size=20,
                    marker=dict(line=dict(color='#000000', width=2)))
    fig.update_layout(height=400, width=400, showlegend=True, plot_bgcolor='rgba(240,240,245,1)')

    fig.update_layout(paper_bgcolor='WhiteSmoke', 
                    title="Repartition of Response Labels",
                    title_font=dict(size=20),
                    title_x=0.5)
        
    fig.show()