import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

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
        # Calculate 'values' and 'pull_values' for each 'var'
        values = df[var].fillna('NaN').value_counts()
        pull_values = [0.1 if label == 1 else 0 for label in values.index]

        fig.add_trace(
            go.Pie(labels=df[var].unique(), values=values, name=var,
                marker=dict(colors=COLORS[2:], line=dict(color='#000000', width=2)),
                pull=pull_values,
                rotation=90),  # Rotate the pie chart
            row=(i // cols) + 1, col=(i % cols) + 1)
        
    fig.update_traces(hoverinfo='value', textinfo='label+percent', textfont_size=20,
                    marker=dict(line=dict(color='#000000', width=2)))
    fig.update_layout(height=800, showlegend=True, plot_bgcolor='rgba(240,240,245,1)',)

    fig.update_layout(paper_bgcolor='WhiteSmoke', 
                    title='Binary Features: Class Repartition',
                    title_font=dict(size=30),
                    title_x=0.5)
        
    fig.show()
    
    
def plot_target_pie(df, target):
    fig = make_subplots(rows=1, cols=1)

    # Get the value counts
    value_counts = df[target[0]].fillna('NaN').value_counts()
    pull_values = [0.2 if label == 1 else 0 for label in value_counts.index]

    fig.add_trace(
        go.Pie(labels=value_counts.index, values=value_counts.values, name=target[0],
            marker=dict(colors=COLORS2, line=dict(color='#000000', width=2)),
            pull=pull_values, # Use the 'pull' attribute here
            rotation=90),) # Rotate the pie chart

    fig.update_traces(hoverinfo='value', textinfo='label+percent', textfont_size=20,
                    marker=dict(line=dict(color='#000000', width=2)))
    fig.update_layout(height=400, width=400, showlegend=True, plot_bgcolor='rgba(240,240,245,1)')

    fig.update_layout(paper_bgcolor='WhiteSmoke', 
                    title="Repartition of Response Labels",
                    title_font=dict(size=20),
                    title_x=0.5)
        
    fig.show()
    
    
    
def plot_numerical_features(df):
    col_names = ['Age', 'Annual_Premium'] # 'Vintage']
    COLORS = px.colors.qualitative.Set3
    cols = 1
    rows = len(col_names)
    specs = [[{'type': 'xy'}] for _ in range(rows)]  # 'xy' for histograms
    nbins=[len(df['Age'].unique()), 40]

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=col_names, specs=specs)

    # Histogram for 'Age' with a bin for each unique age
    fig.add_trace(
        go.Histogram(x=df['Age'], 
                    nbinsx=len(df['Age'].unique()),  # number of bins equals the number of unique ages
                    marker=dict(color=COLORS[0], line=dict(color='#000000', width=2)),
                    hovertemplate = '<b>%{x}</b><br><i>Count</i>: %{y}'),
        row=1, col=1)

    fig.add_trace(
        go.Violin(x=df['Annual_Premium'], 
                line=dict(color='#000000', width=2),
                box_visible=True,  # show box plot inside the violin
                meanline_visible=True,  # show the mean line
                fillcolor=COLORS[4],
                hovertemplate = '<b>%{x}</b><br><i>Count</i>: %{y}'),
                row=2, col=1)

    fig.update_layout(height=700, showlegend=False, plot_bgcolor='rgba(240,240,245,1)')

    fig.update_layout(paper_bgcolor='WhiteSmoke', 
                    title='Numerical Features Distribution',
                    title_font=dict(size=30),
                    title_x=0.5
                    )
        
    fig.show()





def plot_categorical_features(df):
    categoricals = ['Policy_Sales_Channel', 'Region_Code', 'Vintage']
    cols = 1
    rows = len(categoricals)
    specs = [[{'type': 'xy'}] for _ in range(rows)]  # 'xy' for bar charts

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=categoricals, specs=specs)

    for i, var in enumerate(categoricals):
        counts = df[var].apply(str).value_counts().head(5)
        percentages = counts / len(df) * 100  # calculate percentages based on total count
        fig.add_trace(
            go.Bar(x=counts.index, y=counts.values, 
                marker=dict(color=COLORS[:len(counts)], line=dict(color='#000000', width=2)),
                hovertemplate = '<b>%{x}</b><br><i>Count</i>: %{y}<br><i>Percentage</i>: %{text}%',
                text=[f'{p:.2f}%' for p in percentages], ),
            row=i + 1, col=1)
        
    fig.update_layout(height=800, showlegend=False, plot_bgcolor='rgba(240,240,245,1)', bargap=0.1)

    fig.update_layout(paper_bgcolor='WhiteSmoke', 
                    title='Categorical Features Repartition',
                    title_font=dict(size=30),
                    title_x=0.5,
                    )
        
    fig.show()
    
    
    
    
    
def plot_correlation_heatmap(df:pd.DataFrame, title='',):
    """
    Generates a heatmap visualization of the correlation matrix for numeric columns in a DataFrame.   
    """
    display_col_names = ['Vehicle_Age', 'Vehicle_Damage', 'Gender','Vintage', 'Policy_Sales_Channel', 
           'Annual_Premium', 'Previously_Insured', 'Region_Code', 'Driving_License', 'Age']

    
    df = df.select_dtypes(include=[np.number])
    correlation_matrix = df.corr()
    heatmap = go.Heatmap(z=correlation_matrix, 
                         x=correlation_matrix.columns, 
                         y=correlation_matrix.columns, 
                         colorscale='RdBu_r') #GnBu
    columns_rename_dict = dict(zip(correlation_matrix.columns, display_col_names))
    correlation_matrix.rename(columns_rename_dict)
    # Affichage des coefficients de corrélation
    annotations = []
    for i, row in enumerate(correlation_matrix.values):
        for j, value in enumerate(row):
            color  = 'white' if value < -0.5 or value > 0.5 else 'black'
            annotations.append(
                go.layout.Annotation(text=str(round(value, 2)), 
                                                    x=correlation_matrix.columns[j], 
                                                    y=correlation_matrix.columns[i], 
                                                    showarrow=False, 
                                                    font=dict(color=color, size=16)))

    fig = go.Figure(data=heatmap)
    fig.update_layout(
        title=dict(
            text=f'Feature Correlations {title}',
            x=0.55,  # Center the title
            font=dict(size=30)),
        autosize=False,
        width=750,
        height=650,
        annotations=annotations,
        xaxis=dict(tickvals=correlation_matrix.columns, ticktext=display_col_names, tickfont=dict(size=14)),  # set the xtick labels to the new names
        yaxis=dict(tickvals=correlation_matrix.columns, ticktext=display_col_names, tickfont=dict(size=14), tickangle=-40),  # set the ytick labels to the new names
        paper_bgcolor='WhiteSmoke', 
    )

    fig.show()
    
    
    


def plot_qqplots(df:pd.DataFrame, title=''):
    """
    Plot QQ plots for numeric columns in a DataFrame.
    """
    
    COLORS = px.colors.qualitative.Set3
    n_samples = min(10000, df.shape[0])  # Sample 10,000 or the total number of rows, whichever is smaller
    df_sample = df.select_dtypes(include=[np.number]).sample(n=n_samples, random_state=1)

    # Create a subplot with 2 rows and 2 columns
    fig = make_subplots(rows=4, cols=2)

    # Loop over the first 4 columns of the DataFrame
    for i, col in enumerate(df_sample.columns[0:8]):
        # Calculate the theoretical quantiles and order them
        theoretical_quantiles = np.sort(stats.norm.ppf((np.arange(len(df_sample[col])) + 0.5) / len(df_sample[col])))
        
        # Calculate the sample quantiles and order them
        sample_quantiles = np.sort(df_sample[col])
        
        fig.add_trace(go.Scatter(x=theoretical_quantiles, 
                                 y=sample_quantiles, 
                                 mode='markers', 
                                 name=col, 
                                 marker=dict(color=COLORS[i % len(COLORS)])),  # Use color from palette
                                 row=(i//2)+1, 
                                 col=(i%2)+1)
    # Update layout
    fig.update_layout(paper_bgcolor='WhiteSmoke', 
                      height=1000, 
                      width=800, 
                      title_text=f"Observed quantiles vs quantiles of a normal distribution (QQ Plots) {title}")
    fig.show()
    
    
    
    
def plot_roc_curve(y_test, y_pred, title=''):
    """
    Plot ROC curve.

    """
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Create a Figure
    fig = go.Figure()

    # Add ROC curve to the Figure
    color = px.colors.qualitative.Set3[0]
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (area = {0:0.2f})'.format(roc_auc), line=dict(color=color)))

    # Add diagonal line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='black', dash='dash')))

    # Update layout
    fig.update_layout(title='ROC Curve: '+title, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', autosize=False, width=600, height=600, margin=dict(l=50, r=50, b=100, t=100, pad=4), plot_bgcolor='whitesmoke')
    fig.update_layout(height=700, width=1000)
    
    fig.show()