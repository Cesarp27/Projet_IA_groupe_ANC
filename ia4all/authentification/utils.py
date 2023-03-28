
import plotly.express as px
# import plotly.graph_objects as go
# import plotly.offline as pyo
# import seaborn as sns
# import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import plotly.graph_objs as go
from plotly.subplots import make_subplots



def plot_histograme(data, colonne):
    # Graphique histogramme
    fig = px.histogram(data, x=data.columns[0], color=colonne, nbins=30)

    fig.update_layout(
        title_text="Nombre d'individus par valeur unique dans la colonne target",
        xaxis_title=data.columns[0],
        yaxis_title="Nombre d'individus"
    )
    return fig


def plot_correlation(df_numerique):
    # Gráfico de correlación
    fig = px.imshow(df_numerique.corr(),
                    labels=dict(x="Caractéristiques", y="Caractéristiques", color="Corrélation"),
                    x=df_numerique.columns,
                    y=df_numerique.columns,
                    color_continuous_scale='RdBu',
                    zmin=-1,
                    zmax=1
                   )
    
    fig.update_layout(
        title=dict(
            text="Corrélation entre les caractéristiques",
            x=0.5,
            font=dict(size=18)
        ),    
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(df_numerique.columns))),
            ticktext=list(df_numerique.columns),
            side="bottom",
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            autorange="reversed",
            showgrid=False,
            zeroline=False
        )
    )

    return fig


def plot_feature_importances(df_numerique, model, fig):
    # Graphique de l'importance des caractéristiques avec les noms des colonnes
    feature_names = list(df_numerique.columns)

    importances = model.feature_importances_
    feature_importances = dict(zip(feature_names, importances))
    sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    sorted_features, sorted_importances = zip(*sorted_importances)

    fig2 = px.bar(y=list(sorted_features), x=list(sorted_importances), orientation='h')
    fig2.update_layout(title='Importances des caractéristiques pour la prédiction', yaxis_title='Caractéristiques', xaxis_title='Importances')
    fig2.update_yaxes(title_standoff=0)
    fig2.update_layout(title_x=0.5)

    return fig2


# def scatter_plot(y_test, y_pred):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='data'))
#     # fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='lines', name='regressor'))
#     fig.update_layout(title='Scatter plot with GradientBoostingRegressor', xaxis_title='True values', yaxis_title='Predictions')
#     return fig


# def plot_prediction_error(y_true, y_pred):
#     errors = np.abs(y_true - y_pred)
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers',
#                              name='Prediction Error'))
#     fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[np.mean(y_pred), np.mean(y_pred)],
#                              mode='lines', name='Mean Prediction'))
#     fig.update_layout(title='Prediction Error Plot', xaxis_title='True Value', yaxis_title='Prediction',
#                       legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)'))
#     return fig

def plot_prediction_error(y_test, y_pred):
    # Calculate prediction error
    error = y_test - y_pred
    
    
    # Create scatter plot for y_test and y_pred
    trace1 = go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='original vs Predicted',
        marker=dict(
            color='blue'
        )
    )

    trace2 = go.Scatter(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                   mode='lines', name='True value'))
    
    # Create layout
    layout = go.Layout(
        title=dict(text='Prediction Error', x=0.5),
        xaxis=dict(
            title='True'
        ),
        yaxis=dict(
            title='Predicted Value'
        ),
        hovermode='closest'
    )
    
    # Combine traces and layout into figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    
    return fig


def plot_diagramme_de_dispersion(data, colonne):
    # Graphique diagramme de dispersion
    g = sns.pairplot(data, palette="Set2", diag_kind='kde', hue= colonne)
    plt.subplots_adjust(top=0.95)
    plt.suptitle("Diagramme de dispersion des paires de variables", fontsize=16)

    return g