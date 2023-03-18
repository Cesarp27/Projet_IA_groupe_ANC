import plotly.express as px
# import plotly.graph_objects as go
# import plotly.offline as pyo
# import seaborn as sns
# import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt



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


# def plot_diagramme_de_dispersion(data, colonne):
#     # Graphique diagramme de dispersion
#     fig = px.scatter_matrix(data, dimensions=data.columns[:-1], color=colonne, width=1000, height=1200)
#     fig.update_traces(diagonal_visible=False)
#     fig.update_layout(title="Diagramme de dispersion des paires de variables")
#     return fig



def plot_diagramme_de_dispersion(data, colonne):
    # Graphique diagramme de dispersion
    g = sns.pairplot(data, palette="Set2", diag_kind='kde', hue= colonne)
    plt.subplots_adjust(top=0.95)
    plt.suptitle("Diagramme de dispersion des paires de variables", fontsize=16)

    return g