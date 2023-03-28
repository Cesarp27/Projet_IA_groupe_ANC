from django.shortcuts import render, redirect, HttpResponse 
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from authentification.models import Utilisateur
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.express as px
from plotly.tools import mpl_to_plotly
# import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
# import numpy as np
from sklearn.cluster import DBSCAN, KMeans
# from sklearn import metrics
import pandas as pd

import pandas as pd
#import matplotlib.pyplot as plt 
import numpy as np
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

# from django.core.files.storage import FileSystemStorage

from .models import FilesUpload 

from .utils import plot_histograme, plot_correlation, plot_feature_importances, plot_diagramme_de_dispersion, plot_prediction_error
# import io
import base64 #, urllib

from io import BytesIO




# Fonction de nettoyage et entraînement du modèle
# il nous faut: #df, #target, X_numérique, model_entraine et score
def machine_learning(df_utilisateur, target, model):
    df_utilisateur = df_utilisateur.dropna()

    lb_encod = LabelEncoder()

    if df_utilisateur[target].dtype == 'object':
        y = lb_encod.fit_transform(df_utilisateur[target])
    else:
        y = df_utilisateur[target].values
        
    X_original = df_utilisateur.drop(target, axis=1)
    X_numerique = X_original.select_dtypes(include=['float64', 'int64'])#.values
    X_scaler = StandardScaler().fit_transform(X_numerique)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=2, stratify=y)
    model_entraine = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    
    # Results
    machine_learning_results = []
    machine_learning_results.append(X_numerique)
    machine_learning_results.append(model_entraine)    
    machine_learning_results.append(score.round(4))
    
    return(machine_learning_results)    

def machine_learning_regression(df_utilisateur, target, model):
    df_utilisateur = df_utilisateur.dropna()

    lb_encod = LabelEncoder()

    # if df_utilisateur[target].dtype == 'object':
    #     y = lb_encod.fit_transform(df_utilisateur[target])
    #else:
    y = df_utilisateur[target].values

    X_original = df_utilisateur.drop(target, axis=1)
    X_numerique = X_original.select_dtypes(include=['float64', 'int64'])  # .values
    X_scaler = StandardScaler().fit_transform(X_numerique)

    X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=2)
    model_entraine = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Results
    machine_learning_results_regressor = []
    machine_learning_results_regressor.append(X_numerique)
    machine_learning_results_regressor.append(model_entraine)
    machine_learning_results_regressor.append(rmse.round(4))
    machine_learning_results_regressor.append(mae.round(4))
    machine_learning_results_regressor.append(y_test)
    machine_learning_results_regressor.append(y_pred)

    return (machine_learning_results_regressor)


# Mes graphiques
fig = go.Figure()
scatter = go.Scatter(x=[0,1,2,3], y=[0,1,2,3],
                     mode='lines', name='test',
                     opacity=0.8, marker_color='green')
fig.add_trace(scatter)
plt_div = plot(fig, output_type='div')

df2 = px.data.iris() # iris is a pandas DataFrame
fig2 = px.scatter(df2, x="sepal_width", y="sepal_length", title="Scatter plot")
graph2 = plot(fig2, output_type='div')


df3 = px.data.tips()
fig3 = px.box(df3, x="time", y="total_bill", title="Boîte à moustache")
graph3 = plot(fig3, output_type='div')

z = [[.1, .3, .5, .7, .9],
     [1, .8, .6, .4, .2],
     [.2, 0, .5, .7, .9],
     [.9, .8, .4, .2, 0],
     [.3, .4, .5, .7, 1]]

fig4 = px.imshow(z, text_auto=True)
graph4 = plot(fig4, output_type='div')

# Clustering
# DBScan

dfPenguins = pd.read_csv("C:/Users/cesar/Documents/Dos/20230306_Patrick_Projet_IA/iaouverte/ia4all/authentification/penguins.csv")
X = dfPenguins[dfPenguins.describe().columns].dropna().values

X = StandardScaler().fit_transform(X)

db = DBSCAN().fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
#Homogeneity = metrics.homogeneity_score(labels_true, labels)



def inscription(request):
    message = ""
    if request.method == "POST":
        if request.POST["motdepasse1"] == request.POST["motdepasse2"]:
            modelUtilisaleur = get_user_model()
            identifiant = request.POST["identifiant"]
            motdepasse = request.POST["motdepasse1"]
            utilisateur = modelUtilisaleur.objects.create_user(username=identifiant,
                                                       password=motdepasse)
            return redirect("connexion")
        else:
            message = "⚠️ Les deux mots de passe ne concordent pas ⚠️"
    return render(request, "inscription.html", {"message" : message})

def connexion(request):
    # La méthode POSt est utilisé quand des infos
    # sont envoyées au back-end
    # Autrement dit, on a appuyé sur le bouton
    # submit
    message = ""
    if request.method == "POST":
        identifiant = request.POST["identifiant"]
        motdepasse = request.POST["motdepasse"]
        utilisateur = authenticate(username = identifiant,
                                   password = motdepasse)
        if utilisateur is not None:
            login(request, utilisateur)
            return redirect("index")
        else:
            message = "Identifiant ou mot de passe incorrect"
            return render(request, "connexion.html", {"message": message})
    # Notre else signifie qu'on vient d'arriver
    # sur la page, on a pas encore appuyé sur le
    # bouton submit
    else:
        return render(request, "connexion.html")

def deconnexion(request):
    logout(request)
    return redirect("connexion")

def suppression(request, id):
    utilisateur = Utilisateur.objects.get(id=id)
    logout(request)
    utilisateur.delete()
    return redirect("connexion")

@login_required
def index(request):
    # if request.method == "POST" and request.POST["fichier"]:
    #     # print(request.POST["fichier"], request.POST)
    #     file = request.FILES['fichier']
    #     file_name = default_storage.save(file.name, file)
    #     #fs = FileSystemStorage()
    #     #filename = fs.save(myfile.name, myfile)
        
    #     return render(request, "index.html", context)
    print("User name:", request.user.username) 
    print("User id:", request.user.id) 
    print("type User id:", type(request.user.id))
    
    if request.method == "POST":
        # if the post request has a file under the input name 'file', then save the file.
        request_file = request.FILES['file'] if 'file' in request.FILES else None
        if request_file:
            # save attached file
            document = FilesUpload.objects.create(userid = request.user.id, file=request_file) 
            document.save() 
            return HttpResponse('<script>alert("Votre fichier a été chargé avec succès"); window.location.replace("/index");</script>') 

    
    context = {"n_clusters_" : n_clusters_,
               "n_noise_" : n_noise_,
               "graphique": plt_div,
               "graph2": graph2,
               "graph3": graph3,
               "graph4": graph4
               }
    return render(request, "index.html", context)




@login_required
def classification(request):
    files = FilesUpload.objects.filter(userid=request.user.id)
    df = None
    
    
    
    if request.method == 'POST' and 'file' in request.POST:
        # selected_file_id = request.POST['file']
        selected_file_id = request.POST.get('file')
        request.session['selected_file_id'] = selected_file_id # "selected_file_id" est enregistré dans la session de l'utilisateur
        selected_file = FilesUpload.objects.get(pk=selected_file_id)
        fichier = selected_file.file
        request.session['fichier'] = str(fichier) # "fichier" est enregistré dans la session de l'utilisateur
        
        print("=======>"+str(fichier))


        # Lire les données du fichier CSV et les convertir en un dataframe
        try:
            data = pd.read_csv(fichier)
            df = pd.DataFrame(data)
            print("df.shape ----> "+str(df.shape))
            request.session['df'] = df.to_dict()  # le dataframe est enregistré dans la session de l'utilisateur
            
            context = {"fichier" : fichier, 
            'df': df,
            "files": files}
            
        except Exception as e:
            context = {"error_message": f"Veuillez sélectionner un fichier CSV valide - Erreur: {str(e)}", "files": files}
            return render(request, "classification.html", context)
        



        
    if request.method == 'POST' and 'colonne' in request.POST:
        # le deuxième formulaire a été envoyé
        selected_column = request.POST['colonne']
        print(selected_column)
        request.session['selected_column'] = selected_column
        
        fichier = request.session.get('fichier')
        selected_file_id = request.session.get('selected_file_id')
        
        df_dict = request.session.get('df') # récupérer le df sous la forme d'un dictionnaire à partir de la session de l'utilisateur
        if df_dict is not None:
            df = pd.DataFrame.from_dict(df_dict)

        context = {"fichier" : fichier, 
            'df': df,
            "files": files,
            "selected_column" : selected_column}
        
            
    if request.method == 'POST' and 'rfc' in request.POST:
        fichier = request.session.get('fichier')
        selected_column = request.session.get('selected_column')
        df_dict = request.session.get('df') # récupérer le df sous la forme d'un dictionnaire à partir de la session de l'utilisateur
        if df_dict is not None:
            df = pd.DataFrame.from_dict(df_dict)
        selected_column = request.session.get('selected_column') 
        
        df = df.dropna()
        
        X_numerique, model_entraine, score = machine_learning(df, selected_column, RandomForestClassifier())
        
        fig = plot_histograme(df, selected_column)
        graph = plot(fig, output_type='div')
        
        fig2 = plot_correlation(X_numerique)
        graph2 = plot(fig2, output_type='div')
        
        graph3 = plot(plot_feature_importances(X_numerique, model_entraine, fig3), output_type='div')
        
        # Utilise la fonction plot_diagramme_de_dispersion pour générer le graphique
        g = plot_diagramme_de_dispersion(df, selected_column)
        buffer = BytesIO()
        g.fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graph4 = base64.b64encode(image_png)
        graph4 = graph4.decode('utf-8')
        
        
        context = {'files': files,
                'fichier' : fichier,
                'df': df,
                # 'selected_file_id': selected_file_id,
                'selected_column': selected_column,
                'graph' : graph,
                'graph2' : graph2,
                'graph3': graph3,
                'graph4' : graph4,
                "score" : score
                }
            
        
        
        
    elif request.method == 'POST' and 'adac' in request.POST:
        fichier = request.session.get('fichier')
        selected_column = request.session.get('selected_column')
        df_dict = request.session.get('df') # récupérer le df sous la forme d'un dictionnaire à partir de la session de l'utilisateur
        if df_dict is not None:
            df = pd.DataFrame.from_dict(df_dict)
        selected_column = request.session.get('selected_column') 
        
        df = df.dropna()
        
        X_numerique, model_entraine, score = machine_learning(df, selected_column, AdaBoostClassifier())
        
        fig = plot_histograme(df, selected_column)
        graph = plot(fig, output_type='div')
        
        fig2 = plot_correlation(X_numerique)
        graph2 = plot(fig2, output_type='div')
        
        graph3 = plot(plot_feature_importances(X_numerique, model_entraine, fig3), output_type='div')
        
        # Utilise la fonction plot_diagramme_de_dispersion pour générer le graphique
        g = plot_diagramme_de_dispersion(df, selected_column)
        buffer = BytesIO()
        g.fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graph4 = base64.b64encode(image_png)
        graph4 = graph4.decode('utf-8')
        
        
        context = {'files': files,
                'fichier' : fichier,
                'df': df,
                # 'selected_file_id': selected_file_id,
                'selected_column': selected_column,
                'graph' : graph,
                'graph2' : graph2,
                'graph3': graph3,
                'graph4' : graph4,
                "score" : score
                }
        
    elif request.method == 'POST' and 'gbc' in request.POST:
        fichier = request.session.get('fichier')
        selected_column = request.session.get('selected_column')
        df_dict = request.session.get('df') # récupérer le df sous la forme d'un dictionnaire à partir de la session de l'utilisateur
        if df_dict is not None:
            df = pd.DataFrame.from_dict(df_dict)
        selected_column = request.session.get('selected_column') 
        
        df = df.dropna()        
        
        X_numerique, model_entraine, score = machine_learning(df, selected_column, GradientBoostingClassifier())
        
        fig = plot_histograme(df, selected_column)
        graph = plot(fig, output_type='div')
        
        fig2 = plot_correlation(X_numerique)
        graph2 = plot(fig2, output_type='div')
        
        graph3 = plot(plot_feature_importances(X_numerique, model_entraine, fig3), output_type='div')
        
        # Utilise la fonction plot_diagramme_de_dispersion pour générer le graphique
        g = plot_diagramme_de_dispersion(df, selected_column)
        buffer = BytesIO()
        g.fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graph4 = base64.b64encode(image_png)
        graph4 = graph4.decode('utf-8')
        
        
        context = {'files': files,
                'fichier' : fichier,
                'df': df,
                # 'selected_file_id': selected_file_id,
                'selected_column': selected_column,
                'graph' : graph,
                'graph2' : graph2,
                'graph3': graph3,
                'graph4' : graph4,
                "score" : score
                }


            

            
            
            

    
    elif request.method == 'GET':
        context = {"files": files,'df': df}
        
        
        #selected_file_id = request.session.get('selected_file_id')
        # if request.method == 'POST':
        #     context['selected_file_id'] = selected_file_id
        #     context['fichier'] = fichier
    
    
    return render(request, "classification.html", context)



@login_required
def regression(request):
    files = FilesUpload.objects.filter(userid=request.user.id)
    df = None

    if request.method == 'POST' and 'file' in request.POST:
        # selected_file_id = request.POST['file']
        selected_file_id = request.POST.get('file')
        request.session[
            'selected_file_id'] = selected_file_id  # "selected_file_id" est enregistré dans la session de l'utilisateur
        selected_file = FilesUpload.objects.get(pk=selected_file_id)
        fichier = selected_file.file
        request.session['fichier'] = str(fichier)  # "fichier" est enregistré dans la session de l'utilisateur

        print("=======>" + str(fichier))

        # Lire les données du fichier CSV et les convertir en un dataframe
        try:
            data = pd.read_csv(fichier)
            df = pd.DataFrame(data)
            print("df.shape ----> " + str(df.shape))
            request.session['df'] = df.to_dict()  # le dataframe est enregistré dans la session de l'utilisateur

            context = {"fichier": fichier,
                       'df': df,
                       "files": files}

        except Exception as e:
            context = {"error_message": f"Veuillez sélectionner un fichier CSV valide - Erreur: {str(e)}",
                       "files": files}
            return render(request, "regression.html", context)

    if request.method == 'POST' and 'colonne' in request.POST:
        # le deuxième formulaire a été envoyé
        selected_column = request.POST['colonne']
        print(selected_column)
        request.session['selected_column'] = selected_column

        fichier = request.session.get('fichier')
        selected_file_id = request.session.get('selected_file_id')

        df_dict = request.session.get(
            'df')  # récupérer le df sous la forme d'un dictionnaire à partir de la session de l'utilisateur
        if df_dict is not None:
            df = pd.DataFrame.from_dict(df_dict)

        context = {"fichier": fichier,
                   'df': df,
                   "files": files,
                   "selected_column": selected_column}

    if request.method == 'POST' and 'rfr' in request.POST:
        fichier = request.session.get('fichier')
        selected_column = request.session.get('selected_column')
        df_dict = request.session.get(
            'df')  # récupérer le df sous la forme d'un dictionnaire à partir de la session de l'utilisateur
        if df_dict is not None:
            df = pd.DataFrame.from_dict(df_dict)
        selected_column = request.session.get('selected_column')

        df = df.dropna()

        X_numerique, model_entraine, rmse, mae, y_test, y_pred = machine_learning_regression(df, selected_column, RandomForestRegressor())

        fig = plot_histograme(df, selected_column)
        graph = plot(fig, output_type='div')

        fig2 = plot_correlation(X_numerique)
        graph2 = plot(fig2, output_type='div')

        graph3 = plot(plot_feature_importances(X_numerique, model_entraine, fig3), output_type='div')

        fig4 = plot_prediction_error(y_test, y_pred)
        graph4 = plot(fig4, output_type='div')

        context = {'files': files,
                    'fichier': fichier,
                    'df': df,
                    'selected_column': selected_column,
                    'graph': graph,
                    'graph2': graph2,
                    'graph3': graph3,
                    'graph4': graph4,
                    "rmse" : rmse,
                    "mae" : mae
                   }




    elif request.method == 'POST' and 'adar' in request.POST:
        fichier = request.session.get('fichier')
        selected_column = request.session.get('selected_column')
        df_dict = request.session.get(
            'df')  # récupérer le df sous la forme d'un dictionnaire à partir de la session de l'utilisateur
        if df_dict is not None:
            df = pd.DataFrame.from_dict(df_dict)
        selected_column = request.session.get('selected_column')

        df = df.dropna()

        X_numerique, model_entraine, rmse, mae, y_test, y_pred = machine_learning_regression(df, selected_column, AdaBoostRegressor())

        fig = plot_histograme(df, selected_column)
        graph = plot(fig, output_type='div')

        fig2 = plot_correlation(X_numerique)
        graph2 = plot(fig2, output_type='div')

        graph3 = plot(plot_feature_importances(X_numerique, model_entraine, fig3), output_type='div')

        fig4 = plot_prediction_error(y_test, y_pred)
        graph4 = plot(fig4, output_type='div')

        context = {'files': files,
                    'fichier': fichier,
                    'df': df,
                    'selected_column': selected_column,
                    'graph': graph,
                    'graph2': graph2,
                    'graph3': graph3,
                    'graph4': graph4,
                    "rmse": rmse,
                    "mae": mae
                   }

    elif request.method == 'POST' and 'gbr' in request.POST:
        fichier = request.session.get('fichier')
        selected_column = request.session.get('selected_column')
        df_dict = request.session.get(
            'df')  # récupérer le df sous la forme d'un dictionnaire à partir de la session de l'utilisateur
        if df_dict is not None:
            df = pd.DataFrame.from_dict(df_dict)
        selected_column = request.session.get('selected_column')

        df = df.dropna()

        X_numerique, model_entraine, rmse, mae, y_test, y_pred = machine_learning_regression(df, selected_column, GradientBoostingRegressor())

        fig = plot_histograme(df, selected_column)
        graph = plot(fig, output_type='div')

        fig2 = plot_correlation(X_numerique)
        graph2 = plot(fig2, output_type='div')

        graph3 = plot(plot_feature_importances(X_numerique, model_entraine, fig3), output_type='div')

        fig4 = plot_prediction_error(y_test, y_pred)
        graph4 = plot(fig4, output_type='div')


        context = {'files': files,
                    'fichier': fichier,
                    'df': df,
                    'selected_column': selected_column,
                    'graph': graph,
                    'graph2': graph2,
                    'graph3': graph3,
                    'graph4': graph4,
                    "rmse": rmse,
                    "mae" : mae
                   }


    elif request.method == 'GET':
        context = {"files": files, 'df': df}

        # selected_file_id = request.session.get('selected_file_id')
        # if request.method == 'POST':
        #     context['selected_file_id'] = selected_file_id
        #     context['fichier'] = fichier

    return render(request, "regression.html", context)


@login_required
def clustering(request):
    files = FilesUpload.objects.filter(userid=request.user.id)
    df = None
    
    
    
    if request.method == 'POST' and 'file' in request.POST:
        # selected_file_id = request.POST['file']
        selected_file_id = request.POST.get('file')
        request.session['selected_file_id'] = selected_file_id # "selected_file_id" est enregistré dans la session de l'utilisateur
        selected_file = FilesUpload.objects.get(pk=selected_file_id)
        fichier = selected_file.file
        request.session['fichier'] = str(fichier) # "fichier" est enregistré dans la session de l'utilisateur
        
        print("== le file est ====>"+str(fichier))


        # Lire les données du fichier CSV et les convertir en un dataframe
        try:
            data = pd.read_csv(fichier)
            df = pd.DataFrame(data)
            print("df.shape of our data ----> "+str(df.shape))
            request.session['df'] = df.to_dict()  # le dataframe est enregistré dans la session de l'utilisateur
            
        except Exception as e:
            context = {"error_message": f"Veuillez sélectionner un fichier CSV valide - Erreur: {str(e)}", "files": files}
            return render(request, "clustering.html", context)

        
    if request.method == 'POST' :
        # le deuxième formulaire a été envoyé
  
        fichier = request.session.get('fichier')
        selected_file_id = request.session.get('selected_file_id')
        
        df_dict = request.session.get('df') # récupérer le df sous la forme d'un dictionnaire à partir de la session de l'utilisateur
        if df_dict is not None:
            df = pd.DataFrame.from_dict(df_dict)

        
        
        if df is not None and not df.empty:
           
            df = df.dropna()
            print('cest ici notre df ---->>>>', df)
            """ y = df[selected_column].values
            print(y) """
        
            X = df[df.describe().columns].values

            X = StandardScaler().fit_transform(X)

            db = DBSCAN().fit(X)
            labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            # db['labels'] = db.predict(X), ,color="labels"
            df_num = df.select_dtypes(exclude=['object'])

            model_kmean3 = KMeans(n_clusters= n_clusters_, random_state=1, n_init = 5).fit(df_num)
            model_kmean3

            Clusters = model_kmean3.fit_predict(df_num)

            centers = model_kmean3.cluster_centers_

            df["Clusters"] = Clusters

            
            #Homogeneity = metrics.homogeneity_score(labels_true, labels)
            #print(df)

            # df2 = px.data.iris() # iris is a pandas DataFrame
            df_num = df.select_dtypes(exclude=['object'])
            fig2 = px.scatter(df, x=df_num.columns[0], y=df_num.columns[1],color="Clusters" , title="Scatter plot")
            # fig2 =  plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
            graph2 = plot(fig2, output_type='div')
            
        
            
            context = {"n_clusters_" : n_clusters_,
                    "n_noise_" : n_noise_,
                    # "graphique": plt_div,
                    "graph2": graph2,
                    # "graph3": graph3,
                    # "graph4": graph4,
                    'files': files,
                    'fichier' : fichier,
                    'df': df,
                    'selected_file_id': selected_file_id,
                    }
                   
    
    else:
        context = {"files": files,
                   'df': df}
        
        #selected_file_id = request.session.get('selected_file_id')
        if request.method == 'POST':
            context['selected_file_id'] = selected_file_id
            context['fichier'] = fichier
    
    
    return render(request, "clustering.html", context)

