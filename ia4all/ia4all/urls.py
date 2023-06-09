"""ia4all URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from authentification.views import inscription, connexion, deconnexion, index, suppression, classification, regression, clustering

from django.conf.urls.static import static
from django.conf import settings

from django.urls import include


urlpatterns = [
    path("admin/", admin.site.urls),
    path("inscription", inscription, name="inscription"),
    path("connexion", connexion, name="connexion"),
    path("deconnexion", deconnexion, name="deconnexion"),
    path("index", index, name="index"),
    path("classification", classification, name="classification"),
    path("regression", regression, name="regression"),
    path("clustering", clustering, name="clustering"),
    path('', include("fileupload.urls")),
    path("suppression/<int:id>", suppression, name="suppression"),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)




# # only in development
# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)