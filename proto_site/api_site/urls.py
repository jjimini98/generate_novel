from django.urls import path
from api_site.views import HtmlView   


urlpatterns = [
    path('', HtmlView.as_view(), name='html_view'),
]
