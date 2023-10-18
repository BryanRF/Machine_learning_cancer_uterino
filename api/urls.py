from django.urls import path
from django.urls import re_path
from .views import cnn_view,svm_view
from django.conf import settings
from django.conf.urls.static import static
urlpatterns =[
    re_path(r'^analisis/(?P<filename>[^/]+)$', cnn_view.serve_analisis, name='serve_analisis'),
    path('tipo_imagen', cnn_view.as_view(), name='image_list'),
    path('tipo_imagen_svm', svm_view.as_view(), name='image_list')
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
