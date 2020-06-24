from django.urls import path
from comment_sentiment.views import PredSentiment

app_name = 'comment_sentiment'
urlpatterns = [
    path('', PredSentiment.as_view(), name='PredSentiment')
]