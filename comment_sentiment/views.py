from django import forms
from django.http import HttpResponse
from django.shortcuts import render

from .model import predict_sentiment
from django.views import View


class MyForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea)
    # rating = forms.IntegerField(name='rating', verbose_name='rating', default=10)
    # is_pos = forms.BooleanField(name='is_pos', verbose_name='is_pos', default=True)


class PredSentiment(View):
    def post(self, request, *args, **kwargs):
        form = MyForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data.get('text')
            rating, is_pos = predict_sentiment(text)
            return render(request, 'result.html', {'text': text,
                                                   'rating': rating,
                                                   'is_pos': is_pos})
        else:
            raise forms.ValidationError('Form is incorectly filled')

    def get(self, request, *args, **kwargs):
        form = MyForm()
        return render(request, 'form.html', {'form': form})