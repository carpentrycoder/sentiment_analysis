from django.urls import path
from .views import InputDataListCreate , SessionDataView,ResumeParsedInfoView,CareerSuggestionsView,ShortlistingPredictionView,ResumeInsightsView,ResumeMarketValueView

urlpatterns = [
   path('data/', InputDataListCreate.as_view(), name='data-list-create'),
   path("session-data/", SessionDataView.as_view(), name="session-data"),
   path('resume-parsed-info/', ResumeParsedInfoView.as_view(), name='resume_parsed_info'),
   path('career-suggestions/', CareerSuggestionsView.as_view(), name='career_suggestions'),
   path("shortlisting-probability/", ShortlistingPredictionView.as_view(), name="shortlisting_probability"),
   path("resume-insights/", ResumeInsightsView.as_view(), name="resume_insights"),
   path("market-value/", ResumeMarketValueView.as_view(), name="market_value"),
]

