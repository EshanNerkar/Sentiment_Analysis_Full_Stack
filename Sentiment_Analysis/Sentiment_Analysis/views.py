# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import CommentSerializer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # Import the joblib library to load your trained model
from .preprocess import preprocess
import nltk
import pandas as pd

nltk.download('stopwords')

class SentimentAnalysisView(APIView):
    def post(self, request):
        # Deserialize the input comment
        serializer = CommentSerializer(data=request.data)
        if serializer.is_valid():
            comment = serializer.validated_data['comment']

            # Preprocess the comment text (apply tokenization, lowercasing, etc.)
            preprocessed_comment = preprocess(comment)

            

            with open('sentiment_vectorizer.pkl','rb') as vectorizer_file:
                vectorizer = joblib.load(vectorizer_file)

            # Load your trained model using joblib
            with open('sentiment_model2.pkl', 'rb') as model_file:
                model = joblib.load(model_file)

            # Vectorize the preprocessed comment using your TF-IDF vectorizer
            
            tfidf_vector = vectorizer.transform([preprocessed_comment])  # Use transform here

            # Make a prediction using the loaded model
            sentiment_prediction = model.predict(tfidf_vector)

            # Define sentiment labels
            sentiment_labels = {
                'positive':0,
                'neutral':1,
                'negative':2
            }


            # Return the sentiment prediction as the API response
            response_data = {
                'comment': comment,
                'sentiment': sentiment_prediction[0] # Assuming it's a single prediction
            }

            return Response(response_data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
