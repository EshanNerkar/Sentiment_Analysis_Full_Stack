import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')

def preprocess(text):
    #lowercasing
    lower_cased = str(text).lower()
    #tokeinzation
    tokenized = word_tokenize(lower_cased)
    #stopwords removal
    stop_word_rem = [word for word in tokenized if word not in stop_words]
    #lemmatization
    lemmatized_list = [lemmatizer.lemmatize(word) for word in stop_word_rem]
    preprocessed_text = " ".join(lemmatized_list)
    return preprocessed_text