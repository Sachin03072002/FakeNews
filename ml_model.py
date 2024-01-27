import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download the NLTK stopwords and punkt resources
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w\d\w', '', text)

    # Tokenization and removing stop words
    tokens = word_tokenize(text)
    tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]

    return ' '.join(tokens)

def preprocess_data(data_fake, data_true):
    data_fake['text'] = data_fake['text'].apply(preprocess_text)
    data_true['text'] = data_true['text'].apply(preprocess_text)

    data_fake['class'] = 0
    data_true['class'] = 1

    data = pd.concat([data_fake, data_true], axis=0)

    return data

def train_models(x_train, y_train):
    print("Before vectorization")
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    print("After vectorization")

    LR = LogisticRegression()
    LR.fit(xv_train, y_train)
    print("Logistic Regression trained")

    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)
    print("Decision Tree trained")

    GB = GradientBoostingClassifier(random_state=0)
    GB.fit(xv_train, y_train)
    print("Gradient Boosting trained")

    RF = RandomForestClassifier(random_state=0)
    RF.fit(xv_train, y_train)
    print("Random Forest trained")

    return LR, DT, GB, RF, vectorization
