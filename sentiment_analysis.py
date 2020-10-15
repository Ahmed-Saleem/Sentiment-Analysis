import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

train_data=pd.read_csv('train.csv')
print(train_data.info)
print(train_data.describe())
print(train_data.isna().sum())
print(round(train_data.Is_Response.value_counts(normalize=True)*100,2))
round(train_data.Is_Response.value_counts(normalize=True)*100,2).plot(kind='bar')
train_data.drop(columns=['User_ID','Browser_Used','Device_Used'],inplace=True)
def text_clean(text):
    text=text.lower()
    text=re.sub('\[.*?\]', '', text)
    text=re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text=re.sub('\w*\d\w*', '', text)
    text=re.sub('[''""...]', '', text)
    text=re.sub('\n', '', text)
    return text
train_data['cleaned_reviews']=train_data.agg({'Description': lambda x: text_clean(x)})

X=train_data.cleaned_reviews
y=train_data.Is_Response
X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=225, test_size=0.1)

tvec= TfidfVectorizer()
clf= LogisticRegression(solver='lbfgs')

model= Pipeline([('vectorizer',tvec),('classifier',clf)])
model.fit(X_train,y_train)
y_pred= model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred, average='weighted'))
print('Recall score: ', recall_score(y_test, y_pred, average='weighted'))