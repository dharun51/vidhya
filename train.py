import pandas as pd
from sklearn.feature_extraction.text import Tfidf vectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
file_name="untitled spreadsheet.xlsx"
df=pd.read_excel(file_name)
print(df)
X=df['Question']
y=df['Answer']
model=Pipeline([('tfidf',TfidfVectorizer()),('classifier',multinomialNB())])
model.fit(X,y)
joblib.dump(model,'untitled spreadsheet_model.pkl')
print("model saved successfully")