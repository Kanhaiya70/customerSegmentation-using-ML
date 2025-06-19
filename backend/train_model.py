import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv('../data/customers.csv')
X = data[['age', 'income', 'score']]
y = data['segment']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

joblib.dump(clf, 'segment_classifier.pkl')
print('Model trained and saved as segment_classifier.pkl') 