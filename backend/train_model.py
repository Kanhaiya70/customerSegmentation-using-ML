import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load enhanced dataset
data = pd.read_csv('../data/enhanced_customers.csv')

# Use new labels
X = data[['age', 'income', 'score']]
y = data['segment']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

joblib.dump(clf, 'segment_classifier.pkl')
joblib.dump(clf, '../segment_classifier.pkl')
print('✅ Model retrained with logic-based segments.')