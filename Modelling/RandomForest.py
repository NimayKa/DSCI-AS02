import pandas as pd 
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

current_directory = os.getcwd()
file_path = os.path.join(current_directory, 'Pickle/rf.pickle')
file_path2 = os.path.join(current_directory, 'Pickle/rf_output.pickle')

df = pd.read_csv('shopping_behavior_new_updated.csv')

features = df[['Age', 'Gender', 'Item Purchased', 'Category',
               'Purchase Amount (USD)', 'Review Rating','Previous Purchases',
                'Discount Applied', 'Payment Method', 'Frequency of Purchases']]

output = df['Subscription Status']

label_encoder = LabelEncoder()

for column in features:
    if features[column].dtype == 'object':
        features[column] = label_encoder.fit_transform(features[column])

X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.2, random_state=30)

rf_classifier = RandomForestClassifier(random_state=30)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

rf_classifier = RandomForestClassifier(**best_params, random_state=30)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

y_train_pred = rf_classifier.predict(X_train)
y_test_pred = rf_classifier.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

with open(file_path, 'wb') as rf_pickle:
	pickle.dump(rf_classifier, rf_pickle) 
	rf_pickle.close() 

# passing the mapping values
with open(file_path2, 'wb') as output_pickle:
	pickle.dump(output, output_pickle) 
	output_pickle.close() 

fig, ax = plt.subplots() 
ax = sns.barplot(x=rf_classifier.feature_importances_, y=features.columns) 
plt.title('Important Features that could predict user subcription') 
plt.xlabel('Importance') 
plt.ylabel('Feature') 
plt.tight_layout() 

fig.savefig('Pickle/rf_feature_importance.png') 