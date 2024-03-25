import pandas as pd 
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

current_directory = os.getcwd()
file_path = os.path.join(current_directory, 'Pickle/gb.pickle')
file_path2 = os.path.join(current_directory, 'Pickle/gb_output.pickle')

df = pd.read_csv('shopping_behavior_new_updated.csv')
df.dropna(inplace=True)

output = df['Subscription Status']
output,uniques = pd.factorize(output)

features = df[['Gender', 'Item Purchased', 'Category',
                     'Discount Applied', 'Payment Method', 'Age Group', 'Frequency of Purchases']]
num_features =  df[['Age','Purchase Amount (USD)','Review Rating','Previous Purchases']]

encoders = {}
for feature in features:
    encoder = LabelEncoder()
    encoded_values = encoder.fit_transform(features[feature])
    features.loc[:, feature] = encoded_values
    encoders[feature] = encoder

num_features = pd.get_dummies(num_features)  
features = pd.concat([features, num_features], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.2, random_state=30)

gb_classifier = GradientBoostingClassifier(random_state=30)

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# gridsearchcv for hyperparameter tuning
grid_search = GridSearchCV(estimator=gb_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Iterate over each parameter combination and print the parameters along with their corresponding values
print("Grid search results:")
for i in range(len(grid_search.cv_results_['params'])):
    print("Parameters:", grid_search.cv_results_['params'][i])
    print("Mean Test Score:", grid_search.cv_results_['mean_test_score'][i])
    print("Rank:", grid_search.cv_results_['rank_test_score'][i])
    print()
best_params = grid_search.best_params_
print (best_params)

# After fitting the GridSearchCV object
grid_search.fit(X_train, y_train)

gb_classifier = GradientBoostingClassifier(**best_params, random_state=30)
gb_classifier.fit(X_train, y_train)

y_pred = gb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

y_train_pred = gb_classifier.predict(X_train)
y_test_pred = gb_classifier.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

with open(file_path, 'wb') as gb_pickle:
    pickle.dump(gb_classifier, gb_pickle)
    gb_pickle.close()

# passing the mapping values
with open(file_path2, 'wb') as output_pickle:
    pickle.dump(output, output_pickle) 
    output_pickle.close()

fig, ax = plt.subplots()
ax = sns.barplot(x=gb_classifier.feature_importances_, y=features.columns)
plt.title('Important Features that could predict user subscription')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()

fig.savefig('Pickle/gb_feature_importance.png')