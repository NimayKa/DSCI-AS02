import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
current_directory = os.getcwd()
current_directory = current_directory + '/logisticRegression'
file_path = os.path.join(current_directory, 'shopping_behavior_new_updated.csv')

store_df = pd.read_csv('./shopping_behavior_new_updated.csv')
store_df.dropna(inplace=True)
print(store_df['Age Group'].unique())

output = store_df['Season']
features = store_df[['Frequency of Purchases','Age Group','Location','Item Purchased','Category']]

for feature in features:
    print (len(features[feature].unique()))


features = pd.get_dummies(features)
output, uniques = pd.factorize(output) 

x_train, x_test, y_train, y_test = train_test_split(
    features, output, test_size=.2)

# Define the hyperparameters grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 6, 9, 12, 90],
    'min_samples_split': [2, 90],
    'min_samples_leaf': [1, 90]
}

decision_tree_model = DecisionTreeClassifier()

grid_search = GridSearchCV(decision_tree_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

print("Hyperparameter Tuning Results:")
print()
results = grid_search.cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print("Accuracy:", mean_score)
    print("Hyperparameters:", params)
    print()
