import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
current_directory = os.getcwd()
current_directory = current_directory + '/logisticRegression'
file_path = os.path.join(current_directory, 'shopping_behavior_new_updated.csv')

store_df = pd.read_csv('./shopping_behavior_new_updated.csv')
store_df.dropna(inplace=True)

output = store_df['Season']

features = store_df[['Frequency of Purchases','Item Purchased', 'Age Group', 'Location', 'Category']]
output, uniques = pd.factorize(output)
features = pd.get_dummies(features) 

x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.3, random_state=42)

decision_tree_model = DecisionTreeClassifier(criterion= 'entropy',max_depth=12,min_samples_leaf=1,min_samples_split=3)

decision_tree_model.fit(x_train, y_train)

y_train_pred = decision_tree_model.predict(x_train)
y_test_pred = decision_tree_model.predict(x_test)

train_accuracy = accuracy_score(y_train_pred, y_train)
test_accuracy = accuracy_score(y_test_pred, y_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)