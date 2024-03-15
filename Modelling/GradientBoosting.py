import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

pd.set_option('display.max_columns', None)
current_directory = os.getcwd()
current_directory = current_directory + '/gradientBoosting'
file_path = os.path.join(current_directory, 'shopping_behavior_new_updated.csv')

store_df = pd.read_csv('./shopping_behavior_new_updated.csv')
store_df.dropna(inplace=True)

output = store_df[['Subscription Status']]
output = np.ravel(output)

features = store_df[['Age', 'Gender', 'Item Purchased', 'Category',
                     'Purchase Amount (USD)', 'Location', 'Size', 'Color',
                     'Season', 'Review Rating', 'Frequency of Purchases', 'Shipping Type',
                     'Discount Applied', 'Promo Code Used', 'Previous Purchases', 'Payment Method', 'Age Group']]

features = pd.get_dummies(features)

x_train, x_test, y_train, y_test = train_test_split(
    features, output, test_size=.2)

gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)

y_pred = gbc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
