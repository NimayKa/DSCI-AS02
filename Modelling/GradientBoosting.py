import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)

current_directory = os.getcwd()
file_path = os.path.join(current_directory, 'shopping_behavior_new_updated.csv')
store_df = pd.read_csv(file_path)
store_df.dropna(inplace=True)

output = store_df['Subscription Status']
features = store_df[['Age', 'Gender', 'Item Purchased', 'Category',
                     'Purchase Amount (USD)', 'Review Rating','Previous Purchases',
                     'Discount Applied', 'Payment Method', 'Age Group', 'Frequency of Purchases']]
features = pd.get_dummies(features) 

x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.3, random_state=42)

gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gradient_boosting_model.fit(x_train, y_train)

y_train_pred = gradient_boosting_model.predict(x_train)
y_test_pred = gradient_boosting_model.predict(x_test)
train_accuracy = accuracy_score(y_train_pred, y_train)
test_accuracy = accuracy_score(y_test_pred, y_test)
train_classification_report = classification_report(y_train, y_train_pred)
test_classification_report = classification_report(y_test, y_test_pred)

file_path_model = os.path.join(current_directory, 'Pickle/gbs.pickle')
file_path_predictions = os.path.join(current_directory, 'Pickle/gbs_predictions.pickle')

with open(file_path_model, 'wb') as rf_pickle:
    pickle.dump(gradient_boosting_model, rf_pickle) 

with open(file_path_predictions, 'wb') as output_pickle:
    pickle.dump(y_test_pred, output_pickle) 

fig, ax = plt.subplots() 
ax = sns.barplot(x=gradient_boosting_model.feature_importances_, y=features.columns) 
plt.title('Which features are the most important for species prediction?') 
plt.xlabel('Importance') 
plt.ylabel('Feature') 
plt.tight_layout() 
fig.savefig('Pickle/gbc_feature_importance.png') 

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Classification Report for Training Data:")
print(train_classification_report)
print("Classification Report for Testing Data:")
print(test_classification_report)
print(features.columns)
