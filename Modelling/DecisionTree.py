import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option('display.max_columns', None)
current_directory = os.getcwd()
file_path = os.path.join(current_directory,'Pickle/dtc.pickle')
file_path2 = os.path.join(current_directory,'Pickle/dtc_output.pickle')

store_df = pd.read_csv('shopping_behavior_new_updated.csv')
store_df.dropna(inplace=True)

output = store_df['Subscription Status']

features = store_df[['Age', 'Gender', 'Item Purchased', 'Category',
                     'Purchase Amount (USD)', 'Review Rating','Previous Purchases',
                     'Discount Applied', 'Payment Method', 'Age Group', 'Frequency of Purchases']]

features = pd.get_dummies(features) 

x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.3, random_state=42)

decision_tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=12, min_samples_leaf=1, min_samples_split=3)

decision_tree_model.fit(x_train, y_train)

y_train_pred = decision_tree_model.predict(x_train)
y_test_pred = decision_tree_model.predict(x_test)

train_accuracy = accuracy_score(y_train_pred, y_train)
test_accuracy = accuracy_score(y_test_pred, y_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

print("Classification Report for Training Data:")
print(classification_report(y_train, y_train_pred))

print("Classification Report for Testing Data:")
print(classification_report(y_test, y_test_pred))

with open(file_path, 'wb') as rf_pickle:
	pickle.dump(decision_tree_model, rf_pickle) 
	rf_pickle.close() 

with open(file_path2, 'wb') as output_pickle:
	pickle.dump(output, output_pickle) 
	output_pickle.close() 

print (features.columns)
fig, ax = plt.subplots() 
ax = sns.barplot(x=decision_tree_model.feature_importances_, y=features.columns) 
plt.title('Which features are the most important for species prediction?') 
plt.xlabel('Importance') 
plt.ylabel('Feature') 
plt.tight_layout() 
fig.savefig('Pickle/dtc_feature_importance.png') 
