import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
current_directory = os.getcwd()
file_path = os.path.join(current_directory,'Pickle/dtc.pickle')
file_path2 = os.path.join(current_directory,'Pickle/dtc_output.pickle')

store_df = pd.read_csv('shopping_behavior_new_updated.csv')
store_df.dropna(inplace=True)

output = store_df['Subscription Status']
output,uniques = pd.factorize(output)

features = store_df[['Gender', 'Item Purchased', 'Category',
                     'Discount Applied', 'Payment Method', 'Age Group', 'Frequency of Purchases']]
num_features =  store_df[['Age','Purchase Amount (USD)','Review Rating','Previous Purchases']]

for feature in features:
    print (feature)
    print (features[feature].unique())
encoders = {}

for feature in features:
    encoder = LabelEncoder()
    encoded_values = encoder.fit_transform(features[feature])
    features.loc[:, feature] = encoded_values
    encoders[feature] = encoder
for feature in features:
    print (feature)
    print (features[feature].unique())  
     
num_features = pd.get_dummies(num_features)  
features = pd.concat([features, num_features], axis=1)

x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.3, random_state=42)

decision_tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=12, min_samples_leaf=1, min_samples_split=3)

decision_tree_model.fit(x_train, y_train)

y_train_pred = decision_tree_model.predict(x_train)
y_test_pred = decision_tree_model.predict(x_test)

train_accuracy = accuracy_score(y_train_pred, y_train)
test_accuracy = accuracy_score(y_test_pred, y_test)
train_f1score = f1_score(y_train_pred, y_train)
test_f1score = f1_score(y_test_pred, y_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

print("Training Accuracy:", train_f1score)
print("Testing Accuracy:", test_f1score)

with open(file_path, 'wb') as rf_pickle:
	pickle.dump(decision_tree_model, rf_pickle) 
	rf_pickle.close() 

with open(file_path2, 'wb') as output_pickle:
	pickle.dump(uniques, output_pickle) 
	output_pickle.close() 

fig, ax = plt.subplots() 
ax = sns.barplot(x=decision_tree_model.feature_importances_, y=features.columns) 
plt.title('Important Features that could predict user subcription') 
plt.xlabel('Importance') 
plt.ylabel('Feature') 
plt.tight_layout() 

fig.savefig('Pickle/dtc_feature_importance.png') 
