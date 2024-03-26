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
rf_classifier = RandomForestClassifier(max_depth= 10, min_samples_leaf= 4, min_samples_split= 10, n_estimators= 150, random_state=42)
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
	pickle.dump(uniques, output_pickle) 
	output_pickle.close() 

fig, ax = plt.subplots() 
ax = sns.barplot(x=rf_classifier.feature_importances_, y=features.columns) 
plt.title('Important Features that could predict user subcription') 
plt.xlabel('Importance') 
plt.ylabel('Feature') 
plt.tight_layout() 

fig.savefig('Pickle/rf_feature_importance.png') 