import pandas as pd 
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report , f1_score
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
gb_classifier = GradientBoostingClassifier(learning_rate= 0.05, max_depth= 3, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 50, random_state=30)
gb_classifier.fit(X_train, y_train)

y_train_pred = gb_classifier.predict(X_train)
y_test_pred = gb_classifier.predict(X_test)

train_accuracy = accuracy_score(y_train_pred, y_train)
test_accuracy = accuracy_score(y_test_pred, y_test)
train_f1score = f1_score(y_train_pred, y_train)
test_f1score = f1_score(y_test_pred, y_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

print("Training F1 Score:", train_f1score)
print("Testing F1 Score:", test_f1score)

with open(file_path, 'wb') as gb_pickle:
    pickle.dump(gb_classifier, gb_pickle)
    gb_pickle.close()

with open(file_path2, 'wb') as output_pickle:
    pickle.dump(uniques, output_pickle) 
    output_pickle.close()

fig, ax = plt.subplots()
ax = sns.barplot(x=gb_classifier.feature_importances_, y=features.columns)
plt.title('Important Features that could predict user subscription')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()

fig.savefig('Pickle/gb_feature_importance.png')