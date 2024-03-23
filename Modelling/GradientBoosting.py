# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle 
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report

# import warnings
# from sklearn.exceptions import DataConversionWarning
# warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')
# warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)

# pd.set_option('display.max_columns', None)

# current_directory = os.getcwd()
# file_path = os.path.join(current_directory, 'shopping_behavior_new_updated.csv')
# store_df = pd.read_csv(file_path)
# store_df.dropna(inplace=True)

# output = store_df['Subscription Status']
# features = store_df[['Age', 'Gender', 'Item Purchased', 'Category',
#                      'Purchase Amount (USD)', 'Review Rating','Previous Purchases',
#                      'Discount Applied', 'Payment Method', 'Age Group', 'Frequency of Purchases']]
# features = pd.get_dummies(features) 

# x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.3, random_state=42)

# gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
# gradient_boosting_model.fit(x_train, y_train)

# y_train_pred = gradient_boosting_model.predict(x_train)
# y_test_pred = gradient_boosting_model.predict(x_test)
# train_accuracy = accuracy_score(y_train_pred, y_train)
# test_accuracy = accuracy_score(y_test_pred, y_test)
# train_classification_report = classification_report(y_train, y_train_pred)
# test_classification_report = classification_report(y_test, y_test_pred)

# file_path_model = os.path.join(current_directory, 'Pickle/gbs.pickle')
# file_path_predictions = os.path.join(current_directory, 'Pickle/gbs_predictions.pickle')

# with open(file_path_model, 'wb') as rf_pickle:
#     pickle.dump(gradient_boosting_model, rf_pickle) 

# with open(file_path_predictions, 'wb') as output_pickle:
#     pickle.dump(y_test_pred, output_pickle) 

# fig, ax = plt.subplots() 
# ax = sns.barplot(x=gradient_boosting_model.feature_importances_, y=features.columns) 
# plt.title('Which features are the most important for species prediction?') 
# plt.xlabel('Importance') 
# plt.ylabel('Feature') 
# plt.tight_layout() 
# fig.savefig('Pickle/gbc_feature_importance.png') 

# print("Training Accuracy:", train_accuracy)
# print("Testing Accuracy:", test_accuracy)
# print("Classification Report for Training Data:")
# print(train_classification_report)
# print("Classification Report for Testing Data:")
# print(test_classification_report)
# print(features.columns)

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

features = df[['Age', 'Gender', 'Item Purchased', 'Category',
                     'Purchase Amount (USD)', 'Review Rating','Previous Purchases',
                     'Discount Applied', 'Payment Method', 'Age Group', 'Frequency of Purchases']]

output = df['Subscription Status']

label_encoder = LabelEncoder()

for column in features:
    if features[column].dtype == 'object':
        features[column] = label_encoder.fit_transform(features[column])

X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.2, random_state=30)

gb_classifier = GradientBoostingClassifier(random_state=30)

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=gb_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

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
