import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],  # Solver algorithm
    'max_iter': [100, 500, 1000, 2000]  # Maximum number of iterations
}

pd.set_option('display.max_columns', None)
current_directory = os.getcwd()
current_directory = current_directory + '/logisticRegression'
file_path = os.path.join(current_directory, 'shopping_behavior_new_updated.csv')

store_df = pd.read_csv('./shopping_behavior_new_updated.csv')
store_df.dropna(inplace=True)
print(store_df.columns)

output = store_df[['Season']]
features = store_df[['Age', 'Gender', 'Item Purchased', 'Category',
                     'Purchase Amount (USD)','Frequency of Purchases' , 'Size', 'Color',
                     'Location', 'Review Rating', 'Subscription Status', 'Shipping Type',
                     'Discount Applied', 'Promo Code Used', 'Previous Purchases', 'Payment Method', 'Age Group']]

features = pd.get_dummies(features)

x_train, x_test, y_train, y_test = train_test_split(
    features, output, test_size=.2)

logistic_model = LogisticRegression(max_iter=2000)
logistic_model.fit(x_train, y_train)

y_pred = logistic_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)


# # Encode categorical variables using LabelEncoder
# output = pd.get_dummies(store_df[['Subscription Status' , 'Frequency of Purchases','Previous Purchases']])
# print (output)
# label_encoder = LabelEncoder()
# non_ordinal_categorical_features = store_df[['Location', 'Item Purchased', 'Category', 'Shipping Type', 'Payment Method', 'Age Group', 'Season', 'Color', 'Size','Subscription Status' , 'Frequency of Purchases','Previous Purchases']]
# label_encoders = {}
# for feature in non_ordinal_categorical_features.columns:
#     label_encoders[feature] = LabelEncoder()
#     store_df[f'Encoded_{feature}'] = label_encoders[feature].fit_transform(store_df[feature])
    

    
# # Scale numerical features
# scaler = StandardScaler()
# numerical_features = ['Purchase Amount (USD)', 'Review Rating', 'Age']
# store_df[numerical_features] = scaler.fit_transform(store_df[numerical_features])


# # print (store_df)
# # print (store_df.columns)

# for column in store_df.columns:
#     print (column)
#     print (store_df[column].unique())
    
    
    
    
# # Step 3: Split Data
# Features = store_df[[]]

# Output = store_df['Subscription Status' , 'Frequency of Purchases','Previous Purchases']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 4: Model Training
# logistic_model = LogisticRegression()
# logistic_model.fit(X_train, y_train)

# # Step 5: Model Evaluation
# y_pred = logistic_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("Confusion Matrix:")
# print(conf_matrix)

