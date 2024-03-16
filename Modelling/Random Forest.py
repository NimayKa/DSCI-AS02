import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('shopping_behavior_new_updated.csv')

features = df[['Age', 'Gender', 'Item Purchased', 'Category',
                     'Purchase Amount (USD)', 'Review Rating','Previous Purchases',
                     'Discount Applied', 'Payment Method', 'Age Group', 'Frequency of Purchases']]

output = df['Subscription Status']

features = pd.get_dummies(features)

X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.2, random_state=30)

rf_classifier = RandomForestClassifier(random_state=30)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

rf_classifier = RandomForestClassifier(**best_params, random_state=30)
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
