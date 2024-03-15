import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('shopping_behavior_new_updated.csv')

features = df[['Age', 'Gender', 'Item Purchased', 'Category',
                     'Purchase Amount (USD)', 'Location', 'Size', 'Color',
                     'Season', 'Review Rating','Previous Purchases' , 'Shipping Type',
                     'Discount Applied', 'Promo Code Used', 'Payment Method', 'Age Group', 'Frequency of Purchases']]

output = df['Subscription Status']

features = pd.get_dummies(features)

X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.2, random_state=30)

rf_classifier = RandomForestClassifier(random_state=30)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))