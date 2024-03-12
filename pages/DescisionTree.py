import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

current_directory = os.getcwd()
current_directory = current_directory+'/DecisionTree'
print (current_directory)
file2 = os.path.join(current_directory,'shopping_behavior_new_updated.csv.csv')

# Customer ID,Age,Gender,Item Purchased,Category,
# Purchase Amount (USD),Location,Size,Color,Season,
# Review Rating,Subscription Status,Shipping Type,
# Discount Applied,Promo Code Used,Previous Purchases,
# Payment Method,Frequency of Purchases,Age Group

store_df = pd.read_csv('./shopping_behavior_new_updated.csv')
store_df.dropna(inplace=True)

output = store_df['Location']
features = store_df[['Category','Season',
                     'Shipping Type','Payment Method',
                     'Frequency of Purchases', 'Age Group']]

features =  pd.get_dummies(features)
print (features)
output, uniques = pd.factorize(output) 
x_train, x_test, y_train, y_test = train_test_split(
	features, output, test_size=.2) 

dtc = DecisionTreeClassifier(random_state=42)

dtc.fit(x_train, y_train)
y_pred_dtc = dtc.predict(x_test)
score_dtc = accuracy_score(y_pred_dtc, y_test)
print('Our accuracy score for dtc model is {}'.format(score_dtc))

file_path = os.path.join(current_directory,'dtc.pickle')
with open(file_path, 'wb') as rf_pickle:
	pickle.dump(dtc, rf_pickle) 
	rf_pickle.close() 
 
file_path2 = os.path.join(current_directory,'dtcoutput.pickle')
with open(file_path2, 'wb') as output_pickle:
	pickle.dump(uniques, output_pickle) 
	output_pickle.close()
 
fig, ax = plt.subplots() 
ax = sns.barplot(x=dtc.feature_importances_, y=features.columns) 
plt.title('Which features are the most important for species prediction?') 
plt.xlabel('Importance') 
plt.ylabel('Feature') 
plt.tight_layout() 
fig.savefig(current_directory+'/feature_importance_dtc.png')  
print (features.columns)
# print (store_df.head(10))
# print (store_df.columns)