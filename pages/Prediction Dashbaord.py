import streamlit as st
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle 
from sklearn.metrics import accuracy_score ,classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

 
st.title('Subscription Prediction using ML') 

st.write("This app uses 6 inputs to predict the species of penguin using " 

         "a model built on the Palmer's Penguin's dataset. Use the form below" 

         " to get started!") 

store_file = st.file_uploader('Upload your own penguin data') 

if store_file is None: 
    store_df = pd.read_csv('shopping_behavior_new_updated.csv') 
    
    #Decision Tree Pickle
    dtc_pickle = open('./Pickle/dtc.pickle', 'rb') 
    
    map_pickle = open('./Pickle/dtc_output.pickle', 'rb') 

    decision_tree_model = pickle.load(dtc_pickle) 

    output = pickle.load(map_pickle) 

    dtc_pickle.close() 

    map_pickle.close() 
    
    #Random Forest Pickle
    rf_pickle = open('./Pickle/rf.pickle', 'rb') 
    rf_output_pickle = open('./Pickle/rf_output.pickle', 'rb') 

    rf_classifier = pickle.load(rf_pickle)
    rf_output = pickle.load(rf_output_pickle)
    
    rf_pickle.close()
    rf_output_pickle.close()

else: 
    store_df = pd.read_csv(store_file) 
    store_df = store_df.dropna() 
    
    #Decision Tree Model
    output = store_df['Subscription Status']

    features = store_df[['Gender', 'Item Purchased', 'Category',
                        'Discount Applied', 'Payment Method', 'Age Group', 'Frequency of Purchases']]
    num_features =  store_df[['Age','Purchase Amount (USD)','Review Rating','Previous Purchases']]

    encoders = {}

    for feature in features:
        encoder = LabelEncoder()
        encoded_values = encoder.fit_transform(features[feature])
        features.loc[:, feature] = encoded_values
        encoders[feature] = encoder
        
    num_features = pd.get_dummies(num_features)  
    features = pd.concat([features, num_features], axis=1)

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

    # Random Forest Model
    features = df[['Age', 'Gender', 'Item Purchased', 'Category',
               'Purchase Amount (USD)', 'Review Rating','Previous Purchases',
                'Discount Applied', 'Payment Method', 'Frequency of Purchases']]

    output = df['Subscription Status']

    label_encoder = LabelEncoder()

    for column in features:
        if features[column].dtype == 'object':
            features[column] = label_encoder.fit_transform(features[column])

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


unique_gender = store_df['Gender'].unique()
unique_category = store_df['Category'].unique()
unique_item_purchase = store_df['Item Purchased'].unique()
unique_payment = store_df['Payment Method'].unique()
unique_age_group = store_df['Age Group'].unique()
unique_fop = store_df['Frequency of Purchases'].unique()
unique_discount = store_df['Discount Applied'].unique()
    
with st.form('user_inputs'): 
    age = st.number_input('Age', min_value=0)
    gender = st.selectbox('Gender', options=unique_gender)
    item_purchased = st.selectbox('Item', options=unique_item_purchase) 
    category = st.selectbox('category',options=unique_category)
    purchased_amount = st.number_input('Amount(USD)', min_value=0)
    review_rating = st.number_input('Review Rating', min_value=0)
    previous_purchases = st.number_input('Previous Purchases', min_value=0)
    discount =  st.selectbox('Discount Applied',options =  unique_discount)
    payment_method = st.selectbox('Payment Method', options=unique_payment)
    age_group = st.selectbox('Age Group',options=unique_age_group)
    fop = st.selectbox('Frequency of Purchases',options=unique_fop)
    button_submit = st.form_submit_button('Prediction Result')

#preprocessing
if gender =='Male':
    gender = 0
elif gender == 'Female':
    gender = 1

# Update the corresponding item variable to 1
if item_purchased == "Backpack":
    item_purchased = 2
elif item_purchased == "Belt":
    item_purchased = 23
elif item_purchased == "Blouse":
    item_purchased = 11
elif item_purchased == "Boots":
    item_purchased = 14
elif item_purchased == "Coat":
    item_purchased = 20
elif item_purchased == "Dress_":
    item_purchased = 16
elif item_purchased == "Gloves":
    item_purchased = 18
elif item_purchased == "Handbag":
    item_purchased = 4
elif item_purchased == "Hat":
    item_purchased = 7
elif item_purchased == "Hoodie":
    item_purchased = 17
elif item_purchased == "Jacket":
    item_purchased = 6
elif item_purchased == "Jeans":
    item_purchased = 19
elif item_purchased == "Jewelry":
    item_purchased = 22
elif item_purchased == "Pants":
    item_purchased = 13
elif item_purchased == "Sandals":
    item_purchased = 10
elif item_purchased == "Scarf":
    item_purchased = 9
elif item_purchased == "Shirt":
    item_purchased = 12
elif item_purchased == "Shoes":
    item_purchased = 24
elif item_purchased == "Shorts":
    item_purchased = 15
elif item_purchased == "Skirt":
    item_purchased = 8
elif item_purchased == "Sneakers":
    item_purchased = 21
elif item_purchased == "Socks":
    item_purchased = 0
elif item_purchased == "Sunglasses":
    item_purchased = 1
elif item_purchased == "Sweater":
    item_purchased = 3
elif item_purchased == "Tshirt":
    item_purchased = 6    

if category == 'Clothing':
    category = 1
elif category == 'Footwear':
    category = 2
elif category == 'Outerwater':
    category = 3
elif category == 'Accessories':
    category = 0

if discount == 'Yes':
    discount = 1
elif discount == 'No':
    discount = 0

if payment_method == 'Venmo':
    payment_method = 5
elif payment_method == 'Cash':
    payment_method = 1
elif payment_method == 'Credit Card':
    payment_method = 2
elif payment_method == 'PayPal':
    payment_method = 4
elif payment_method == 'Bank Transfer':
    payment_method = 0
elif payment_method == 'Debit Card':
    payment_method = 3


if age_group == 'Old':
    age_group = 1
elif age_group == 'Teenager':
    age_group = 2
elif age_group == 'Young Adult':
    age_group = 3
elif age_group == 'Adult':
    age_group = 0

if fop == 'Fortnightly':
    fop = 3
elif fop == 'Weekly':
    fop = 6
elif fop == 'Annually':
    fop = 0
elif fop == 'Querterly':
    fop = 5
elif fop == 'Bi-Weekly':
    fop = 1
elif fop == 'Monthly':
    fop = 4
elif fop == 'Every 3 Months':
    fop = 2
    
if button_submit is True:
    new_prediction = decision_tree_model.predict([[gender,item_purchased,category,discount,payment_method,age_group,fop,age,purchased_amount,review_rating,previous_purchases]]) 
    new_prediction_rf = rf_classifier.predict([[age, gender, item_purchased, category, purchased_amount, review_rating, previous_purchases, discount, payment_method, fop]])
    prediction_subscription = output[0]
    st.success('We predict your Subcription is {}'.format(prediction_subscription)) 
    st.success('Random Forest Prediction is {}'.format(new_prediction_rf)) 
