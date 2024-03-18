import streamlit as st
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
 
st.title('Penguin Classifier') 

st.write("This app uses 6 inputs to predict the species of penguin using " 

         "a model built on the Palmer's Penguin's dataset. Use the form below" 

         " to get started!") 

store_file = st.file_uploader('Upload your own penguin data') 

if store_file is None: 
    store_df = pd.read_csv('shopping_behavior_new_updated.csv') 
    dtc_pickle = open('./Pickle/dtc.pickle', 'rb') 
    
    map_pickle = open('./Pickle/dtc_output.pickle', 'rb') 

    decision_tree_model = pickle.load(dtc_pickle) 

    output = pickle.load(map_pickle) 

    dtc_pickle.close() 

    map_pickle.close() 

else: 
    store_df = pd.read_csv(store_file) 
    store_df = store_df.dropna() 
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
    st.form_submit_button() 

#preprocessing
Gender_Female, Gender_Male = 0,0
if gender =='Male':
    Gender_Male = 1
    Gender_Female = 0
elif gender == 'Female':
    Gender_Male = 0
    Gender_Female = 1

Item_Purchased_Backpack = 0
Item_Purchased_Belt = 0
Item_Purchased_Blouse = 0
Item_Purchased_Boots = 0
Item_Purchased_Coat = 0
Item_Purchased_Dress= 0
Item_Purchased_Gloves = 0
Item_Purchased_Handbag = 0
Item_Purchased_Hat = 0
Item_Purchased_Hoodie = 0
Item_Purchased_Jacket = 0
Item_Purchased_Jeans = 0
Item_Purchased_Jewelry = 0
Item_Purchased_Pants = 0
Item_Purchased_Sandals = 0
Item_Purchased_Scarf = 0
Item_Purchased_Shirt = 0
Item_Purchased_Shoes = 0
Item_Purchased_Shorts = 0
Item_Purchased_Skirt = 0
Item_Purchased_Sneakers = 0
Item_Purchased_Socks = 0
Item_Purchased_Sunglasses = 0
Item_Purchased_Sweater = 0
Item_Purchased_Tshirt = 0

# Update the corresponding item variable to 1
if item_purchased == "Backpack":
    Item_Purchased_Backpack = 1
elif item_purchased == "Belt":
    Item_Purchased_Belt = 1
elif item_purchased == "Blouse":
    Item_Purchased_Blouse = 1
elif item_purchased == "Boots":
    Item_Purchased_Boots = 1
elif item_purchased == "Coat":
    Item_Purchased_Coat = 1
elif item_purchased == "Dress_":
    Item_Purchased_Dress = 1
elif item_purchased == "Gloves":
    Item_Purchased_Gloves = 1
elif item_purchased == "Handbag":
    Item_Purchased_Handbag = 1
elif item_purchased == "Hat":
    Item_Purchased_Hat = 1
elif item_purchased == "Hoodie":
    Item_Purchased_Hoodie = 1
elif item_purchased == "Jacket":
    Item_Purchased_Jacket = 1
elif item_purchased == "Jeans":
    Item_Purchased_Jeans = 1
elif item_purchased == "Jewelry":
    Item_Purchased_Jewelry = 1
elif item_purchased == "Pants":
    Item_Purchased_Pants = 1
elif item_purchased == "Sandals":
    Item_Purchased_Sandals = 1
elif item_purchased == "Scarf":
    Item_Purchased_Scarf = 1
elif item_purchased == "Shirt":
    Item_Purchased_Shirt = 1
elif item_purchased == "Shoes":
    Item_Purchased_Shoes = 1
elif item_purchased == "Shorts":
    Item_Purchased_Shorts = 1
elif item_purchased == "Skirt":
    Item_Purchased_Skirt = 1
elif item_purchased == "Sneakers":
    Item_Purchased_Sneakers = 1
elif item_purchased == "Socks":
    Item_Purchased_Socks = 1
elif item_purchased == "Sunglasses":
    Item_Purchased_Sunglasses = 1
elif item_purchased == "Sweater":
    Item_Purchased_Sweater = 1
elif item_purchased == "Tshirt":
    Item_Purchased_Tshirt = 1
    
    
Category_Accessories, Category_Clothing, Category_Footwear, Category_Outerwear=0,0,0,0 
if category == 'Accessories':
    Category_Accessories = 1
elif category == 'Clothing':
    Category_Clothing = 1
elif category == 'Footwear':
    Category_Footwear = 1
elif category == 'Outerwear':
    Category_Outerwear = 1

Discount_Applied_No, Discount_Applied_Yes = 0,0
if discount == 'Yes':
    Discount_Applied_Yes = 1
elif discount == 'No':
    Discount_Applied_No = 1

Payment_Method_Bank_Transfer, Payment_Method_Cash, Payment_Method_Credit_Card, Payment_Method_Debit_Card, Payment_Method_PayPal, Payment_Method_Venmo= 0,0,0,0,0,0
if payment_method == 'Bank Transfer':
    Payment_Method_Bank_Transfer = 1
elif payment_method == 'Cash':
    Payment_Method_Cash = 1
elif payment_method == 'Credit Card':
    Payment_Method_Credit_Card = 1
elif payment_method == 'Debit Card':
    Payment_Method_Debit_Card = 1
elif payment_method == 'PayPal':
    Payment_Method_PayPal = 1
elif payment_method == 'Venmo':
    Payment_Method_Venmo = 1

Age_Group_Adult, Age_Group_Old, Age_Group_Teenager, Age_Group_Young_Adult  = 0,0,0,0
if age_group == 'Adult':
    Age_Group_Adult = 1
elif age_group == 'Old':
    Age_Group_Old = 1
elif age_group == 'Teenager':
    Age_Group_Teenager = 1
elif age_group == 'Young Adult':
    Age_Group_Young_Adult = 1

FOP_Annually, FOP_Bi_Weekly, FOP_Every_3_Months, FOP_Fortnightly, FOP_Monthly, FOP_Quarterly, FOP_Weekly =0,0,0,0,0,0,0
if fop == 'Annually':
    FOP_Annually = 1
elif fop == 'Bi-Weekly':
    FOP_Bi_Weekly = 1
elif fop == 'Every 3 Months':
    FOP_Every_3_Months = 1
elif fop == 'Fortnightly':
    FOP_Fortnightly = 1
elif fop == 'Monthly':
    FOP_Monthly = 1
elif fop == 'Quarterly':
    FOP_Quarterly = 1
elif fop == 'Weekly':
    FOP_Weekly = 1

new_prediction = decision_tree_model.predict([[
    age,purchased_amount,review_rating,previous_purchases,
    Gender_Female, Gender_Male,Item_Purchased_Backpack,
    Item_Purchased_Belt,Item_Purchased_Blouse,Item_Purchased_Boots,
    Item_Purchased_Coat,Item_Purchased_Dress,Item_Purchased_Gloves,
    Item_Purchased_Handbag,Item_Purchased_Hat,Item_Purchased_Hoodie,
    Item_Purchased_Jacket, Item_Purchased_Jeans,
    Item_Purchased_Jewelry, Item_Purchased_Pants,
    Item_Purchased_Sandals,Item_Purchased_Scarf,
    Item_Purchased_Shirt, Item_Purchased_Shoes,Item_Purchased_Shorts,
    Item_Purchased_Skirt,Item_Purchased_Sneakers,
    Item_Purchased_Socks, Item_Purchased_Sunglasses,
    Item_Purchased_Sweater,Item_Purchased_Tshirt,
    Category_Accessories, Category_Clothing, Category_Footwear,
    Category_Outerwear,Discount_Applied_No, Discount_Applied_Yes,
    Payment_Method_Bank_Transfer, Payment_Method_Cash,
    Payment_Method_Credit_Card,Payment_Method_Debit_Card,
    Payment_Method_PayPal, Payment_Method_Venmo,Age_Group_Adult,
    Age_Group_Old, Age_Group_Teenager, Age_Group_Young_Adult,
    FOP_Annually, FOP_Bi_Weekly,
    FOP_Every_3_Months,
    FOP_Fortnightly, FOP_Monthly,
    FOP_Quarterly, FOP_Weekly
    ]]) 
prediction_subscription = output[0]
st.write('We predict your Subcription is {}'.format(prediction_subscription)) 