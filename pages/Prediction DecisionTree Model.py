import streamlit as st
import pandas as pd 
import pickle 
from Modelling import DecisionTree as dc

st.title('Subscription Prediction using Decision Tree Model') 
st.markdown("**This page uses the Random Forest model to predict the subscription status based on the 11 features available in the user input section.**")
st.markdown("**Please click the 'Prediction Result' button to display the prediction results and the model evaluation (accuracy).**")
st.divider()


def load_model_and_output(model_file, output_file):
    with open(model_file, 'rb') as model_pickle, open(output_file, 'rb') as output_pickle:
        model = pickle.load(model_pickle)
        output = pickle.load(output_pickle)
    return model, output

def update_parameters(gender, item_purchased, category, discount, payment_method, age_group, fop):
    if gender =='Male':
        gender = 1
    elif gender == 'Female':
        gender = 0
        
    item_purchased_map = {
        'Blouse': 2,
        'Sweater': 23,
        'Jeans': 11,
        'Sandals': 14,
        'Sneakers': 20,
        'Shirt': 16,
        'Shorts': 18,
        'Coat': 4,
        'Handbag': 7,
        'Shoes': 17,
        'Dress': 6,
        'Skirt': 19,
        'Sunglasses': 22,
        'Pants': 13,
        'Jacket': 10,
        'Hoodie': 9,
        'Jewelry': 12,
        'T-shirt': 24,
        'Scarf': 15,
        'Hat': 8,
        'Socks': 21,
        'Backpack': 0,
        'Belt': 1,
        'Boots': 3,
        'Gloves': 6
    }
    item_purchased = item_purchased_map.get(item_purchased, item_purchased)

    category_map = {
        'Clothing': 1,
        'Footwear': 2,
        'Outerwater': 3,
        'Accessories': 0
    }
    category = category_map.get(category, category)
    
    discount = 1 if discount == 'Yes' else 0

    payment_method_map = {
        'Venmo': 5,
        'Cash': 1,
        'Credit Card': 2,
        'PayPal': 4,
        'Bank Transfer': 0,
        'Debit Card': 3
    }
    payment_method = payment_method_map.get(payment_method, payment_method)

    age_group_map = {
        'Old': 1,
        'Teenager': 2,
        'Young Adult': 3,
        'Adult': 0
    }
    age_group = age_group_map.get(age_group, age_group)

    fop_map = {
        'Fortnightly': 3,
        'Weekly': 6,
        'Annually': 0,
        'Querterly': 5,
        'Bi-Weekly': 1
    }
    fop = fop_map.get(fop, fop)

    return gender, item_purchased, category, discount, payment_method, age_group, fop

dt_model, dt_output = load_model_and_output('./Pickle/dtc.pickle', './Pickle/dtc_output.pickle')

 
store_df = pd.read_csv('shopping_behavior_new_updated.csv') 
unique_gender = store_df['Gender'].unique()
unique_category = store_df['Category'].unique()
unique_item_purchase = store_df['Item Purchased'].unique()
unique_payment = store_df['Payment Method'].unique()
unique_age_group = store_df['Age Group'].unique()
unique_fop = store_df['Frequency of Purchases'].unique()
unique_discount = store_df['Discount Applied'].unique()            
for _ in range (2):
    st.markdown("") 
       
_, cols1, cols2, _, cols3, _ = st.columns((0.3,2,2,0.5,7,0.2), gap='medium')
with cols1:
            age = st.slider('Age slider', min_value=17, max_value=70, key='sliderage')
            
            if age <= 19:
                st.write("Age Group: Teenager")
            elif age >= 20 and age <= 24:
                st.write("Age Group: Young Adult")
            elif age >= 25 and age <= 49:
                st.write("Age Group: Adult")
            else:
                st.write("Age Group: Old")
                
            st.divider()
            
            gender = st.selectbox('Gender', options=unique_gender)
            item_purchased = st.selectbox('Item', options=unique_item_purchase) 
            fop = st.selectbox('Frequency of Purchases',options=unique_fop)
            purchased_amount = st.number_input('Amount(USD)', min_value=20, max_value=100)
            discount =  st.selectbox('Discount Applied',options = unique_discount)
            
with cols2:
            age_group = st.selectbox('Age Group',options=unique_age_group)
            for _ in range (3):
                st.markdown("") 
            st.divider()
            
            category = st.selectbox('Product Category',options=unique_category)
            previous_purchases = st.number_input('Previous Purchases', min_value=0, max_value=2)
            payment_method = st.selectbox('Payment Method', options=unique_payment)
            review_rating = st.number_input('Review Rating', min_value=0, max_value=5)
            
            button_submit = st.button('Prediction Result')
            
with cols3:
            with st.container():
                st.subheader('Result', divider='gray')
                if button_submit is True:
                    gender, item_purchased, category, discount, payment_method, age_group, fop = update_parameters(gender, item_purchased, category, discount, payment_method, age_group, fop)
                    prediction_dtc = dt_model.predict([[gender, item_purchased, category, discount, payment_method, age_group, fop, age, purchased_amount, review_rating, previous_purchases]])
                    prediction_subscription = dt_output[prediction_dtc][0]
                    st.success('**Decision Tree Prediction is {}**'.format(prediction_subscription)) 
                    st.markdown("")
                    st.write('**Model Evaluation:**')
                    st.success("**Prediction Accuracy: {:.2f}%**".format(round(dc.test_accuracy * 100)))
            
            for _ in range (3):
                st.markdown("") 
            
            with st.container():
                st.subheader('Feature Importance For Decision Tree Model', divider='gray')
                st.image('./Pickle/dtc_feature_importance.png',width=500)
                
                  

