import streamlit as st
import pandas as pd 
import pickle 

st.title('Subscription Prediction using ML') 
st.write("This application contains 3 Machine Learning models such as Decision Tree, Random Forest and Gradient Boosting model."
         " It is use to predict the subscription status based of the 11 features available in the user input section.\n"
         "\nPlease click the 'Prediction Result' button to show all the 3 Machine Learning Models.") 
st.divider()


def load_model_and_output(model_file, output_file):
    with open(model_file, 'rb') as model_pickle, open(output_file, 'rb') as output_pickle:
        model = pickle.load(model_pickle)
        output = pickle.load(output_pickle)
    return model, output

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
                
            prediction_dtc = dt_model.predict([[gender, item_purchased, category, discount, payment_method, age_group, fop, age, purchased_amount, review_rating, previous_purchases]])
            prediction_subscription = dt_output[prediction_dtc][0]
            st.success('Decision Tree Prediction is {}'.format(prediction_subscription)) 

