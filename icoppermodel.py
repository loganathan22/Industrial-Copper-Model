import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import sklearn
import re
import pickle
import joblib
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer

# --------------------------------------------------Logo & details on top

st.set_page_config(page_title= "Industrial Copper Modeling | loganathan",
                   layout= "wide",
                   initial_sidebar_state= "expanded")
with st.sidebar:
    opt = option_menu("Menu",
                    ['HOME',"PREDICT SELLING PRICE", "PREDICT STATUS"])
if opt=="HOME":
        st.title(''':orange[_INDUSTRIAL COPPER MODELING_]''')
        st.write("#")
        st.write("This project aims to develop two machine learning models for the copper industry to address the challenges of predicting selling price and lead classification.")
        st.write("Manual predictions can be time-consuming and may not result in optimal pricing decisions or accurately capture leads.")
        st.write("The models will utilize advanced techniques such as data normalization, outlier detection and handling, handling data in the wrong format, identifying the distribution of features, and leveraging tree-based models, specifically the decision tree algorithm, to predict the selling price and leads accurately.")
        st.write(" ")
        st.markdown("### :orange[DOMAIN :] MANUFACTURING")
        st.markdown("""
                     ### :orange[TECHNOLOGIES USED :] 

                        - PYTHON
                        - PANDAS
                        - DATA PREPROCESING
                        - EDA
                        - SCIKIT - LEARN
                        - STREAMLIT
                       
                    """)
        
if opt=="PREDICT SELLING PRICE":
        st.write( f'<h5 style="color:magenta;">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
        # Define the possible values for the dropdown menus
        status_options = ['Won', 'Lost']
        item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
        product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                     '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                     '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                     '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                     '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
        with st.form("my_form"):
            col1,col2=st.columns([5,5])
            with col2:
               
                status = st.selectbox("Status", status_options,key=1)
                item_type = st.selectbox("Item Type", item_type_options,key=2)
                country = st.selectbox("Country", sorted(country_options),key=3)
                application = st.selectbox("Application", sorted(application_options),key=4)
                product_ref = st.selectbox("Product Reference", product,key=5)

            with col1:               
               
                quantity_tons = st.text_input("Enter Quantity Tons (Min:-2000 & Max:1000000000.0)")
                thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                width = st.text_input("Enter width (Min:1, Max:2990)")
                customer = st.text_input("customer ID (Min:12458, Max:30408185)")
                submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
                st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: magenta;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)
    
            flag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [quantity_tons,thickness,width,customer]:             
                if re.match(pattern, i):
                    pass
                else:                    
                    flag=1  
                    break
            
        if submit_button and flag==1:
            if len(i)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",i)  
             
        if submit_button and flag==0:
             
            with open(r"regmodel.pkl", 'rb') as file:
               loaded_model = pickle.load(file)
            with open(r'regscaler.pkl', 'rb') as f:
               scaler = pickle.load(f)

            with open(r"regitype.pkl", 'rb') as f:
               item = pickle.load(f)

            with open(r"regstatus.pkl", 'rb') as f:
                stat = pickle.load(f)

            sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
            sample_it = item.transform(sample[:, [7]]).toarray()
            sample_s = stat.transform(sample[:, [8]]).toarray()
            sample = np.concatenate((sample[:, [0,1,2, 3, 4, 5, 6,]], sample_it, sample_s), axis=1)
            sample1 = scaler.transform(sample)
            pred = loaded_model.predict(sample1)[0]
            st.write('## :violet[Predicted selling price:] ', np.exp(pred))

if opt=="PREDICT STATUS":
        st.write( f'<h5 style="color:magenta;">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )

        with st.form("my_form1"):
                status_options = ['Won', 'Lost']
                item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
                country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
                application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
                product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                            '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                            '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                            '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                            '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
                col1,col2,=st.columns([5,5])
                with col2:
                    cquantity_tons = st.text_input("Enter Quantity Tons (Min:-2000 & Max:1000000000.0)")
                    cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                    cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                    ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                    cselling = st.text_input("Selling Price (Min:1, Max:100001015)") 
                    csubmit_button = st.form_submit_button(label="PREDICT STATUS")
                
                with col1:    
                    st.write(' ')
                    item_type = st.selectbox("Item Type", item_type_options,key=2)
                    country = st.selectbox("Country", sorted(country_options),key=3)
                    application = st.selectbox("Application", sorted(application_options),key=4)
                    product_ref = st.selectbox("Product Reference", product,key=5)          
                   
                    st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: magenta;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)
        
            
                cflag=0 
                pattern = "^(?:\d+|\d*\.\d+)$"
                for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:             
                    if re.match(pattern, k):
                        pass
                    else:                    
                        cflag=1  
                        break
                
        if csubmit_button and cflag==1:
            if len(k)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",k)  
             
        if csubmit_button and cflag==0:
            import pickle
            with open(r"clsmodel.pkl", 'rb') as file:
                cloaded_model = pickle.load(file)

            with open(r'clsscaler.pkl', 'rb') as f:
                cscaler_loaded = pickle.load(f)

            with open(r"itypecls.pkl", 'rb') as f:
                ct_loaded = pickle.load(f)

            # Predict the status for a new sample
            # 'quantity tons_log', 'selling_price_log','application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe
            sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), application, np.log(float(cthickness)),float(cwidth),country,int(ccustomer),int(product_ref),item_type]])
            sample_it = ct_loaded.transform(sample[:, [8]]).toarray()
            sample = np.concatenate((sample[:, [0,1,2, 3, 4, 5, 6,7]], sample_it), axis=1)
            sample = cscaler_loaded.transform(sample)
            pred = cloaded_model.predict(sample)
            if pred==1:
                st.write('## :rainbow[The Status is Won] ')
            else:
                st.write('## :orange[The status is Lost] ')