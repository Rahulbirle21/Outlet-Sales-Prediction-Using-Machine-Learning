import pandas as pd
import numpy as np
import pickle as pk
from sklearn.preprocessing import OrdinalEncoder
import streamlit as st

model = pk.load(open('model.pkl','rb'))

html_temp = """ 
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Welcome to Food Sales Prediction </h2>
</div> <br/>"""
st.markdown(html_temp,unsafe_allow_html=True)
df = pd.read_csv('After_EDA.csv')

item_id = st.selectbox('Select Item Identifier',df['Item_Identifier'].unique())
fat = st.selectbox('Select Fat Content',df['Item_Fat_Content'].unique())
type = st.selectbox('Select Item Type',df['Item_Type'].unique())
outlet = st.selectbox('Select Outlet ID',df['Outlet_Identifier'].unique())
size = st.selectbox('Select Outlet Size',df['Outlet_Size'].unique())
location = st.selectbox('Select Outlet Location',df['Outlet_Location_Type'].unique())
out_type = st.selectbox('Select Outlet_Type ',df['Outlet_Type'].unique())
weight = st.slider('Select Weight',0,10)
vis = st.slider('Select Item Visibility',0,5)
mrp = st.slider('Select Item MRP',25,3500)
year = st.selectbox('Select Outlet Establishment year',df['Outlet_Establishment_Year'].unique())

input = pd.DataFrame([[item_id,weight,fat,vis,type,mrp,outlet,year,size,location,out_type]],columns=['Item_Identifier','Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type'])

enc = OrdinalEncoder()
cols = ['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']
input[cols] = enc.fit_transform(input[cols])


if st.button('Predict Sales'):

    sale = model.predict(input).round(2)
    st.markdown('The value Sale of the given food item is : ' + str(sale[0]) + ' '+ 'Rupees')



