
import streamlit as st
import pickle
import sklearn
import pandas as pd
import json
import numpy as np
from PIL import Image
import math
import base64

with open(
        "bangalore_home_prices_model.pickle",
        'rb') as f:
    __model = pickle.load(f)

with open("columns.json", 'r') as obj:
    __data_columns = json.load(obj)['data_columns']
    __locations = __data_columns[4:]

st.title('Bangalore House Price Prediction')
image = Image.open('bb1.jpeg')
st.image(image,'')


def get_predicted_price( location, sqft, balcony, bathroom, BHK):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError as e:
        loc_index = -1

    lis = np.zeros(len(__data_columns))
    lis[0] = sqft
    lis[1] = bathroom
    lis[2] = balcony
    lis[3] = BHK

    if loc_index >= 0 :
        lis[loc_index] = 1

    price = round(__model.predict([lis])[0], 2)
    strp = ' lakhs'

    if math.log10(price) >= 2:
        price = price / 100
        price = round(price, 2)
        strp = " crores"

    return str(price) + strp


def main():
    global result
    html_temp = """
           <div>
           <h2>House Price Prediction ML app</h2>
           </div>
           """
    st.markdown(html_temp, unsafe_allow_html=True)
    total_sqft = st.text_input("Total_sqft")
    balcony = st.text_input("Number of Balconies")
    bathroom = st.text_input("Number of Bathrooms")
    BHK = st.text_input("BHK")
    location = st.selectbox("Location", __locations)

    result=0
    if st.button("Predict"):
        result = get_predicted_price(location, total_sqft, balcony, bathroom, BHK)

    st.success(f"Price = {result}")


if __name__ == "__main__":
    main()