import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from streamlit_lottie import st_lottie
import xgboost

unique_storey = pickle.load(open("unq_storey_ranges.pkl", "rb"))
unique_town = pickle.load(open("unq_towns.pkl", "rb"))
unique_flat_types = pickle.load(open("unq_flat_types.pkl", "rb"))
unique_flat_models = pickle.load(open("unq_flat_models.pkl", "rb"))

min_least_commence = pickle.load(open("min_least_commence.pkl", "rb"))
max_least_commence = pickle.load(open("max_least_commence.pkl", "rb"))

container = st.container()
try:
    with container:
        @st.cache_data(ttl=60 * 60)
        def load_lottie_file(filepath : str):
            with open(filepath, "r") as f:
                gif = json.load(f)
            st_lottie(gif, speed=1, width=650, height=250)
            
        load_lottie_file("singapore.json")
except:
    print("Don't raise exception")

# X = np.concatenate((X[["floor_area_sqm" , "lease_commence_date"]].values, town_ohe, flat_type_ohe, storey_range_ohe , flat_model_ohe), axis=1)
st.title(":orange[Predicting Resale Price Of Flat In Singapore]")

town = st.selectbox("select town name: ",unique_town,key = 12)
value = st.selectbox("select storey_range: ",unique_storey)
storey_value = st.text_input("slidet selected: ",value)
st.warning("**Note**: storey_range must be in the format of :red[initial floor To final_floor]")
flat_type = st.selectbox("select flat type: ",unique_flat_types,key=13)
flat_model = st.selectbox("select model of your flat: ",unique_flat_models,key=14)

floor_sqm = st.text_input("Enter floor_square_meters: ",key=15)
least_commence = st.text_input("Enter least commence date: ",key=16)
st.warning(f"**Note**: min year is :red[{min_least_commence}] and max year is :red[{max_least_commence}]")

with open(r"xgboost_model.pkl",'rb') as file:
    loaded_model = pickle.load(file)

with open(r'scaler1.pkl', 'rb') as f:
    scaler_loaded = pickle.load(f)

with open(r"town.pkl", 'rb') as f:
    town_loaded = pickle.load(f)

with open(r"flat_type.pkl", 'rb') as f:
    flat_type_loaded = pickle.load(f)

with open(r"storey_range.pkl", 'rb') as f:
    storey_range_loaded = pickle.load(f)

with open(r"flat_model.pkl", 'rb') as f:
    flat_model_loaded = pickle.load(f)

button = st.button(label="Predicting Price of flat", type="primary", key="center_button",use_container_width=True)
if button:
    new_sample = np.array([[int(floor_sqm),int(least_commence),town,flat_type,storey_value,flat_model]])
    new_sample_town = town_loaded.transform(new_sample[:, [2]]).toarray()
    new_sample_flat_type = flat_type_loaded.transform(new_sample[:, [3]]).toarray()
    new_sample_storey = storey_range_loaded.transform(new_sample[:,[4]]).toarray()
    new_sample_flat_model = flat_model_loaded.transform(new_sample[:,[5]]).toarray()
    new_sample = np.concatenate((new_sample[:, [0,1]], new_sample_town, new_sample_flat_type , new_sample_storey,new_sample_flat_model), axis=1)
    # new_sample[0].shape
    new_sample1 = scaler_loaded.transform(new_sample)
    new_pred = loaded_model.predict(new_sample1)
    st.write(f'## :green[Predicted Re-sale price of flat: {np.exp(new_pred)}]')