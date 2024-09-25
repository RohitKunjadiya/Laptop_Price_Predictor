import streamlit as st
import pickle
import numpy as np

#import the model

st.title("Laptop Price Predictor")

pipe = pickle.load(open('pipe.pkl','rb'))

df = pickle.load(open('df.pkl','rb'))

#brand
Company = st.selectbox("Brand",df['Company'].unique())

#TypeName
Type = st.selectbox('Type',df['TypeName'].unique())

# Ram
Ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
Weight = st.number_input('Weight of the Laptop(KG)')

# Touchscreen
Touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
Ips = st.selectbox('IPS',['No','Yes'])

# screen size
Screen_size = st.number_input('Screen Size', min_value=10.0)

CPU_Freq = st.selectbox('CPU Frequency(GHz)',df['CPU_Frequency (GHz)'].unique())


# resolution
Resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
CPU = st.selectbox('CPU',df['CPU_Company'].unique())

HDD = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

SSD = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

GPU = st.selectbox('GPU',df['GPU_Company'].unique())

OS = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):

    ppi = None
    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0

    if Ips == 'Yes':
        Ips = 1
    else:
        Ips = 0

    X_res = int(Resolution.split('x')[0])
    Y_res = int(Resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / Screen_size
    query = np.array([Company, Type, CPU, CPU_Freq, Ram, GPU, Weight, Touchscreen, Ips, ppi, HDD, SSD, OS])

    query = query.reshape(1, 13)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))) + " Euro")