import streamlit as st 
import base64
st.title("Plant Disease Detection ")

#####################Set background################
#######################################################################
from PIL import Image
image = Image.open('./show-photo/plant.jpeg')
st.image(image, caption='')


st.markdown(
    '''
    ### Introduction

###### Creating an  Web application that detects diseases in plants using Convolutional Neural Network (CNN) , which is a Deep Learning Algorithm. According to the Food and Agriculture Organization of the United Nations (UN), transboundary plant pests and diseases affect food crops, causing significant losses to farmers and threatening food security.

### Motive

###### For this challenge, we used the “plant_disease” dataset. This dataset contains an open access repository of images on plant health to enable the development of web disease diagnostics. The dataset contains 87,0000 images. The images span 14 crop species: Apple, Blueberry, Cherry, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato. It contains images of 17 fundal diseases, 4 bacterial diseases, 2 molds (oomycete) diseases, 2 viral diseases, and 1 disease caused by a mite. 12 crop species also have images of healthy leaves that are not visibly affected by a disease.


'''
)



