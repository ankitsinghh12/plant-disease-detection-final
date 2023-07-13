import streamlit as st
import os
from matplotlib.cbook import file_requires_unicode
from PIL import Image
import tensorflow as tf
from skimage import transform
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np

st.header("UPLOAD IMAGE TO PREDICT")
Classes = ["Apple___Apple_scab", 
           "Apple___Black_rot", 
           "Apple___Cedar_apple_rust", 
           "Apple___healthy", 
           "Blueberry___healthy", 
           "Cherry_(including_sour)___Powdery_mildew", 
           "Cherry_(including_sour)___healthy",
           "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 
           "Corn_(maize)___Common_rust",  
           "Corn_(maize)___Northern_Leaf_Blight", 
           "Corn_(maize)___healthy",
           "Grape___Black_rot", 
           "Grape__Esca(Black_Measles)", 
           "Grape__Leaf_blight(Isariopsis_Leaf_Spot)",  
           "Grape___healthy",
           "Orange__Haunglongbing(Citrus_greening)", 
           "Peach___Bacterial_spot", 
           "Peach___healthy", 
           "Pepper_bell__Bacterial_spot", 
           "Pepper_bell__healthy", 
           "Potato___Early_blight",
           "Potato___Late_blight",
           "Potato___healthy",
           "Raspberry___healthy", 
           "Soybean___healthy", 
           "Squash___Powdery_mildew", 
           "Strawberry___Leaf_scorch",
           "Strawberry___healthy",
           "Tomato___Bacterial_spot",
           "Tomato___Early_blight",
           "Tomato___Late_blight",
           "Tomato___Leaf_Mold",
           "Tomato___Septoria_leaf_spot",
           "Tomato___Spider_mites Two-spotted_spider_mite",
           "Tomato___Target_Spot",
           "Tomato___Tomato_Yellow_Leaf_Curl_Virus" ,
           "Tomato___Tomato_mosaic_virus",
           "Tomato___healthy",
           ]
image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
def load_image(image_file):
    img = Image.open(image_file)
    return img
if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    file_name=list(file_details.values())
    img_name=file_name[0]
    img = load_image(image_file)
    with open(os.path.join("./images/",image_file.name),"wb") as f: 
      f.write(image_file.getbuffer())         
    st.success("Saved")

clk=st.button("Predict")
model=tf.keras.models.load_model('./model/Plant_final_1.h5') 

if clk:  
    directory="./images/{}".format(img_name)
    #directory=file_name[0]
    #files = [os.path.join(directory,p) for p in sorted(os.listdir(directory))] 
    #for i in range(0,5):
    image_path = directory
    new_img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    prediction = model.predict(img)
    #print(prediction)
    probabilty = prediction.flatten()
    #print(probabilty)
    max_prob = probabilty.max()
    index=prediction.argmax(axis=-1)[0]
    #print(index)
    class_name = Classes[index]
    #print(class_name)
    #ploting image with predicted class name        
    plt.figure(figsize = (4,4))
    #ax = plt.subplot(3,3, i + 1)
    plt.imshow(new_img)
    st.image(new_img)
    plt.axis('off')
    plt.title(class_name+" "+ str(max_prob)[0:4]+"%")
    st.write(class_name)
    plt.show()




#files = [os.path.join(directory,p) for p in sorted(os.listdir(directory))] 
    #st.write(directory)
    #st.write(directory)
    # st.write(files[0])
    # st.image(files)

    # directory="./images/"
    # files = [os.path.join(directory,p) for p in sorted(os.listdir(directory))] 
        
    #remove_file=files[0]
    #os.remove(files[0])


    
      