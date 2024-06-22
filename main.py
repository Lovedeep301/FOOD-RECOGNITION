import streamlit as st
import tensorflow as tf
import numpy as np
#tensorflow model Prediction
def model_Prediction(test_image):
    model = tf.keras.models.load_model("train_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # CONVERT SINGLE IMAGE TO BATCH
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #RETURN INDEX OF MAX ELEMENT 
#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])
#Main Page
if(app_mode == "Home"):
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img.jpeg"
    st.image(image_path)
elif(app_mode == "About Project"):
    image_path = "home_img.jpeg"
    st.image(image_path)
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear,grapes, orange, kiwi, watermelon, pomegranate, pineaapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower")
    st.subheader("Content")
    st.text("This dataset contain three folder:")
    st.text("train (100images each)")
    st.text("validation (10 images each)")
    
    
    
#Prediction Page
elif(app_mode == "Prediction"):
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show_image")):
        st.image(test_image,width=4,use_column_width=True)
    #Prediction button
    if(st.button("Prediction")):
        st.balloons()
        
        st.write("Our Prediction")
        result_index = model_Prediction(test_image)
        #Reading Lables
        with open("labels.txt") as f:
           content = f.readlines()
        label =[]
        for i in content:
            label.append(i[:-1])
        st.success(" {}".format(label[result_index]))
st.snow()
