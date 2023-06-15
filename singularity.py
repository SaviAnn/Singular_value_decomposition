import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
st.write("""
# Singular Value Decomposition (SVD) and Image Compression
""")
# создаем кнопку загрузки файла
uploaded_file = st.file_uploader("Select image", type=["jpg","jpeg","png"])

# если файл загружен, отображаем его
if uploaded_file is not None:
    # читаем файл в формате PIL
    image = Image.open(uploaded_file)
    # отображаем изображение
    st.write("""
    #### Selected image
        """)
    st.image(image,use_column_width=True)

    #col_img = plt.imread('Kit_colour.jpeg')
    #img = Image.fromarray(image)
    img = image
    img = img.convert("L")
     # отображаем изображение
    st.write("""
    #### Change to grayscale in order to ease calculations
        """)
    st.image(img,use_column_width=True)
    
    #img.ravel().shape
    img=np.asarray(img)
    img = img/255
    U, sing_vals, V = np.linalg.svd(img)
    sigma = np.zeros(shape=(U.shape[0], V.shape[0]))
    np.fill_diagonal(sigma, sing_vals)
    st.write("""
    #### You can choose quality of the processed image
        """)
    top_k = st.slider(label=' Main components number', min_value=0, max_value=200, value=40)

    trunc_U = U[:, :top_k]
    trunc_sigma = sigma[:top_k, :top_k]
    trunc_V = V[:top_k, :]
    trunc_img = trunc_U@trunc_sigma@trunc_V
    st.image(trunc_img, clamp=True)