import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import io



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
    buffer_col = img.tobytes()
    # Получаем размер изображения в байтах
    img_size = len(buffer_colr)

    st.write(f"Initial image: {img_size} byte")
    img = image
    img = img.convert("L")
     # отображаем изображение
    st.write("""
    #### Change to grayscale in order to ease calculations
        """)
    st.image(img,use_column_width=True)
    buffer_gr = img.tobytes()

    # Получаем размер изображения в байтах
    img_size = len(buffer_gr)

    st.write(f"Greyscale initial image: {img_size} byte")
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
    trunc_img = Image.fromarray(np.uint8(trunc_img))
    buffer_gr = trunc_img.tobytes()
    size_in_bytes = len(buffer_gr)
   
   
    st.write(f"Greyscale compressed image: {size_in_bytes}  byte")
     #Теперь для цветного
     # Преобразуем изображение в массив NumPy
    img_array = np.array(image)

    # Разделяем массив на цветовые каналы
    red = img_array[:, :, 0]
    green = img_array[:, :, 1]
    blue = img_array[:, :, 2]
   
    # Выполняем SVD разложение для каждого цветового канала
    U_red, sing_vals_red, V_red = np.linalg.svd(red)
    U_green, sing_vals_green, V_green = np.linalg.svd(green)
    U_blue, sing_vals_blue, V_blue = np.linalg.svd(blue)

    # Обрезаем матрицы U, sigma и V до top_k компонент
    trunc_U_red = U_red[:, :top_k]
    trunc_sigma_red = np.diag(sing_vals_red[:top_k])
    trunc_V_red = V_red[:top_k, :]

    trunc_U_green = U_green[:, :top_k]
    trunc_sigma_green = np.diag(sing_vals_green[:top_k])
    trunc_V_green = V_green[:top_k, :]

    trunc_U_blue = U_blue[:, :top_k]
    trunc_sigma_blue = np.diag(sing_vals_blue[:top_k])
    trunc_V_blue = V_blue[:top_k, :]

    # Восстанавливаем цветные каналы
    trunc_red = trunc_U_red @ trunc_sigma_red @ trunc_V_red
    trunc_green = trunc_U_green @ trunc_sigma_green @ trunc_V_green
    trunc_blue = trunc_U_blue @ trunc_sigma_blue @ trunc_V_blue

    # Объединяем цветные каналы
    trunc_img = Image.fromarray(np.uint8(np.dstack((trunc_red, trunc_green, trunc_blue))))

    # Отображаем восстановленное цветное изображение
    st.image(trunc_img, clamp=True, use_column_width=True)
    # Получаем буфер изображения и считаем его размер
    buffer = trunc_img.tobytes()
    size_in_bytes = len(buffer)
   

    st.write(f"Compressed image: {size_in_bytes} byte")
