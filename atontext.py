import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# **Анализ тональности сообщения**
""")

st.sidebar.write(   time.strftime("%d%m%Y")        )
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')

st.sidebar.header('Выбрать модель')
#st.sidebar.subheader('для семантического анализа')

option = st.sidebar.selectbox('Модели ',['Word2Vec' , ' CNN'])


#st.sidebar.write(   time.strftime("%Y%m%d%h%M%S")        )



pw = st.sidebar.slider('Словарь: количество слов для теста ', 1000., 5000., 2000.)


def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 9., 5.)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 5., 3.)
    petal_length = st.sidebar.slider('Petal length', 1.0, 8., 1.)
    petal_width = st.sidebar.slider('Petal width', 0.1, 5., 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

#df = user_input_features()


url = st.text_input('Текст сообщения:')

st.write(' ')
if st.checkbox('Привести к верхнему регистру'):
    st.write(' ', url.upper())

st.write(' ')
a = 1
if a==1:
    st.write(' Тональность сообщения:  положительная ')

#st.subheader('User Input parameters')
#st.subheader('Установите значения параметров')
#st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Разделение набора данных
x_axis = iris.data[:, 0]  # Sepal Length
y_axis = iris.data[:, 1]  # Sepal Width

# Построение
#plt.figure(figsize=(10,1))


#plt.xlabel(iris.feature_names[0])
#plt.ylabel(iris.feature_names[1])
#plt.scatter(x_axis, y_axis, c=iris.target)
#st.pyplot()


clf = RandomForestClassifier()
clf.fit(X, Y)

#prediction = clf.predict(df)
#prediction_proba = clf.predict_proba(df)

#st.subheader('Class labels and their corresponding index number')

#st.write(iris.target_names)

#st.subheader('Prediction')
#st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Вероятность')
#st.write(prediction_proba)
st.write('0.81')


