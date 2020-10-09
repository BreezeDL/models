import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import streamlit as st


def main():
    st.write("""
 
    This app predicts the **Iris flower** type!
    """)

    st.sidebar.header('User Input Parameters')
    st.sidebar.subheader('Установите значения параметров')
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

    df = user_input_features()


    st.subheader('User Input parameters')
    #st.subheader('Установите значения параметров')
    st.write(df)

    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # Разделение набора данных
    x_axis = iris.data[:, 0]  # Sepal Length
    y_axis = iris.data[:, 1]  # Sepal Width

    # Построение
    plt.figure(figsize=(16,4))

    #fig, ax = plt.subplots(figsize=(20, 10))
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.scatter(x_axis, y_axis, c=iris.target)
    st.pyplot()
    #plt.show()

    clf = RandomForestClassifier()
    clf.fit(X, Y)

    prediction = clf.predict(df)
    prediction_proba = clf.predict_proba(df)

    st.subheader('Class labels and their corresponding index number')

    st.write(iris.target_names)

    st.subheader('Prediction')
    st.write(iris.target_names[prediction])
    #st.write(prediction)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)


if __name__== "__main__":
    main()