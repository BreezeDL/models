import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App
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


url = st.text_input('Enter URL')
st.write('The Entered URL is', url)

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
plt.figure(figsize=(10,1))

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


import plotly.express as px
data = dict(
    number=[39, 24, 21, 14, 2],
    stage=["Visit", "Downloads", "Potential customers", "Requested price", "Invoice sent"])
fig = px.funnel(data, x='number', y='stage', width=600, height=400)
#st.fig.show()
#st.plotly
st.plotly_chart(fig)


import plotly.express as px
#import pandas as pd
stages = ["Visit", "Downloads", "Potential customers", "Requested price", "Invoice sent"]
df_mtl = pd.DataFrame(dict(number=[39, 24, 21, 14, 2], stage=['Visit', "Downloads", "Potential customers", "Requested price", "Invoice sent"]))
df_mtl['office'] = 'Belgorod'
df_toronto = pd.DataFrame(dict(number=[52, 36, 18, 14, 5], stage=stages))
df_toronto['office'] = 'Kursk'
df = pd.concat([df_mtl, df_toronto], axis=0)
fig = px.funnel(df, x='number', y='stage', color='office', width=600, height=400)
#fig.show()
st.plotly_chart(fig)



import plotly.figure_factory as ff
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
# Group data together
hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']

 # Create distplot with custom bin_size
fig = ff.create_distplot(
hist_data, group_labels, bin_size=[.1, .25, .5])

# Plot!
st.plotly_chart(fig, use_container_width=True)
