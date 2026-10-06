import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris  
from sklearn.ensemble import RandomForestClassifier  
import streamlit as st


@st.cache_data
def load_data(): 
    iris=load_iris() 
    df=pd.DataFrame(data=iris.data,columns=iris.feature_names) 
    df["Species"]=iris.target 
    return df,iris.target_names

 
df,target_names=load_data()  
model=RandomForestClassifier()
model.fit(df.iloc[:,:-1],df.iloc[:,-1])  

st.sidebar.title("Iris Flower Prediction")  
sepal_length=st.sidebar.slider("Sepal Length",float(df["sepal length (cm)"].min()),float(df["sepal length (cm)"].max())) 
sepal_width=st.sidebar.slider("Sepal Width",float(df["sepal width (cm)"].min()),float(df["sepal width (cm)"].max()))
petal_length=st.sidebar.slider("Petal Length",float(df["petal length (cm)"].min()),float(df["petal length (cm)"].max()))
petal_width=st.sidebar.slider("Petal Width",float(df["petal width (cm)"].min()),float(df["petal width (cm)"].max()))  


input_data=[[sepal_length,sepal_width,petal_length,petal_width]]
prediction=model.predict(input_data)
st.write(f"Predicted Species: {target_names[prediction[0]]}")
