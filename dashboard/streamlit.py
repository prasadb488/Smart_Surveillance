import pandas as pd
import streamlit as st


st.title("Smart Surveillance Fatigue Dashboard")

df = pd.read_csv("data/log.csv", names=["Timestamp", "Status"])

st.subheader("Fatigue Events")
st.dataframe(df.tail(20))

st.subheader("Fatigue Summary")
st.write(df["Status"].value_counts())
st.bar_chart(df["Status"].value_counts())
