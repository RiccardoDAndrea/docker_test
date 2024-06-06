import streamlit as st
st.markdown("# This is a header")
st.write("This is a simple Streamlit app.")# Path: app.py
st.markdown("# Das wurde aus dem Internet abefragt")
st.image("https://static.streamlit.io/examples/cat.jpg", caption="A cat")

st.markdown("# Das wurde aus den Lokalen Datei abefragt")
st.image("cat.jpg", caption="A cat")