import streamlit as st
import requests

st.title("🏥 MediAssist AI")

question = st.text_input(
    "Ask a question"
)

if st.button("Search"):

    response = requests.post(
        "http://localhost:8000/chat",
        json={
            "question": question
        }
    )

    data = response.json()

    st.subheader("Retrieved Chunks")

    for item in data["results"]:

        st.write("Similarity Score:")
        st.write(item["score"])

        st.write(item["content"])

        st.divider()