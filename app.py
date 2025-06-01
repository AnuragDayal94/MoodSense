# emotion_predictor_app.py
import streamlit as st
import joblib

# Load custom model and vectorizer
emotion_engine = joblib.load("my_emotion_classifier.pkl")
text_transformer = joblib.load("text_vector_features.pkl")

# Custom emoji map
mood_icons = {
    'joy': '😊', 'sadness': '😢', 'anger': '😡',
    'fear': '😨', 'love': '❤️', 'surprise': '😲'
}

# UI
st.title("MoodSense - Text Emotion Decoder")
st.write("Type something and get a glimpse of the feeling it conveys.")

user_text = st.text_area("Enter your message below:")

if st.button("Analyze Emotion"):
    if user_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        cleaned_input = user_text.lower()
        transformed_input = text_transformer.transform([cleaned_input])
        mood = emotion_engine.predict(transformed_input)[0]
        mood_icon = mood_icons.get(mood, "")
        st.success(f"**Emotion Identified:** {mood.capitalize()} {mood_icon}")
