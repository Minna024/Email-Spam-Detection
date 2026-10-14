import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Example training data
emails = [
    "Congratulations you won a free prize",
    "Claim your free cash reward now",
    "Click here to win money",
    "You have won a lottery",
    "Limited offer buy now",
    "Hello, how are you?",
    "Please send me the homework",
    "The meeting is scheduled for tomorrow",
    "Can we talk after class?",
    "Thank you for your email",
]

labels = [
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Not Spam", "Not Spam", "Not Spam",
    "Not Spam", "Not Spam"
]

# Build and train the model
model = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)
model.fit(emails, labels)

# Website interface
st.title("📧 Email Spam Detection")
st.write("Enter an email below to check whether it is spam.")

message = st.text_area("Paste your email here:")

if st.button("Check Email"):
    if message.strip():
        prediction = model.predict([message])[0]

        if prediction == "Spam":
            st.error("⚠️ This email may be SPAM!")
        else:
            st.success("✅ This email may be NOT SPAM.")
    else:
        st.warning("Please enter an email first.")
