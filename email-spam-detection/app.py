
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# ----------------------------------
# 1. TRAINING DATA
# ----------------------------------

emails = [
    # SPAM EMAILS
    "Congratulations you won a free prize",
    "Claim your free cash reward now",
    "Click here to win money",
    "You have won a lottery",
    "Limited offer buy now",
    "URGENT claim your free gift today",
    "You are the lucky winner of 1000000 dollars",
    "Win a brand new iPhone for free",
    "Get rich quickly with this amazing opportunity",
    "Congratulations your account has won a reward",
    "Earn thousands of dollars from home instantly",
    "Free vacation waiting for you click now",
    "Claim your cash bonus before it expires",
    "Exclusive deal buy now and save big",
    "You have been selected for a special prize",
    "Act now to receive your free shopping voucher",
    "Make money fast with no effort required",
    "You won a free laptop enter your details",
    "Your prize is waiting confirm your information",
    "Get a free gift card by clicking this link",
    "Lowest prices ever buy immediately",
    "Congratulations you have been chosen as a winner",
    "Double your money in just one day",
    "Limited time offer claim your reward now",
    "You are eligible for a free cash prize",
    "Click this link to unlock your bonus",
    "Get instant approval for an unbelievable loan",
    "Free shopping coupons available today",
    "Your lucky number has won a huge jackpot",
    "Send your bank details to receive your reward",
    "Act fast your exclusive offer ends tonight",
    "Congratulations claim your million dollar reward",
    "Get free products by registering today",
    "You have won a brand new car",
    "Earn money online with guaranteed profits",
    "Your account qualifies for a free bonus",
    "Claim your special discount immediately",
    "You have received an unexpected cash gift",
    "Win exciting prizes in our exclusive giveaway",
    "Click now to collect your free reward",

    # NOT SPAM EMAILS
    "Hello how are you",
    "Please send me the homework",
    "The meeting is scheduled for tomorrow",
    "Can we talk after class",
    "Thank you for your email",
    "Please find the attached assignment",
    "I will reach school at eight thirty",
    "Can you share your class notes",
    "The project submission deadline is Friday",
    "Please confirm the meeting time",
    "I have completed the homework",
    "Let us study together this evening",
    "The teacher has uploaded the notes",
    "I will call you after lunch",
    "Please bring your textbook tomorrow",
    "Our class will start at nine",
    "Thank you for helping me with the project",
    "I have attached the requested document",
    "Could you explain the last chapter",
    "The examination timetable has been announced",
    "Your appointment is confirmed for Monday",
    "This is a reminder about our team meeting",
    "Please review the report before submission",
    "I received your message yesterday",
    "The school will remain closed tomorrow",
    "Can you send me the presentation slides",
    "I will be absent from class today",
    "Please let me know when you are available",
    "The assignment looks good to me",
    "We need to discuss our project",
    "Your order has been shipped successfully",
    "Your delivery is scheduled for tomorrow",
    "Your payment was completed successfully",
    "Your bank statement is now available",
    "Your password was changed successfully",
    "Your library book is due next week",
    "The package has arrived at the delivery centre",
    "Your appointment reminder is for ten AM",
    "Thank you for contacting customer support",
    "Your subscription renewal is confirmed"
]

labels = (
    ["Spam"] * 40
    + ["Not Spam"] * 40
)

# ----------------------------------
# 2. BUILD AND TRAIN THE MODEL
# ----------------------------------

model = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

model.fit(emails, labels)

# ----------------------------------
# 3. STREAMLIT WEBSITE INTERFACE
# ----------------------------------

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Detection")

st.write(
    "Enter an email message below to check "
    "whether it may be spam or not spam."
)

message = st.text_area(
    "Paste your email here:",
    height=180,
    placeholder="Type or paste an email message..."
)

if st.button("🔍 Check Email", use_container_width=True):

    if message.strip():

        prediction = model.predict([message])[0]

        probabilities = model.predict_proba([message])[0]
        confidence = max(probabilities) * 100

        if prediction == "Spam":

            st.error("⚠️ This email may be SPAM!")

        else:

            st.success("✅ This email may be NOT SPAM.")

        st.write(
            f"**Model confidence:** {confidence:.1f}%"
        )

        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee that the classification is correct."
        )

    else:

        st.warning("Please enter an email first.")

st.divider()

st.caption(
    "Email Spam Detection Project | "
    "Uses CountVectorizer and Multinomial Naive Bayes"
)

