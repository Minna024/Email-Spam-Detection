
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ----------------------------------
# 1. TRAINING DATA
# ----------------------------------

spam_emails = [
    "Congratulations you won a free prize",
    "Claim your free cash reward now",
    "Click here to win money",
    "You have won a lottery",
    "Limited offer buy now",
    "URGENT your bank account will be blocked verify your password",
    "Your account will be suspended unless you confirm your bank details",
    "Pay a redelivery fee immediately using this link",
    "Your parcel is waiting pay now to release your package",
    "Your email was selected to receive a million dollar reward",
    "Send your banking details to claim your prize",
    "Your account has unusual activity click this unknown link now",
    "You owe an unpaid fee pay immediately to avoid penalties",
    "Your computer is infected call this number immediately",
    "Earn guaranteed money with no experience and no effort",
    "Your password expires today confirm it through this link",
    "You have inherited money from a stranger reply with your details",
    "Congratulations claim your gift card by entering your credit card",
    "Exclusive investment guarantees you double your money overnight",
    "Your bank needs your PIN to stop a suspicious transaction",
    "You have won a free phone pay shipping fees to receive it",
    "Final warning your account will close in one hour",
    "Act now to unlock your cash bonus",
    "Your payment failed click here and enter your card details",
    "A stranger has sent you money claim it with your password",
    "You are selected for a prize but must pay a processing fee",
    "Get rich quickly with guaranteed daily profits",
    "Your tax refund is waiting enter your banking password",
    "Your account is locked verify your identity at this link",
    "Free luxury holiday claim now by paying a small fee",
    "Reply with your OTP to prevent your account from closing",
    "Your package cannot be delivered until you pay this urgent fee",
    "You have been chosen as a winner in an unknown lottery",
    "Click immediately to avoid losing your reward",
    "Your bank account needs verification send your PIN now",
    "Earn thousands daily by investing a small amount today",
    "You have an unpaid fine pay through this suspicious link",
    "Claim your cash prize by sharing your credit card number",
    "Your subscription is cancelled unless you provide your password",
    "Urgent security alert login through this link to avoid account closure",
    "Congratulations you are eligible for free money send your details",
    "Your wallet is blocked share your secret recovery phrase",
    "Pay a small fee to unlock your unexpected inheritance",
    "Your refund is ready provide your card number and PIN",
    "Get a free tablet by paying a processing charge now",
    "You won a huge jackpot in a lottery you never entered",
    "Immediate action required transfer money to protect your account",
    "Confirm your account through this unknown link to receive cash",
    "Your bank representative needs your one time password",
    "Risk free investment guarantees enormous profits overnight"
]

ham_emails = [
    "Hello how are you",
    "Please send me the homework",
    "The meeting is scheduled for tomorrow",
    "Can we talk after class",
    "Thank you for your email",
    "Please find the attached assignment",
    "The teacher uploaded the class notes",
    "Can you share the presentation slides",
    "Our examination timetable is available",
    "Your appointment is confirmed for Monday",
    "Your bank statement is ready in the official banking app",
    "Your parcel has been delivered successfully",
    "Your password was changed successfully",
    "Please review the report before the meeting",
    "Congratulations you have been selected for the school debate",
    "Free snacks will be available after the club meeting",
    "The teacher announced a special prize for the quiz winner",
    "Our school is offering a free career guidance session",
    "The library is giving free access to online learning resources",
    "Please confirm your attendance for tomorrow's meeting",
    "Your monthly statement is available through our official app",
    "Your delivery is scheduled for tomorrow afternoon",
    "Thank you for contacting customer support",
    "I will send you the project file tonight",
    "The class presentation was excellent congratulations",
    "Please bring your textbook to school tomorrow",
    "The club meeting has been moved to Friday",
    "Your payment was received successfully",
    "The school will remain closed tomorrow",
    "I have attached the requested document",
    "Your order has been shipped and is on its way",
    "Please call me when you are available",
    "We need to discuss our commerce project",
    "The teacher has shared the assignment deadline",
    "Your appointment reminder is for ten AM",
    "You can reset your password using the official website",
    "The bank will never ask you to share your PIN by email",
    "Please check the meeting agenda before joining",
    "Your library book is due next week",
    "Thank you for helping me with my homework",
    "The school competition offers prizes to participants",
    "You can get free educational materials from the school library",
    "Your refund request has been received and is being reviewed",
    "Please sign in through the official app to view your statement",
    "The team has completed the project successfully",
    "Our presentation is scheduled for next Wednesday",
    "I received your email yesterday",
    "Please let me know if you can attend",
    "The package has arrived at the delivery centre",
    "We will discuss the results during our next meeting"
]

emails = spam_emails + ham_emails
labels = (
    ["Spam"] * len(spam_emails)
    + ["Not Spam"] * len(ham_emails)
)

# ----------------------------------
# 2. TRAIN THE MODEL
# ----------------------------------

model = make_pipeline(
    CountVectorizer(ngram_range=(1, 2)),
    MultinomialNB(alpha=0.2)
)

model.fit(emails, labels)

# ----------------------------------
# 3. WEBSITE INTERFACE
# ----------------------------------

st.title("📧 Email Spam Detection")
st.write(
    "Paste an email to check whether it may be spam."
)

st.warning(
    "Educational demonstration only. "
    "This model can make incorrect predictions."
)

message = st.text_area(
    "Paste your email here:",
    height=180,
    placeholder="Enter the complete email message..."
)

if st.button("🔍 Check Email", use_container_width=True):
    if not message.strip():
        st.warning("Please enter an email first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = model.classes_

        spam_index = list(classes).index("Spam")
        spam_probability = probabilities[spam_index] * 100

        if prediction == "Spam":
            st.error("⚠️ This email may be SPAM!")
        else:
            st.success("✅ This email may be NOT SPAM.")

        st.metric(
            "Estimated spam probability",
            f"{spam_probability:.1f}%"
        )

        st.progress(int(spam_probability))

        st.caption(
            "This percentage is the model's estimate, "
            "not a guarantee of whether the email is dangerous."
        )

# ----------------------------------
# 4. BUILT-IN TRICKY TESTS
# ----------------------------------

st.divider()
st.subheader("🧪 Test the Model")

test_emails = {
    "Suspicious bank verification": (
        "Your account will be suspended today. "
        "Reply with your PIN and OTP to keep it active.",
        "Spam"
    ),
    "Fake delivery charge": (
        "Your parcel is held. Pay a small redelivery fee "
        "immediately through this link.",
        "Spam"
    ),
    "Prize with a fee": (
        "You have won a prize in a lottery you never entered. "
        "Pay the processing fee to receive your reward.",
        "Spam"
    ),
    "Normal school prize": (
        "Congratulations you won first prize in our school quiz. "
        "The certificate will be given at assembly.",
        "Not Spam"
    ),
    "Normal free class": (
        "Join our free online commerce revision class tomorrow. "
        "The meeting link will be shared by the teacher.",
        "Not Spam"
    ),
    "Normal delivery": (
        "Your parcel has been shipped and is expected to arrive "
        "tomorrow afternoon.",
        "Not Spam"
    )
}

for name, (test_message, expected) in test_emails.items():
    with st.expander(name):
        st.write(test_message)
        if st.button("Test this email", key=name):
            result = model.predict([test_message])[0]

            st.write("Expected result:", expected)
            st.write("Model prediction:", result)

            if result == expected:
                st.success("Prediction matches the expected label.")
            else:
                st.error(
                    "Incorrect prediction. The model needs better "
                    "training data."
                )

st.divider()
st.caption(
    f"Training examples: {len(emails)} "
    f"({len(spam_emails)} spam, {len(ham_emails)} not spam)"
)

