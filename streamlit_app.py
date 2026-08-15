import streamlit as st
import requests

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="centered"
)

# --------------------------------------------------
# FastAPI Backend
# --------------------------------------------------

API_URL = "https://toxic-comment-classifier-7lee.onrender.com"

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}

.category-card {
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 12px;
    font-size: 17px;
}

.detected {
    background-color: #421f23;
    border: 1px solid #7f2934;
}

.safe {
    background-color: #123b29;
    border: 1px solid #1f7a4d;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🛡️ Toxic Comment Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered toxicity detection using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Comment Input
# --------------------------------------------------

st.subheader("💬 Analyze a Comment")

comment = st.text_area(
    "Enter your comment",
    placeholder="Type a comment here...",
    height=150,
    label_visibility="collapsed"
)

# --------------------------------------------------
# Buttons
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    analyze = st.button(
        "🔍 Analyze Comment",
        use_container_width=True
    )

with col2:
    clear = st.button(
        "🗑️ Clear",
        use_container_width=True
    )

if clear:
    st.rerun()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if analyze:

    if not comment.strip():

        st.warning("⚠️ Please enter a comment first.")

    else:

        with st.spinner("Analyzing comment..."):

            try:

                response = requests.post(
                    f"{API_URL}/predict",
                    json={"comment": comment},
                    timeout=60
                )

                if response.status_code == 200:

                    result = response.json()

                    is_toxic = result["is_toxic"]
                    predictions = result["predictions"]

                    # ----------------------------------
                    # Overall Result
                    # ----------------------------------

                    st.divider()

                    st.subheader("📊 Analysis Result")

                    if is_toxic:

                        st.markdown(
                            """
                            <div class="result-box detected">
                                <h2>⚠️ Toxic Comment</h2>
                                <p>The model detected toxic content.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            """
                            <div class="result-box safe">
                                <h2>✅ Non-Toxic Comment</h2>
                                <p>No toxic content was detected.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # ----------------------------------
                    # Toxicity Summary
                    # ----------------------------------

                    detected_count = sum(
                        1 for value in predictions.values()
                        if value
                    )

                    total_categories = len(predictions)

                    st.write("")

                    st.metric(
                        "Toxic Categories Detected",
                        f"{detected_count} / {total_categories}"
                    )

                    st.progress(
                        detected_count / total_categories
                    )

                    # ----------------------------------
                    # Categories
                    # ----------------------------------

                    st.subheader("🛡️ Toxicity Categories")

                    for category, value in predictions.items():

                        category_name = (
                            category
                            .replace("_", " ")
                            .title()
                        )

                        if value:

                            st.markdown(
                                f"""
                                <div class="category-card detected">
                                    🔴 <b>{category_name}</b>
                                    <br>
                                    &nbsp;&nbsp;&nbsp;&nbsp;
                                    Toxic content detected
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        else:

                            st.markdown(
                                f"""
                                <div class="category-card safe">
                                    🟢 <b>{category_name}</b>
                                    <br>
                                    &nbsp;&nbsp;&nbsp;&nbsp;
                                    No toxic content detected
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                else:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ Could not connect to the FastAPI server.\n\n{e}"
                )

# --------------------------------------------------
# Example Comments
# --------------------------------------------------

st.divider()

st.subheader("💡 Try an Example")

st.write(
    "You can copy one of these comments into the box above:"
)

st.code(
    "I really enjoyed this discussion!",
    language="text"
)

st.code(
    "You are a stupid idiot.",
    language="text"
)

st.code(
    "I disagree with your opinion.",
    language="text"
)
