import streamlit as st

# Page config: custom tab title and favicon
st.set_page_config(
    page_title="GuestPulses – Hospitality review intelligence. Sentiment, emotions, themes, exports.",
    page_icon="guestpulses_icon.ico",  
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display the logo and title
col1, col2 = st.columns([1, 8])
with col1:
    st.image("guestpulses.png", width=70)
with col2:
    st.markdown(
        "<h1 style='margin-bottom: 0; color: #4B8BBE;'>GuestPulses</h1>"
        "<p style='margin-top: 0;'>Hospitality review intelligence. Upload CSV exports of guest reviews to generate sentiment, emotions, keyword frequency, top phrases, and exportable results.</p>",
        unsafe_allow_html=True
    )

import pandas as pd
import re
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK assets
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Initialisations
sid = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# === Expanded Bonfanti-based Category Classifier ===
def classify_category(review):
    review = review.lower()

    service_keywords = [
        'waiter', 'waitress', 'staff', 'team', 'host', 'hostess', 'manager',
        'service', 'attentive', 'helpful', 'friendly', 'welcoming', 'hospitality',
        'rude', 'slow service', 'excellent service', 'poor service', 'outstanding service',
        'unprofessional', 'inattentive', 'rushed', 'accommodating', 'ignored', 'prompt'
    ]

    personalisation_keywords = [
        'remembered', 'recognised', 'personal touch', 'special treatment',
        'birthday', 'anniversary', 'celebration', 'customised', 'tailored',
        'personalised', 'went the extra mile', 'made us feel special',
        'genuine attention', 'bespoke', 'individual service', 'surprised us'
    ]

    booking_keywords = [
        'book', 'booking', 'reservation', 'reserved', 'no reservation', 'walk-in',
        'table', 'online booking', 'open table', 'opentable', 'hard to book',
        'easy to book', 'available slot', 'double booked', 'late reservation',
        'queue', 'waited', 'long wait', 'reservation confirmed', 'booking link'
    ]

    atmosphere_keywords = [
        'ambience', 'atmosphere', 'environment', 'decor', 'lighting', 'vibe', 'setting',
        'romantic', 'cosy', 'elegant', 'intimate', 'noisy', 'quiet', 'music', 'background music',
        'comfortable', 'interior', 'outdoor seating', 'terrace', 'patio', 'garden', 'charming',
        'rustic', 'modern decor', 'aesthetic', 'busy atmosphere'
    ]

    food_keywords = [
        'food', 'dish', 'meal', 'menu', 'course', 'starter', 'main', 'dessert',
        'drink', 'wine', 'cocktail', 'beverage', 'plate', 'flavour', 'taste',
        'delicious', 'tasty', 'bland', 'overcooked', 'undercooked', 'dry', 'greasy',
        'fresh', 'portion', 'presentation', 'cuisine', 'menu variety', 'plating',
        'specials', 'daily special', 'chef', 'seasonal ingredients', 'creative dishes',
        'traditional', 'authentic', 'home cooked', 'homemade', 'spices',
        'amazing quality', 'pretty decent', 'solid choice', 'surprisingly good', 'loved it'
    ]

    value_keywords = [
        'price', 'cost', 'expensive', 'cheap', 'affordable', 'reasonable', 'overpriced',
        'value for money', 'great value', 'worth it', 'not worth it', 'costly',
        'bill', 'charged', 'money well spent', 'too pricey', 'pricing', 'good deal',
        'high prices', 'budget-friendly', 'waste of money', 'rip-off'
    ]

    loyalty_keywords = [
        'return', 'again', 'next time', 'revisit', 'coming back', 'repeat visit',
        'regular customer', 'every week', 'our favourite', 'go-to place',
        'will be back', 'always come back', 'recommend', 'bring friends',
        'loyal', 'our spot', 'my local', 'worth returning', 'five stars',
        'would recommend', 'highly recommend', 'buy again', 'best purchase',
        'top-notch experience', 'exceeded my expectations', 'solid choice'
    ]

    digital_keywords = [
        'website', 'email', 'online', 'booking system', 'confirmation email',
        'social media', 'instagram', 'facebook', 'google', 'wifi', 'app',
        'qr code', 'digital menu', 'digital booking', 'webpage', 'link', 'meta ad'
    ]

    # Run checks in order of most frequent/impactful matches
    if any(kw in review for kw in service_keywords):
        return 'Service Quality'
    elif any(kw in review for kw in personalisation_keywords):
        return 'Personalisation & Guest Recognition'
    elif any(kw in review for kw in booking_keywords):
        return 'Booking Experience'
    elif any(kw in review for kw in atmosphere_keywords):
        return 'Atmosphere & Environment'
    elif any(kw in review for kw in food_keywords):
        return 'Food & Beverage Quality'
    elif any(kw in review for kw in value_keywords):
        return 'Value for Money'
    elif any(kw in review for kw in loyalty_keywords):
        return 'Loyalty & Return Intention'
    elif any(kw in review for kw in digital_keywords):
        return 'Digital Interaction'
    else:
        return 'Other'

# === Sentiment and Emotion Tools ===
def detect_sentiment(text):
    if not text or len(text.strip()) < 3:
        return 'Neutral'

    # VADER compound scoring
    vader_score = sid.polarity_scores(text)
    compound = vader_score['compound']

    # Adjusted thresholds based on empirical VADER tuning
    if compound >= 0.5:
        return 'Very Positive'
    elif 0.2 <= compound < 0.5:
        return 'Positive'
    elif -0.2 < compound < 0.2:
        # Optional: Use TextBlob when VADER is neutral
        blob_polarity = TextBlob(text).sentiment.polarity
        if blob_polarity >= 0.3:
            return 'Positive'
        elif blob_polarity <= -0.3:
            return 'Negative'
        else:
            return 'Neutral'
    elif -0.5 <= compound <= -0.2:
        return 'Negative'
    else:
        return 'Very Negative'


def detect_emotion(text):
    text = text.lower()

    emotion_keywords = {
        'Happy': [
            'happy', 'pleased', 'joyful', 'satisfied', 'smiling', 'grateful',
            'delight', 'cheerful', 'content', 'glad', 'joy', 'wonderful',
            'pleasure', 'brightened my day', 'uplifted', 'charming', 'positive experience'
        ],
        'Angry': [
            'angry', 'furious', 'outraged', 'enraged', 'livid', 'hostile',
            'irritated', 'rude', 'argument', 'aggressive', 'tension', 'mad'
        ],
        'Frustrated': [
            'frustrated', 'annoyed', 'disappointed', 'irritated', 'let down',
            'helpless', 'misunderstood', 'stressful', 'confused', 'complicated',
            'nobody listened', 'nobody helped', 'no support', 'clueless staff'
        ],
        'Excited': [
            'excited', 'thrilled', 'amazing', 'exhilarated', 'pumped', 'energetic',
            'enthusiastic', 'can’t wait', 'fantastic', 'buzzing', 'euphoric',
            'excitement', 'impressive'
        ],
        'Disappointed': [
            'disappointed', 'underwhelmed', 'not as expected', 'let down',
            'unsatisfied', 'lacklustre', 'not worth it', 'waste', 'regret', 'unimpressed'
        ],
        'Grateful': [
            'thankful', 'grateful', 'appreciate', 'gratitude', 'thank you',
            'so kind', 'much appreciated', 'heartfelt', 'touched', 'great service'
        ],
        'Embarrassed': [
            'awkward', 'embarrassed', 'ashamed', 'uncomfortable', 'felt judged',
            'put on the spot', 'humiliating', 'cringe', 'not professional'
        ],
        'Surprised': [
            'surprised', 'unexpected', 'shocked', 'stunned', 'caught off guard',
            'astonished', 'didn’t expect', 'unexpectedly good', 'wasn’t prepared'
        ],
        'Afraid': [
            'afraid', 'scared', 'fearful', 'intimidated', 'unsafe',
            'threatened', 'worried', 'nervous', 'anxious', 'panic'
        ]
    }

    for emotion, keywords in emotion_keywords.items():
        if any(kw in text for kw in keywords):
            return emotion

    return 'Neutral'

def generate_justification(row):
    return f"This review is classified as {row['Category']} due to the review content and sentiment analysis."

# === Text Preprocessing for Keyword Frequency ===
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = nltk.word_tokenize(text)
    filtered = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return filtered

def generate_ngrams(words, n):
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def detect_sentiment_score_source(text):
    if not text or len(text.strip()) < 3:
        return 'Neutral', 0.0, 'Empty'
    text_clean = re.sub(r"\bnot (\w+)", r"not_\\1", text.lower())
    vader_score = sid.polarity_scores(text_clean)
    compound = vader_score['compound']
    source = "VADER"
    if compound >= 0.5:
        sentiment = 'Very Positive'
    elif 0.2 <= compound < 0.5:
        sentiment = 'Positive'
    elif -0.2 < compound < 0.2:
        blob_polarity = TextBlob(text).sentiment.polarity
        source = "TextBlob"
        if blob_polarity > 0.3:
            sentiment = 'Positive'
        elif blob_polarity < -0.3:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
    elif -0.5 <= compound <= -0.2:
        sentiment = 'Negative'
    else:
        sentiment = 'Very Negative'
    return sentiment, round(compound, 3), source


def detect_emotion_with_trigger(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    lemmatised_text = set([lemmatizer.lemmatize(w) for w in words])

    emotion_keywords = {
        'Happy': ['happy', 'pleased', 'joyful', 'satisfied', 'smiling', 'grateful',
                  'delight', 'cheerful', 'content', 'glad', 'joy', 'wonderful',
                  'pleasure', 'brightened my day', 'uplifted', 'charming', 'positive experience'],
        'Angry': ['angry', 'furious', 'outraged', 'enraged', 'livid', 'hostile',
                  'irritated', 'rude', 'argument', 'aggressive', 'tension', 'mad'],
        'Frustrated': ['frustrated', 'annoyed', 'disappointed', 'irritated', 'let down',
                       'helpless', 'misunderstood', 'stressful', 'confused', 'complicated',
                       'nobody listened', 'nobody helped', 'no support', 'clueless staff'],
        'Excited': ['excited', 'thrilled', 'amazing', 'exhilarated', 'pumped', 'energetic',
                    'enthusiastic', 'can’t wait', 'fantastic', 'buzzing', 'euphoric',
                    'excitement', 'impressive'],
        'Disappointed': ['disappointed', 'underwhelmed', 'not as expected', 'let down',
                         'unsatisfied', 'lacklustre', 'not worth it', 'waste', 'regret', 'unimpressed'],
        'Grateful': ['thankful', 'grateful', 'appreciate', 'gratitude', 'thank you',
                     'so kind', 'much appreciated', 'heartfelt', 'touched', 'great service'],
        'Embarrassed': ['awkward', 'embarrassed', 'ashamed', 'uncomfortable', 'felt judged',
                        'put on the spot', 'humiliating', 'cringe', 'not professional'],
        'Surprised': ['surprised', 'unexpected', 'shocked', 'stunned', 'caught off guard',
                      'astonished', 'didn’t expect', 'unexpectedly good', 'wasn’t prepared'],
        'Afraid': ['afraid', 'scared', 'fearful', 'intimidated', 'unsafe',
                   'threatened', 'worried', 'nervous', 'anxious', 'panic']
    }

    for emotion, keywords in emotion_keywords.items():
        match = [kw for kw in keywords if kw in lemmatised_text]
        if match:
            return emotion, ", ".join(match)
    return 'Neutral', ''


# === Streamlit App UI ===
st.subheader("Review Classifier with Sentiment, Emotion & Keyword Report")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df_raw.head())

    st.subheader("Map Your Columns (optional except for Review)")
    col_options = ["None"] + df_raw.columns.tolist()
    required_col = df_raw.columns.tolist()

    name_col = st.selectbox("Column for Name", col_options)
    date_col = st.selectbox("Column for Date", col_options)
    review_col = st.selectbox("Column for Review Text (Required)", required_col)
    rating_col = st.selectbox("Column for Rating", col_options)

    if st.button("Run Sentiment & Category Analysis"):
        df = pd.DataFrame()
        df['Review'] = df_raw[review_col].astype(str)
        df['Name'] = df_raw[name_col] if name_col != "None" else ""
        df['Date'] = df_raw[date_col] if date_col != "None" else ""
        df['Rating'] = df_raw[rating_col] if rating_col != "None" else ""

        df['Category'] = df['Review'].apply(classify_category)
        df['Sentiment'] = df['Review'].apply(detect_sentiment)
        df['Sentiment'], df['Sentiment Score'], df['Sentiment Source'] = zip(*df['Review'].map(detect_sentiment_score_source))
        df['Emotion'], df['Emotion Trigger'] = zip(*df['Review'].map(detect_emotion_with_trigger))
        df['Justification'] = df.apply(generate_justification, axis=1)

        st.success("Analysis Complete!")
        st.write("### Classified Results")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Analysed Reviews", csv, "classified_reviews.csv", "text/csv")

    st.subheader("Keyword & Phrase Frequency Report")

    if st.button("Generate Keyword & Phrase Report"):
        all_reviews = " ".join(df_raw[review_col].dropna().astype(str).tolist())
        processed_words = preprocess_text(all_reviews)

        unigrams = Counter(processed_words)
        bigrams = Counter(generate_ngrams(processed_words, 2))
        trigrams = Counter(generate_ngrams(processed_words, 3))

        combined = unigrams + bigrams + trigrams
        df_keywords = pd.DataFrame(combined.items(), columns=['Keyword/Phrase', 'Total'])
        df_keywords = df_keywords.sort_values(by='Total', ascending=False).reset_index(drop=True)

        st.success("Keyword Extraction Complete!")
        st.write("### Top Keywords and Phrases")
        st.dataframe(df_keywords)

        keyword_csv = df_keywords.to_csv(index=False).encode('utf-8')
        st.download_button("Download Keyword Report", keyword_csv, "keyword_frequency_report.csv", "text/csv")



