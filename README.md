# GuestPulses

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-success)
![License](https://img.shields.io/github/license/sc-aisuperjack/guestpulses)

Hospitality review intelligence. Sentiment, emotions, themes, exports.

GuestPulses is a Streamlit app for restaurants and hospitality teams. Upload a CSV or Excel export of guest reviews, map your columns, and generate analysis outputs including hospitality category classification, sentiment (with score and source), emotion detection (with trigger words), and keyword and phrase frequency reports. :contentReference[oaicite:2]{index=2}

## What GuestPulses does

GuestPulses produces two exportable outputs:

1. **Classified Reviews Export**

- Hospitality category per review (Service Quality, Booking Experience, Food & Beverage Quality, Atmosphere, Value for Money, Loyalty, Digital Interaction, and more). :contentReference[oaicite:3]{index=3}
- Sentiment label per review (Very Positive to Very Negative)
- Sentiment score and scoring source (VADER or TextBlob fallback)
- Emotion label per review (keyword based) and the trigger word(s) detected
- A short justification string per review :contentReference[oaicite:4]{index=4}

2. **Keyword and Phrase Frequency Report**

- Unigram, bigram (2 word), and trigram (3 word) frequency counts
- Output is sortable and exportable as CSV :contentReference[oaicite:5]{index=5}

## Input files supported

- CSV (`.csv`)
- Excel (`.xlsx`) :contentReference[oaicite:6]{index=6}

## Column mapping

After upload, you can optionally map:

- Name (optional)
- Date (optional)
- Rating (optional)
- Review Text (required) :contentReference[oaicite:7]{index=7}

This means your source export does not need fixed column names.

## Outputs

When you run the analysis, GuestPulses lets you download:

- `classified_reviews.csv`
- `keyword_frequency_report.csv` :contentReference[oaicite:8]{index=8}

## Live demo

Add your link here once deployed:

- App: https://guestpulses.streamlit.app
- Repo: https://github.com/sc-aisuperjack/guestpulses

## Run locally

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```
