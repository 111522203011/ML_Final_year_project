import matplotlib
matplotlib.use("Agg")   # MUST be first

import matplotlib.pyplot as plt
import os
from transformers import pipeline

# Ensure charts folder exists
os.makedirs("static/charts", exist_ok=True)

# Load BERT sentiment model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Generic electronics aspects
ASPECTS = [
    "battery", "heating", "camera", "display", "screen",
    "performance", "speed", "price", "quality",
    "sound", "design", "buttons", "charging"
]


def generate_pie_chart(pos, neg, neu):
    labels = ['Positive', 'Negative', 'Neutral']
    values = [pos, neg, neu]

    plt.figure()
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title('Customer Sentiment Distribution')

    path = "static/charts/sentiment_pie.png"
    plt.savefig(path)
    plt.close()

    return "/static/charts/sentiment_pie.png"   # return URL


def generate_bar_chart(negative_keywords):
    if not negative_keywords:
        return None

    aspects = negative_keywords
    counts = [1] * len(aspects)   # simple presence chart

    plt.figure()
    plt.bar(aspects, counts)
    plt.xlabel("Product Aspects")
    plt.ylabel("Reported Issues")
    plt.title("Most Reported Product Issues")
    plt.xticks(rotation=45)

    path = "static/charts/issue_bar.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return "/static/charts/issue_bar.png"   # return URL


def analyze_reviews(reviews):

    positive_reviews = []
    negative_reviews = []
    neutral_reviews = []

    # ✅ USE SETS INSTEAD OF DICTIONARIES
    keywords = {
        "positive": set(),
        "negative": set(),
        "neutral": set()
    }

    for review in reviews:
        text = review.lower()
        result = sentiment_model(review)[0]
        label = result["label"]
        score = result["score"]

        if label == "POSITIVE" and score >= 0.65:
            positive_reviews.append(review)
            sentiment = "positive"

        elif label == "NEGATIVE" and score >= 0.65:
            negative_reviews.append(review)
            sentiment = "negative"

        else:
            neutral_reviews.append(review)
            sentiment = "neutral"

        # Aspect extraction
        for aspect in ASPECTS:
            if aspect in text:
                keywords[sentiment].add(aspect)

    # Convert sets to list for JSON response
    keywords = {
        "positive": list(keywords["positive"]),
        "negative": list(keywords["negative"]),
        "neutral": list(keywords["neutral"])
    }

    # Generate charts
    pie_chart = generate_pie_chart(
        len(positive_reviews),
        len(negative_reviews),
        len(neutral_reviews)
    )

    bar_chart = generate_bar_chart(keywords["negative"])

    # Recommendation logic
    if len(negative_reviews) > len(positive_reviews):
        recommendation = "Urgent improvements required."
    elif len(negative_reviews) > 0:
        recommendation = "Minor issues detected. Improve negative aspects."
    else:
        recommendation = "Overall positive feedback. Maintain quality."

    return {
        "sentiment_counts": {
            "positive": len(positive_reviews),
            "negative": len(negative_reviews),
            "neutral": len(neutral_reviews)
        },
        "sentiment_examples": {
            "positive": positive_reviews,
            "negative": negative_reviews,
            "neutral": neutral_reviews
        },
        "keywords": keywords,
        "charts": {
            "sentiment_pie_url": pie_chart,
            "issue_bar_url": bar_chart
        },
        "recommendation": recommendation
    }