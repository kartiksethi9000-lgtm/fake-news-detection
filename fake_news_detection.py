import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import re
import string
import warnings
warnings.filterwarnings('ignore')

real_news = [
    "Scientists discover new vaccine that shows 95% effectiveness against flu virus",
    "Government announces new policy to improve public transportation infrastructure",
    "Stock markets rise as economic indicators show positive growth",
    "Researchers publish study on climate change impacts on global weather patterns",
    "Local election results certified after thorough counting process",
    "New hospital opens in city providing advanced medical care to residents",
    "University study reveals benefits of regular exercise on mental health",
    "Central bank raises interest rates to control inflation as planned",
    "International summit reaches agreement on reducing carbon emissions",
    "Tech company launches new product after years of research and development",
    "Court upholds new environmental regulation after legal challenge",
    "Scientists confirm water found on Mars surface through new rover data",
    "Government releases annual budget report with detailed expenditure plans",
    "Medical researchers make breakthrough in Alzheimer's disease treatment",
    "Trade agreement signed between two countries to boost economic ties",
    "New study shows reading books improves cognitive function in adults",
    "City council approves new housing project to address shortage",
    "Athlete breaks world record at international championship event",
    "Weather department issues forecast for upcoming monsoon season",
    "University introduces new computer science curriculum for students",
]

fake_news = [
    "Alien spacecraft lands in New York scientists paid to keep it secret",
    "Drinking bleach cures all diseases doctors do not want you to know",
    "Government puts mind control chips in COVID vaccines to track citizens",
    "Famous celebrity secretly reptilian shapeshifter exposed by insider",
    "Moon landing was completely faked in Hollywood studio leaked footage",
    "5G towers spread coronavirus government hiding the truth from public",
    "Politician caught secretly selling country to foreign power anonymous source",
    "Miracle cure discovered big pharma suppressing it to make money",
    "Secret society controls all world governments through shadow network",
    "Eating raw garlic every hour prevents all known diseases permanently",
    "Scientists admit evolution is completely fake theory exposed finally",
    "World ends next week according to ancient prophecy hidden by elites",
    "Local water supply poisoned by government to reduce population size",
    "Celebrity deaths staged actors hiding in underground bunkers revealed",
    "New world order plans to microchip entire population by next year",
    "Ancient civilization discovered underground proves history is lies",
    "Cancer cure hidden by doctors who profit from ongoing treatments",
    "Birds are not real they are government surveillance drones exposed",
    "Flat earth confirmed by leaked NASA documents scientists silenced",
    "Politician sold national secrets for personal gain insider reveals all",
]


texts  = real_news + fake_news
labels = [1] * len(real_news) + [0] * len(fake_news)   # 1=Real, 0=Fake

df = pd.DataFrame({'text': texts, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print("=" * 55)
print("       FAKE NEWS DETECTION SYSTEM — Kartik Sethi")
print("=" * 55)
print(f"\n📊 Dataset Overview")
print(f"   Total samples : {len(df)}")
print(f"   Real news     : {sum(df.label == 1)}")
print(f"   Fake news     : {sum(df.label == 0)}")


def preprocess_text(text):
    """Clean and normalize text for ML."""
    text = text.lower()                                    # lowercase
    text = re.sub(r'\d+', '', text)                        # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()              # remove extra spaces
    return text

df['clean_text'] = df['text'].apply(preprocess_text)

print(f"\n🔧 Text Preprocessing — Sample")
print(f"   Original : {df['text'][0][:60]}...")
print(f"   Cleaned  : {df['clean_text'][0][:60]}...")


X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

print(f"\n📂 Train/Test Split (80/20)")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")



tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),       
    stop_words='english',
    min_df=1
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"\n📐 TF-IDF Feature Extraction")
print(f"   Vocabulary size  : {len(tfidf.vocabulary_)}")
print(f"   Feature matrix   : {X_train_tfidf.shape}")



models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes"         : MultinomialNB(alpha=0.1),
}

results = {}
print(f"\n🤖 Model Training & Evaluation")
print("-" * 55)

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred   = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'accuracy': accuracy, 'predictions': y_pred}

    print(f"\n  ▶ {name}")
    print(f"    Accuracy : {accuracy * 100:.2f}%")
    print(f"    Classification Report:")
    report = classification_report(y_test, y_pred,
                                   target_names=['Fake', 'Real'],
                                   output_dict=True)
    print(f"      Fake — Precision: {report['Fake']['precision']:.2f}  "
          f"Recall: {report['Fake']['recall']:.2f}  "
          f"F1: {report['Fake']['f1-score']:.2f}")
    print(f"      Real — Precision: {report['Real']['precision']:.2f}  "
          f"Recall: {report['Real']['recall']:.2f}  "
          f"F1: {report['Real']['f1-score']:.2f}")


best_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_name]['model']

print(f"\n🏆 Best Model : {best_name}")
print(f"   Accuracy   : {results[best_name]['accuracy'] * 100:.2f}%")


def predict_news(article, model=best_model, vectorizer=tfidf):
    """Predict whether a news article is Real or Fake."""
    cleaned    = preprocess_text(article)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    label      = "✅ REAL NEWS" if prediction == 1 else "❌ FAKE NEWS"
    confidence = max(probability) * 100
    return label, confidence

print(f"\n{'=' * 55}")
print(f"  🔍 LIVE PREDICTION — Test on New Articles")
print(f"{'=' * 55}")

test_articles = [
    "Government releases new budget plan to improve healthcare system",
    "Secret aliens control world leaders and nobody is allowed to know",
    "Scientists develop new solar panel with 40 percent efficiency",
    "Drinking hot water with lemon cures cancer doctors are hiding this",
]

for article in test_articles:
    label, confidence = predict_news(article)
    print(f"\n  Article  : {article[:55]}...")
    print(f"  Result   : {label}  (Confidence: {confidence:.1f}%)")


