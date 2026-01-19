import gradio as gr
import pickle
import numpy as np

# Load the trained pipeline model (includes TfidfVectorizer + LogisticRegression)
with open('final_model.sav', 'rb') as f:
    model = pickle.load(f)

def predict_fake_news(news_text):
    """Predict if the news is real or fake."""
    if not news_text or not news_text.strip():
        return "⚠️ Please enter some text to analyze.", "N/A", ""
    
    # Predict using the pipeline (handles vectorization internally)
    prediction = model.predict([news_text])[0]
    probability = model.predict_proba([news_text])[0]
    
    # Model outputs: 0 = Fake, 1 = Real (based on LIAR dataset labels)
    if prediction == 1 or prediction == "TRUE":
        result = "✅ LIKELY REAL"
        confidence = probability[1] * 100 if len(probability) > 1 else probability[0] * 100
        explanation = "The linguistic patterns in this text are consistent with factual reporting."
    else:
        result = "❌ LIKELY FAKE"
        confidence = probability[0] * 100
        explanation = "The linguistic patterns suggest potential misinformation. Verify with trusted sources."
    
    return result, f"{confidence:.1f}%", explanation

# Example news for testing
examples = [
    ["Scientists at NASA have confirmed the discovery of water on Mars, marking a significant milestone in the search for extraterrestrial life."],
    ["BREAKING: Government secretly installing mind-control chips in all new smartphones to monitor citizens' thoughts!"],
    ["The Federal Reserve announced a 0.25% interest rate increase today, citing continued inflation concerns in the economy."],
    ["EXPOSED: Famous celebrity admits to being a reptilian alien from another dimension in leaked video!"],
    ["New research published in Nature shows that regular exercise can reduce the risk of heart disease by up to 30%."],
]

# Create Gradio interface
demo = gr.Interface(
    fn=predict_fake_news,
    inputs=[
        gr.Textbox(
            label="📰 Enter News Text",
            placeholder="Paste a news headline or article here...",
            lines=6,
            max_lines=15
        )
    ],
    outputs=[
        gr.Textbox(label="🎯 Prediction", elem_classes=["prediction-output"]),
        gr.Textbox(label="📊 Confidence Score"),
        gr.Textbox(label="💡 Explanation")
    ],
    title="🔍 Fake News Detection System",
    description="""
## AI-Powered Fake News Classifier

This machine learning model analyzes linguistic patterns to detect potentially fake news articles.
Trained on the **LIAR dataset** (20,000+ labeled statements) using **TF-IDF + Logistic Regression**.

### 📈 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | ~70% |
| F1-Score | 0.70 |
| Dataset | LIAR (Politifact) |

### ⚠️ Disclaimer
This is an educational tool. Always verify news through multiple trusted sources.
The model analyzes linguistic patterns, not factual accuracy.

---
**Built by:** [Rahul Tripathi](https://github.com/Rahul-2k4) | [View Source Code](https://github.com/Rahul-2k4/Fake_news_Detection)
    """,
    examples=examples,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
    ),
    css="""
    .prediction-output textarea {
        font-size: 1.5em !important;
        font-weight: bold !important;
    }
    """,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
