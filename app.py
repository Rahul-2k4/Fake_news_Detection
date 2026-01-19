import gradio as gr
import pickle

# Load the trained pipeline model
with open('model_new.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_fake_news(news_text):
    """Predict if the news is real or fake."""
    if not news_text or not news_text.strip():
        return "⚠️ Please enter some text to analyze.", "N/A", ""
    
    # Predict using the pipeline
    prediction = model.predict([news_text])[0]
    probability = model.predict_proba([news_text])[0]
    
    # Model outputs: 0 = Fake, 1 = Real
    if prediction == 1:
        result = "✅ LIKELY REAL"
        confidence = probability[1] * 100
        explanation = "The linguistic patterns in this text are consistent with factual reporting."
    else:
        result = "❌ LIKELY FAKE"
        confidence = probability[0] * 100
        explanation = "The linguistic patterns suggest potential misinformation. Verify with trusted sources."
    
    return result, f"{confidence:.1f}%", explanation

# Example news for testing
examples = [
    ["Scientists at NASA have confirmed the discovery of water on Mars."],
    ["BREAKING: Government secretly installing mind-control chips in smartphones!"],
    ["The Federal Reserve announced a 0.25% interest rate increase today."],
    ["EXPOSED: Celebrity admits to being a reptilian alien!"],
    ["New research shows regular exercise reduces heart disease risk."],
]

# Create Gradio interface (Gradio 6 compatible)
demo = gr.Interface(
    fn=predict_fake_news,
    inputs=gr.Textbox(
        label="📰 Enter News Text",
        placeholder="Paste a news headline or article here...",
        lines=5
    ),
    outputs=[
        gr.Textbox(label="🎯 Prediction"),
        gr.Textbox(label="📊 Confidence"),
        gr.Textbox(label="💡 Explanation")
    ],
    title="🔍 Fake News Detection System",
    description="""
## AI-Powered Fake News Classifier

Analyzes linguistic patterns to detect potentially fake news.
Trained on 10,000+ labeled news statements using TF-IDF + Logistic Regression.

**Disclaimer:** Educational tool only. Always verify news through trusted sources.

[GitHub](https://github.com/Rahul-2k4/Fake_news_Detection) | Built by [Rahul Tripathi](https://github.com/Rahul-2k4)
    """,
    examples=examples,
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()
