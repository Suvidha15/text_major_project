from flask import Flask, render_template, request, jsonify
import pickle
import re
import numpy as np

from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

LABELS = ["Non-Stress", "Stress", "Depression", "Suicidal"]

COLORS = {
    "Non-Stress": "#22c55e",
    "Stress": "#f59e0b",
    "Depression": "#6366f1",
    "Suicidal": "#ef4444",
}

ICONS = {
    "Non-Stress": "😊",
    "Stress": "😰",
    "Depression": "😔",
    "Suicidal": "🆘",
}

INTENSITY = {
    "Non-Stress": "Calm",
    "Stress": "Tense",
    "Depression": "Low",
    "Suicidal": "Critical",
}


# ---------------------------------------------------------
# LIME EXPLAINER
# ---------------------------------------------------------

explainer = LimeTextExplainer(
    class_names=LABELS,
    random_state=42
)


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------
# PROBABILITY FUNCTION
# ---------------------------------------------------------

def get_probabilities(texts):
    """
    LIME needs a function that receives raw text
    and returns probability for every class.
    """

    cleaned_texts = [clean(text) for text in texts]
    vectors = vectorizer.transform(cleaned_texts)

    # Model already supports predict_proba
    if hasattr(model, "predict_proba"):
        return model.predict_proba(vectors)

    # For models such as LinearSVC
    # convert decision scores into softmax-like probabilities
    if hasattr(model, "decision_function"):
        scores = model.decision_function(vectors)

        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])

        scores = scores - np.max(scores, axis=1, keepdims=True)

        probabilities = np.exp(scores)
        probabilities = probabilities / np.sum(
            probabilities,
            axis=1,
            keepdims=True
        )

        return probabilities

    raise ValueError(
        "The loaded model does not support predict_proba "
        "or decision_function."
    )


# ---------------------------------------------------------
# GET MODEL PREDICTION
# ---------------------------------------------------------

def get_predictions(texts):
    cleaned_texts = [clean(text) for text in texts]
    vectors = vectorizer.transform(cleaned_texts)

    predictions = model.predict(vectors)

    probabilities = get_probabilities(texts)

    classes = list(model.classes_) if hasattr(model, "classes_") else LABELS

    return predictions, probabilities, classes


# ---------------------------------------------------------
# LIME EXPLANATION
# ---------------------------------------------------------

def explain_sentence(sentence, predicted_label, classes):
    """
    Generate word-level explanation for one sentence.
    """

    try:
        predicted_index = classes.index(predicted_label)

        explanation = explainer.explain_instance(
            sentence,
            get_probabilities,
            labels=[predicted_index],
            num_features=8,
            num_samples=1000
        )

        explanation_list = explanation.as_list(
            label=predicted_index
        )

        result = []

        for word, weight in explanation_list:

            # Ignore extremely small contributions
            if abs(weight) < 0.01:
                continue

            result.append({
                "word": word,
                "weight": round(float(weight), 4),
                "direction": "positive" if weight > 0 else "negative"
            })

        return result

    except Exception as e:
        print("LIME explanation error:", e)
        return []


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------
# ANALYZE
# ---------------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()
    paragraph = data.get("text", "").strip()

    if not paragraph:
        return jsonify({
            "error": "No text provided"
        }), 400

    # Split paragraph into sentences
    lines = [
        l.strip()
        for l in re.split(r"[.\n!?]+", paragraph)
        if len(l.strip()) > 5
    ]

    if not lines:
        return jsonify({
            "error": "Could not detect sentences"
        }), 400

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    preds, probas, classes = get_predictions(lines)

    line_results = []

    for i, (line, pred) in enumerate(zip(lines, preds)):

        pred = str(pred)

        # Find probability of predicted class
        pred_index = classes.index(pred)

        confidence = round(
            float(probas[i][pred_index]) * 100,
            1
        )

        # -------------------------------------------------
        # LIME XAI
        # -------------------------------------------------

        explanation = explain_sentence(
            line,
            pred,
            classes
        )

        line_results.append({
            "text": line,
            "label": pred,
            "color": COLORS[pred],
            "icon": ICONS[pred],
            "confidence": confidence,
            "intensity": INTENSITY[pred],

            # XAI
            "explanation": explanation
        })

    # -----------------------------------------------------
    # COUNT LABELS
    # -----------------------------------------------------

    counts = {
        label: 0
        for label in LABELS
    }

    for result in line_results:
        counts[result["label"]] += 1

    total = len(line_results)

    # -----------------------------------------------------
    # PIE DATA
    # -----------------------------------------------------

    pie_data = []

    for label in LABELS:

        count = counts[label]

        if count > 0:

            pie_data.append({
                "label": label,
                "count": count,
                "percentage": round(
                    count / total * 100,
                    1
                ),
                "color": COLORS[label],
                "icon": ICONS[label]
            })

    # -----------------------------------------------------
    # DOMINANT CLASS
    # -----------------------------------------------------

    dominant = max(
        counts,
        key=counts.get
    )

    # -----------------------------------------------------
    # CONCLUSIONS
    # -----------------------------------------------------

    conclusions = {

        "Non-Stress":
            "Your text reflects a calm and positive state of mind. "
            "You appear to be managing well emotionally. "
            "Keep nurturing this balance.",

        "Stress":
            "Your text indicates elevated stress levels. "
            "Your mind may be carrying a heavy load right now. "
            "Consider taking breaks and practising relaxation techniques.",

        "Depression":
            "Your text shows patterns associated with emotional low points. "
            "These feelings are valid and you are not alone. "
            "Please consider reaching out to someone you trust.",

        "Suicidal":
            "Your text contains deeply concerning patterns. "
            "Please speak to a mental health professional immediately. "
            "You matter and help is available right now."
    }

    # -----------------------------------------------------
    # VOICE CONCLUSIONS
    # -----------------------------------------------------

    voice_conclusions = {

        "Non-Stress":
            "You are doing well. Your text does not show strong signs "
            "of stress. You appear to be managing well emotionally. "
            "Continue maintaining your positive habits and take care "
            "of yourself.",

        "Stress":
            "You are under stress. Your text indicates that you may "
            "be experiencing a high level of stress right now. "
            "Please take a break, practise some relaxation techniques, "
            "and consider talking to someone you trust.",

        "Depression":
            "Your text shows patterns associated with depression. "
            "Please remember that you are not alone. Talk to someone "
            "you trust and consider reaching out to a mental health "
            "professional for support.",

        "Suicidal":
            "Your text contains deeply concerning patterns. "
            "Please do not face this alone. Talk to someone you trust "
            "immediately and seek professional mental health support. "
            "You matter, and help is available."
    }

    # -----------------------------------------------------
    # TIPS
    # -----------------------------------------------------

    tips = {

        "Non-Stress": [
            "🧘 Keep your routine balanced",
            "🌿 Maintain your positive habits",
            "💤 Protect your sleep schedule"
        ],

        "Stress": [
            "🫁 Try box breathing: 4s in, 4s hold, 4s out",
            "📵 Take a 10-min break from screens",
            "🚶 A short walk can reset your mind"
        ],

        "Depression": [
            "💬 Talk to one trusted person today",
            "🌤️ Step outside even for 5 minutes",
            "📓 Write down one thing you felt today"
        ],

        "Suicidal": [
            "📞 Call iCall: 9152987821 right now",
            "🏥 Visit your nearest mental health center",
            "🤝 Tell someone near you how you feel"
        ]
    }

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return jsonify({

        "lines": line_results,

        "pie": pie_data,

        "dominant": dominant,

        "conclusion": conclusions[dominant],

        "voice_conclusion":
            voice_conclusions[dominant],

        "tips":
            tips[dominant],

        "color":
            COLORS[dominant],

        "icon":
            ICONS[dominant],

        "total_lines":
            total,

        "is_crisis":
            dominant == "Suicidal"
    })


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)