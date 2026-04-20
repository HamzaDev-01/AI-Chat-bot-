from flask import Flask, request, jsonify
import pickle
import json
import re
import random
import os

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "chatbot_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)


intents_path = os.path.join(BASE_DIR, "intents.json")
with open(intents_path, encoding="utf-8") as f:
    responses = json.load(f)


chat_history = []


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text.strip()


def get_response(intent):
    reply = responses.get(intent, responses["fallback"])

    if isinstance(reply, list):
        reply = random.choice(reply)

    if intent == "fees":
        reply += "\n💡 You can pay via challan or online banking."
    elif intent == "admissions":
        reply += "\n📌 Apply early to secure your seat."
    elif intent == "greeting":
        reply += "\n😊 How can I help you today?"
    elif intent == "scholarships":
        reply += "\n🎓 Maintain your CGPA to continue receiving benefits."

    return reply


@app.route('/')
def home():
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(template_path, encoding="utf-8") as f:
        return f.read()


@app.route("/chat", methods=["POST"])
def chat():
    global chat_history

    try:
        user_input = request.json.get("message", "")

        if not user_input.strip():
            return jsonify({
                "response": "Please type a message.",
                "intent": "empty",
                "confidence": 0.0
            })

        user_input_clean = clean(user_input)

        probs = model.predict_proba([user_input_clean])[0]
        intent = model.classes_[probs.argmax()]
        confidence = max(probs)

        if confidence < 0.20:
            reply = random.choice(responses["fallback"])
        else:
            reply = get_response(intent)

        chat_history.append({
            "user": user_input,
            "bot": reply,
            "intent": intent,
            "confidence": round(float(confidence), 2)
        })

        return jsonify({
            "response": str(reply),
            "intent": str(intent),
            "confidence": round(float(confidence), 2)
        })

    except Exception as e:
        return jsonify({
            "response": "Something went wrong. Please try again.",
            "error": str(e)
        }), 500


@app.route("/history", methods=["GET"])
def history():
    return jsonify(chat_history[-20:])  


if __name__ == "__main__":
    app.run(debug=True)