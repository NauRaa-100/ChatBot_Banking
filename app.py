from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import plotly.express as px
import base64
from io import BytesIO

app = Flask(__name__)

# -----------------------------
# Load Model
# -----------------------------
model_name = "NauRaa/banking77-intent-model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# -----------------------------
# Label Mapping
# -----------------------------
label_map = {
    "LABEL_0": "accept_credit_card", "LABEL_1": "activate_card", "LABEL_2": "add_account",
    "LABEL_3": "balance", "LABEL_4": "bank_charge", "LABEL_5": "block_card",
    "LABEL_6": "borrow_money", "LABEL_7": "buy_travel_ticket", "LABEL_8": "cancel",
    "LABEL_9": "cash_withdrawal", "LABEL_10": "cheque", "LABEL_11": "close_account",
    "LABEL_12": "confirm_receipt", "LABEL_13": "contact", "LABEL_14": "credit",
    "LABEL_15": "credit_limit", "LABEL_16": "currency_exchange", "LABEL_17": "debit_card",
    "LABEL_18": "deposit", "LABEL_19": "direct_debit", "LABEL_20": "dispute",
    "LABEL_21": "due_amount", "LABEL_22": "emv_features", "LABEL_23": "exchange_rate",
    "LABEL_24": "fingerprint", "LABEL_25": "foreign_transaction", "LABEL_26": "funds_transfer",
    "LABEL_27": "income", "LABEL_28": "insurance", "LABEL_29": "internet_banking",
    "LABEL_30": "invoice", "LABEL_31": "loan", "LABEL_32": "lost_card",
    "LABEL_33": "mobile_banking", "LABEL_34": "mortgage", "LABEL_35": "net_banking",
    "LABEL_36": "order_checkbook", "LABEL_37": "pay_credit_card", "LABEL_38": "payroll",
    "LABEL_39": "pin", "LABEL_40": "refund", "LABEL_41": "repay_loan",
    "LABEL_42": "reset_password", "LABEL_43": "reward", "LABEL_44": "salary",
    "LABEL_45": "schedule_payment", "LABEL_46": "security", "LABEL_47": "statement",
    "LABEL_48": "stop_payment", "LABEL_49": "subscription", "LABEL_50": "tax",
    "LABEL_51": "terms_conditions", "LABEL_52": "transaction_history", "LABEL_53": "transfer",
    "LABEL_54": "update_account_info", "LABEL_55": "update_contact_info",
    "LABEL_56": "update_password", "LABEL_57": "verify_identity",
    "LABEL_58": "virtual_card", "LABEL_59": "withdrawal_limit", "LABEL_60": "wire_transfer",
    "LABEL_61": "account_blocked", "LABEL_62": "account_closure",
    "LABEL_63": "account_opening", "LABEL_64": "account_statement",
    "LABEL_65": "atm_issue", "LABEL_66": "auto_pay", "LABEL_67": "beneficiary",
    "LABEL_68": "blocked_card", "LABEL_69": "card_delivery",
    "LABEL_70": "card_issue", "LABEL_71": "card_limit", "LABEL_72": "card_pin",
    "LABEL_73": "card_replacement", "LABEL_74": "card_statement",
    "LABEL_75": "change_address", "LABEL_76": "check_status"
}

conversation_history = []

# -----------------------------
# Helper: Convert Plotly → Base64 Image
# -----------------------------
def fig_to_base64(fig):
    buffer = BytesIO()
    fig.write_image(buffer, format='png')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global conversation_history

    text = request.json["text"]

    # Model Prediction
    all_scores = classifier(text)[0]
    all_scores.sort(key=lambda x: x['score'], reverse=True)

    top_label = all_scores[0]['label']
    top_score = all_scores[0]['score']
    top_intent = label_map.get(top_label, top_label)

    # Add to conversation history
    conversation_history.append({
        "input": text,
        "intent": top_intent,
        "score": round(top_score, 2)
    })

    # Build DataFrame
    df = pd.DataFrame({
        "Intent": [label_map.get(s['label'], s['label']) for s in all_scores],
        "Score": [s['score'] for s in all_scores]
    })

    # Plot chart
    fig = px.bar(df, x="Intent", y="Score", title="Intent Probabilities")
    fig.update_layout(xaxis_tickangle=-45)

    # Convert img
    img_base64 = fig_to_base64(fig)

    return jsonify({
        "top_intent": top_intent,
        "top_score": round(top_score, 2),
        "chart_image": img_base64,
        "history": conversation_history
    })


if __name__ == "__main__":
    app.run(debug=True)
