import os
import json
import datetime

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
client = Groq()

FEEDBACK_LOG = os.path.join(os.path.dirname(__file__), "feedback_log.json")

PROMPTS = {
    "answer_question": {
        "label": "Answer Questions",
        "styles": {
            "concise": "Answer the following question in a single short, direct sentence:\n\n{input}",
            "detailed": "Answer the following question in detail, including relevant background context:\n\n{input}",
            "facts": "List exactly three interesting facts related to this question or topic:\n\n{input}",
        },
    },
    "summarize_text": {
        "label": "Summarize Text",
        "styles": {
            "brief": "Summarize the following text in 2-3 sentences:\n\n{input}",
            "bullets": "Summarize the following text as a bulleted list of the main points:\n\n{input}",
            "overview": "Give a plain-language one-paragraph overview of the following text:\n\n{input}",
        },
    },
    "creative_content": {
        "label": "Generate Creative Content",
        "styles": {
            "story": "Write a short creative story (about 150 words) based on this idea:\n\n{input}",
            "poem": "Write a short poem (4-8 lines) about the following theme:\n\n{input}",
            "brainstorm": "Generate 3 original creative ideas based on this prompt:\n\n{input}",
        },
    },
}


def _load_feedback():
    if os.path.exists(FEEDBACK_LOG):
        try:
            with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def log_feedback(function, style, user_input, response_text, helpful):
    data = _load_feedback()
    data.append({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "function": function,
        "style": style,
        "input": user_input,
        "response": response_text,
        "helpful": helpful,
    })
    with open(FEEDBACK_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@app.route('/')
def home():
    return render_template('index.html', prompts=PROMPTS)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() if request.is_json else request.form

    user_message = data.get('message')
    function = data.get('function', 'answer_question')
    style = data.get('style', 'concise')

    if not user_message:
        return jsonify({'error': 'Message content is required'}), 400

    if function not in PROMPTS or style not in PROMPTS[function]['styles']:
        return jsonify({'error': 'Invalid function or style selected'}), 400

    template = PROMPTS[function]['styles'][style]
    full_prompt = template.format(input=user_message)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and concise AI assistant."
                },
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0.7,
            max_tokens=1024,
        )

        ai_response = chat_completion.choices[0].message.content
        return jsonify({'response': ai_response, 'prompt_used': full_prompt})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json() if request.is_json else request.form
    log_feedback(
        data.get('function'),
        data.get('style'),
        data.get('input'),
        data.get('response'),
        data.get('helpful') in (True, 'true', 'True', 'yes', '1'),
    )
    return jsonify({'status': 'ok'})


@app.route('/feedback-stats', methods=['GET'])
def feedback_stats():
    data = _load_feedback()
    helpful = sum(1 for d in data if d.get('helpful') is True)
    not_helpful = sum(1 for d in data if d.get('helpful') is False)
    return jsonify({'total': len(data), 'helpful': helpful, 'not_helpful': not_helpful})


if __name__ == '__main__':
    app.run(debug=True)
