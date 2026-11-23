from flask import Flask, request, render_template
import os
import json
import requests

print("App file is starting...")
# Try to import pypdf (modern) or PyPDF2 (older)
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None


app = Flask(__name__)


def get_analysis_prompt(resume_text):
    return f"""
You are an expert ATS (Applicant Tracking System) resume analyzer.

Analyze the resume below and return your analysis in this EXACT format.

**CANDIDATE SUMMARY**
Write 2-3 sentences about who this candidate is and what they bring.

**KEY SKILLS**
List 6-10 key skills found in the resume, comma separated.

**EXPERIENCE LEVEL**
State one of:
Fresher | Junior | Intermediate | Senior | Expert

Then one sentence explaining why.

**ATS SCORE**
Give a score like:
74/100

Then explain what the score means in 1-2 sentences.

**MISSING SKILLS**
List 4-6 skills that are missing but would strengthen this resume, comma separated.

**IMPROVEMENT SUGGESTIONS**
Give exactly 4 actionable bullet points starting with •

Resume Text:

{resume_text}
"""


def analyze_with_gemini(api_key, resume_text):

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={api_key}"
    )

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": get_analysis_prompt(resume_text)
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2
        }
    }

    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload),
        timeout=90
    )

    response.raise_for_status()

    result = response.json()

    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]

    except (KeyError, IndexError):

        raise ValueError(
            "Invalid response from Gemini API."
        )

print("Routes loaded")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_resume():

    try:

        # PDF library check
        if PdfReader is None:

            return render_template(
                "index.html",
                error="Install pypdf first.\nRun:\npip install pypdf"
            )

        # Check uploaded file
        if "resume" not in request.files:

            return render_template(
                "index.html",
                error="No file uploaded."
            )

        file = request.files["resume"]

        if file.filename == "":

            return render_template(
                "index.html",
                error="Please select a PDF."
            )

        if not file.filename.lower().endswith(".pdf"):

            return render_template(
                "index.html",
                error="Only PDF files are supported."
            )

        # Read PDF
        reader = PdfReader(file)

        resume_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:

                resume_text += text + "\n"

        if not resume_text.strip():

            return render_template(
                "index.html",
                error="Could not extract text. Use a text-based PDF."
            )

        # Limit size
        resume_text = resume_text[:10000]

        # Read Gemini key from environment
        gemini_key = os.environ.get("GEMINI_API_KEY")

        if not gemini_key:

            return render_template(
                "index.html",
                error=(
                    "Server error:\n"
                    "GEMINI_API_KEY is not configured."
                )
            )

        result = analyze_with_gemini(
            gemini_key,
            resume_text
        )

        return render_template(
            "result.html",
            result=result
        )

    except ValueError as e:

        return render_template(
            "index.html",
            error=str(e)
        )

    except requests.exceptions.RequestException as e:

        return render_template(
            "index.html",
            error=f"API error: {str(e)}"
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    print("Starting Flask server...")

    app.run(
        host="0.0.0.0",
        port=5050,
        debug=True,
        use_reloader=False
    )
