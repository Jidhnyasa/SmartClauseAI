import pdfplumber,docx
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["Termination", "Confidentiality", "Payment", "Governing Law", "Force Majeure"]

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def classify_clauses(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        full_text = extract_text_from_pdf(uploaded_file)
    else:
        full_text = extract_text_from_docx(uploaded_file)

    sentences = [s.strip() for s in full_text.split(".") if len(s.strip()) > 20]
    result = {}
    for sentence in sentences:
        prediction = classifier(sentence, candidate_labels)
        label = prediction['labels'][0]
        if label not in result:
            result[label] = sentence
    return result



