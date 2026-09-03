"""
Biomedical RAG Assistant
------------------------
A retrieval-augmented question answering pipeline over a small set of
biomedical documents, with:
  - LSA-based semantic retrieval (TF-IDF + Truncated SVD), which captures
    some meaning beyond exact keyword overlap, entirely offline
  - A blended faithfulness evaluation (word overlap + cosine similarity)
    that checks whether a generated answer is actually supported by the
    retrieved text
  - Experiment logging: every query, retrieval, and evaluation is logged
    to a JSONL file with a timestamp, so runs can be reviewed later

Generation: pluggable — works with the Anthropic API if you set an API
key, and falls back to an extractive (non-generative) answer otherwise
so the retrieval + evaluation logic is still fully runnable and testable
without any API key.
"""

import os
import re
import json
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# 1. Document store
# ---------------------------------------------------------------------
# Multiple passages per topic, so retrieval actually has to discriminate
# between related documents rather than matching one obvious keyword.

DOCUMENTS = [
    {"id": "pneumonia_1", "topic": "pneumonia", "text": (
        "Pneumonia is an infection that inflames the air sacs in one or "
        "both lungs. The air sacs may fill with fluid or pus, causing "
        "cough with phlegm, fever, chills, and difficulty breathing. "
        "A variety of organisms, including bacteria, viruses, and fungi, "
        "can cause pneumonia."
    )},
    {"id": "pneumonia_2", "topic": "pneumonia", "text": (
        "Diagnosis of pneumonia typically involves a chest X-ray, blood "
        "tests, and listening to the lungs with a stethoscope for "
        "abnormal crackling sounds. Treatment depends on the cause and "
        "may include antibiotics for bacterial pneumonia or supportive "
        "care for viral pneumonia."
    )},
    {"id": "afib_1", "topic": "atrial fibrillation", "text": (
        "Atrial fibrillation (AFib) is the most common type of treated "
        "heart arrhythmia. In AFib, the heart's upper chambers beat "
        "irregularly and out of coordination with the lower chambers. "
        "AFib risk increases with age and is more common in adults "
        "over 65."
    )},
    {"id": "afib_2", "topic": "atrial fibrillation", "text": (
        "People with atrial fibrillation may experience heart "
        "palpitations, fatigue, shortness of breath, or dizziness, "
        "though some cases are asymptomatic and only found during a "
        "routine ECG. AFib increases the risk of stroke because blood "
        "can pool and clot in the heart's upper chambers."
    )},
    {"id": "heart_disease_1", "topic": "heart disease", "text": (
        "Coronary heart disease risk factors include high blood "
        "pressure, high LDL cholesterol, smoking, diabetes, obesity, "
        "physical inactivity, and an unhealthy diet. Many of these "
        "risk factors can be modified through lifestyle changes."
    )},
    {"id": "heart_disease_2", "topic": "heart disease", "text": (
        "Lifestyle changes that reduce heart disease risk include "
        "regular physical activity, a diet low in saturated fat and "
        "sodium, maintaining a healthy weight, not smoking, and "
        "managing conditions such as high blood pressure and diabetes."
    )},
    {"id": "ecg_1", "topic": "ecg", "text": (
        "An electrocardiogram (ECG or EKG) records the electrical "
        "signals of the heart. It is a common, painless test used to "
        "quickly detect heart problems and monitor heart health, "
        "including detecting arrhythmias such as atrial fibrillation."
    )},
    {"id": "ecg_2", "topic": "ecg", "text": (
        "An ECG works by placing electrodes on the skin of the chest, "
        "arms, and legs to measure the timing and strength of electrical "
        "signals as they travel through the heart during each heartbeat, "
        "producing a characteristic waveform for each cycle."
    )},
    {"id": "breast_cancer_1", "topic": "breast cancer", "text": (
        "Breast cancer is a disease in which cells in the breast grow "
        "out of control. Early detection through regular screening, "
        "such as mammography, significantly improves treatment "
        "outcomes and survival rates."
    )},
    {"id": "breast_cancer_2", "topic": "breast cancer", "text": (
        "Risk factors for breast cancer include age, family history, "
        "certain inherited gene mutations such as BRCA1 and BRCA2, "
        "early menstruation, late menopause, and hormone replacement "
        "therapy, though many people diagnosed have no known risk "
        "factors."
    )},
]

# ---------------------------------------------------------------------
# 2. Retrieval: TF-IDF -> LSA (Truncated SVD) -> cosine similarity
# ---------------------------------------------------------------------
# LSA projects TF-IDF vectors into a lower-dimensional "semantic" space,
# so documents that share related vocabulary (not just identical words)
# end up closer together. This is a real, classic embedding technique,
# unlike raw TF-IDF cosine similarity, and runs fully offline.

class Retriever:
    def __init__(self, documents, n_components=6):
        self.documents = documents
        self.texts = [d["text"] for d in documents]
        n_components = min(n_components, len(documents) - 1)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(self.texts)

        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.doc_embeddings = self.svd.fit_transform(tfidf_matrix)

    def retrieve(self, query, top_k=1):
        query_tfidf = self.vectorizer.transform([query])
        query_embedding = self.svd.transform(query_tfidf)
        scores = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        ranked_idx = scores.argsort()[::-1][:top_k]
        results = []
        for idx in ranked_idx:
            results.append({
                "id": self.documents[idx]["id"],
                "topic": self.documents[idx]["topic"],
                "text": self.documents[idx]["text"],
                "score": float(scores[idx]),
            })
        return results


# ---------------------------------------------------------------------
# 3. Generation: LLM-based if an API key is available, else extractive
# ---------------------------------------------------------------------

def generate_answer_with_llm(question, context):
    """
    Calls the Anthropic API to generate an answer grounded in the
    retrieved context. Requires ANTHROPIC_API_KEY to be set as an
    environment variable. Returns None if no key is available, so the
    caller can fall back to the extractive method.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        print("[info] anthropic package not installed; run "
              "'pip install anthropic' to enable LLM generation.")
        return None

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "Answer the question using ONLY the information in the context "
        "below. If the context does not contain the answer, say so "
        "explicitly rather than guessing.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def generate_answer_extractive(question, context):
    """
    Fallback generation with no LLM: just returns the retrieved passage.
    Not a real generative answer, but keeps the pipeline fully runnable
    offline so retrieval + evaluation can be tested without an API key.
    """
    return context


# ---------------------------------------------------------------------
# 4. Evaluation: is the answer actually supported by the source?
# ---------------------------------------------------------------------
# Blends two signals:
#   - word overlap: fraction of the answer's content words that also
#     appear in the source (catches answers that introduce unrelated
#     vocabulary entirely)
#   - TF-IDF cosine similarity between answer and source (catches
#     answers that are topically related but phrased very differently,
#     which raw word overlap can miss)
# This is still a heuristic, not a real entailment/faithfulness model,
# but combining two weak signals is more robust than either alone.

def word_overlap_score(answer, source):
    def tokenize(text):
        words = re.findall(r"[a-zA-Z]+", text.lower())
        stopwords = {
            "the", "a", "an", "is", "are", "of", "in", "and", "to",
            "it", "that", "this", "with", "as", "or", "be", "can",
            "on", "for", "by", "at", "which", "these",
        }
        return {w for w in words if w not in stopwords and len(w) > 2}

    answer_words = tokenize(answer)
    source_words = tokenize(source)
    if not answer_words:
        return 0.0
    overlap = answer_words & source_words
    return len(overlap) / len(answer_words)


def cosine_faithfulness_score(answer, source):
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform([answer, source])
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def evaluate_faithfulness(answer, source, threshold=0.5):
    overlap = word_overlap_score(answer, source)
    cosine_sim = cosine_faithfulness_score(answer, source)
    blended = (overlap + cosine_sim) / 2
    verdict = "likely grounded" if blended >= threshold else "possible unsupported content"
    return {
        "word_overlap": round(overlap, 2),
        "cosine_similarity": round(cosine_sim, 2),
        "blended_score": round(blended, 2),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------
# 5. Experiment logging
# ---------------------------------------------------------------------
# Every query is logged as one JSON line, so a full run can be reviewed
# or analyzed later without re-running anything. This is a small,
# honest version of the "robust logging and experiment tracking
# infrastructure" a real evaluation project needs.

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_experiment_log.jsonl")


def log_experiment(record, log_path=LOG_PATH):
    record = dict(record)
    record["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------
# 6. Full pipeline
# ---------------------------------------------------------------------

class BiomedicalRAGAssistant:
    def __init__(self, documents, log_path=LOG_PATH):
        self.retriever = Retriever(documents)
        self.log_path = log_path

    def answer(self, question, top_k=1, verbose=True):
        retrieved = self.retriever.retrieve(question, top_k=top_k)

        if not retrieved or retrieved[0]["score"] <= 0.0:
            if verbose:
                print(f"\nQuestion: {question}")
                print("No relevant document found (retrieval score was 0).")
                print("Answer: I don't have information on that topic yet.")
            log_experiment({
                "question": question,
                "retrieved_id": None,
                "retrieval_score": 0.0,
                "answer": "I don't have information on that topic yet.",
                "generation_mode": "none",
                "evaluation": None,
            }, self.log_path)
            return {"answer": "I don't have information on that topic yet.",
                    "source": None, "evaluation": None}

        top_doc = retrieved[0]
        context = top_doc["text"]

        llm_answer = generate_answer_with_llm(question, context)
        if llm_answer is not None:
            answer = llm_answer
            mode = "llm"
        else:
            answer = generate_answer_extractive(question, context)
            mode = "extractive (no API key set)"

        evaluation = evaluate_faithfulness(answer, context)

        if verbose:
            print(f"\nQuestion: {question}")
            print(f"Retrieved from: {top_doc['id']} (topic={top_doc['topic']}, score={top_doc['score']:.2f})")
            print(f"Generation mode: {mode}")
            print(f"Answer: {answer}")
            print(f"Faithfulness check: {evaluation}")

        log_experiment({
            "question": question,
            "retrieved_id": top_doc["id"],
            "retrieval_score": top_doc["score"],
            "answer": answer,
            "generation_mode": mode,
            "evaluation": evaluation,
        }, self.log_path)

        return {
            "answer": answer,
            "source": top_doc["id"],
            "retrieval_score": top_doc["score"],
            "evaluation": evaluation,
        }


# ---------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    assistant = BiomedicalRAGAssistant(DOCUMENTS)
    print("Biomedical RAG Assistant (type 'quit' to exit)")
    print("Note: set ANTHROPIC_API_KEY as an environment variable to enable "
          "real LLM-generated answers; otherwise the assistant returns the "
          "retrieved passage directly.")
    print(f"All queries are logged to: {LOG_PATH}\n")

    while True:
        question = input("Ask a question: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        assistant.answer(question)
