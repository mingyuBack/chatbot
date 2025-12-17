"""
학교 학사일정/공지/학과 안내 RAG 챗봇
company_docs.csv → school_schedule.csv로 변경
"""

import csv
import requests
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model_name = "gemma3:4b"
api_url = "http://localhost:11434/api/chat"
csv_path = "school_schedule.csv"  # 학교 CSV 파일

app = Flask(__name__)

chat_history = []
last_doc_title = None

def normalize_korean_text(text: str) -> str:
    """한글 공백 제거 전처리"""
    if not text:
        return ""
    return text.strip().replace(" ", "")

documents = []
corpus = []
vectorizer = None
doc_vectors = None

def load_documents_from_csv(path: str):
    """school_schedule.csv에서 학사 문서 로드"""
    docs = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            intent = (row.get("intent") or "").strip()
            if not text or not intent:
                continue
            docs.append({"title": text, "content": intent})

    if not docs:
        raise ValueError("CSV에서 유효한 문서를 하나도 읽지 못했다.")
    return docs

def build_tfidf_index(docs):
    """TF-IDF 벡터 인덱스 구축"""
    local_corpus = [(doc["title"] + "\n" + doc["content"]).strip() for doc in docs]
    local_vectorizer = TfidfVectorizer(
        preprocessor=normalize_korean_text,
        analyzer="char",
        ngram_range=(2, 4),
    )
    local_doc_vectors = local_vectorizer.fit_transform(local_corpus)
    return local_corpus, local_vectorizer, local_doc_vectors

# 초기화
try:
    documents = load_documents_from_csv(csv_path)
    corpus, vectorizer, doc_vectors = build_tfidf_index(documents)
    print(f"[정보] 학교 문서 {len(documents)}개 로드 완료")
except Exception as e:
    print(f"[경고] 문서 로드 실패: {e}")
    documents = []
    corpus = []
    vectorizer = None
    doc_vectors = None

def retrieve_top_docs(query: str, top_k: int = 3):
    """질문과 유사한 상위 문서 검색"""
    if not vectorizer or doc_vectors is None:
        return []
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_vectors)[0]
    ranked_indices = np.argsort(-sims)
    return [(documents[idx], float(sims[idx])) for idx in ranked_indices[:top_k]]

def format_context(docs_with_scores, min_score: float = 0.0) -> str:
    """검색 문서를 프롬프트 형식으로 변환"""
    if not docs_with_scores:
        return "관련 학사 정보를 찾지 못했습니다."
    lines = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        if score < min_score:
            continue
        title = doc["title"] or f"항목 {i}"
        content = doc["content"] or ""
        lines.extend([f"[학사 {i}] 질문: {title}", f"내용: {content}", ""])
    return "\n".join(lines) if lines else "관련 학사 정보를 찾지 못했습니다."

def build_rag_prompt(user_message: str, retrieval_query: str | None = None) -> str:
    """학교 RAG 프롬프트 생성"""
    global last_doc_title
    query_for_search = retrieval_query or user_message
    top_docs = retrieve_top_docs(query_for_search, top_k=5)

    if top_docs:
        last_doc_title = top_docs[0][0].get("title") or last_doc_title

    context_text = format_context(top_docs, min_score=0.05)
    topic_line = f"현재 주제: '{last_doc_title}'\n\n" if last_doc_title else ""

    prompt = (
        "당신은 학교 학사일정, 공지사항, 학과 안내를 제공하는 챗봇입니다.\n"
        "school_schedule.csv의 text(질문),intent(답변) 정보만 사용합니다.\n\n"
        f"{topic_line}"
        "규칙:\n"
        "1) CSV 학사 정보만 한국어로 답변\n"
        "2) 없는 정보는 '학사 정보에 없습니다.'\n"
        "3) 날짜/시간은 정확히 전달\n"
        "4) 학과 정보는 intent 내용 그대로 설명\n\n"
        f"# 검색된 학사 정보\n{context_text}\n\n"
        f"# 학생 질문\n{user_message}\n\n"
        "# 답변\n"
    )
    return prompt

def ask_ollama(prompt: str) -> str:
    """Ollama gemma3:4b 호출"""
    messages = [{
        "role": "system",
        "content": "학교 학사일정/공지/학과 안내 챗봇. CSV 정보 외 질문은 '학사 정보에 없습니다.'로 답변"
    }]

    if chat_history:
        messages.extend(chat_history[-6:])

    messages.append({"role": "user", "content": prompt})

    payload = {"model": model_name, "messages": messages, "stream": False}
    resp = requests.post(api_url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()

@app.route("/", methods=["GET"])
def index():
    return render_template("school_chatbot.html")

@app.route("/chat", methods=["POST"])
def chat():
    global last_doc_title
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "질문을 입력해주세요."}), 400

    if not documents:
        return jsonify({"error": "학사 정보가 로드되지 않았습니다."}), 500

    try:
        retrieval_query = f"{last_doc_title} {user_message}" if last_doc_title else user_message
        rag_prompt = build_rag_prompt(user_message, retrieval_query)
        answer = ask_ollama(rag_prompt)

        chat_history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer}
        ])
        if len(chat_history) > 10:
            chat_history[:] = chat_history[-10:]

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)  # 5001 포트 사용
