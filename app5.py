# importing dependencies
import os
import sys
import io
import re
import fitz  # PyMuPDF
from collections import Counter
from typing import List, Tuple, Dict, Any

# Remove incompatible homebrew site-packages (py3.13) from path before heavy imports
BAD_PREFIX = "/opt/homebrew/lib/python3.13/site-packages"
if "PYTHONPATH" in os.environ and BAD_PREFIX in os.environ["PYTHONPATH"]:
    os.environ.pop("PYTHONPATH", None)
sys.path = [p for p in sys.path if BAD_PREFIX not in p]

from dotenv import load_dotenv
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi

from htmlTemplates import css, bot_template, user_template

# ⚠️ Hard-coded demo key (per request). For production, load from env or st.secrets.
GROQ_API_KEY = "your_groq_api_key_here"

# creating custom templates
custom_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question in the same language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

qa_template = """You are a grounded assistant answering questions only from the provided document excerpts.

Rules:
- Use only the provided sources.
- You may summarize, paraphrase, and make careful high-level inferences that are directly supported by the sources.
- Do not invent papers, authors, results, numbers, or claims not present in the sources.
- Respect document boundaries using metadata like title, source, and page.
- If the user asks about a specific paper, focus on that paper only if the sources support it.
- If the user asks about multiple papers, organize the answer by document.
- If the information is partially available, answer with what is supported and clearly say what is not visible in the excerpts.
- Only say "Not present in provided documents" when there is truly no relevant information in the sources.
- Prefer concise, clear answers over vague or generic explanations.

Sources:
{context}

Question: {question}

Answer:
"""

comparison_template = """You are comparing uploaded documents using only the provided excerpts.

Rules:
- Organize the answer by document.
- Then give key differences clearly.
- Use only the provided excerpts.
- Do not invent facts.
- If a comparison point is not supported by the excerpts, say so.

Sources:
{context}

Question: {question}

Answer:
"""

summary_multi_template = """You are summarizing each uploaded document separately using only the provided excerpts.

Rules:
- Provide 1–2 concise sentences per document.
- Keep documents separate; do not mix their content.
- Do not invent facts not supported by the excerpts.

Sources:
{context}

Question: {question}

Answer:
"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
QA_PROMPT = PromptTemplate.from_template(qa_template)


def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def looks_like_author_line(line: str) -> bool:
    low = line.lower()
    bad_tokens = [
        "abstract", "introduction", "conference", "university", "department",
        "arxiv", "www.", "http", "figure", "table", "keywords"
    ]
    if any(tok in low for tok in bad_tokens):
        return False
    if "@" in line:
        return True
    if "," in line and len(line.split()) <= 30:
        return True
    words = line.split()
    capitalized = sum(1 for w in words if w[:1].isupper())
    return 2 <= len(words) <= 25 and capitalized >= max(2, len(words) // 2)


def extract_title_and_authors(first_page_text: str, fallback_name: str) -> Dict[str, Any]:
    lines = [clean_line(x) for x in first_page_text.splitlines()]
    lines = [x for x in lines if x]

    if not lines:
        return {"title": fallback_name, "authors": []}

    # inspect only early lines before abstract
    early_lines = []
    for line in lines[:25]:
        low = line.lower()
        if "abstract" in low:
            break
        early_lines.append(line)

    if not early_lines:
        return {"title": fallback_name, "authors": []}

    # NEW LOGIC: first strong line near top = title
    title = fallback_name
    title_idx = 0

    for i, line in enumerate(early_lines[:8]):
        low = line.lower()

        # skip garbage lines
        if any(x in low for x in ["abstract", "introduction", "arxiv", "http", "www"]):
            continue

        # skip obvious author lines (comma-heavy)
        if line.count(",") >= 2:
            continue

        # skip email lines
        if "@" in line:
            continue

        # accept first reasonable line as title
        if len(line.split()) >= 3:
            title = line
            title_idx = lines.index(line) if line in lines else 0
            break

    author_lines = []
    for line in lines[title_idx + 1:title_idx + 8]:
        low = line.lower()
        if "abstract" in low:
            break
        if looks_like_author_line(line):
            author_lines.append(line)

    author_text = " ".join(author_lines)
    author_text = re.sub(r"\s*[*†‡]+\s*", " ", author_text)
    author_text = re.sub(r"\S+@\S+", "", author_text)
    author_text = re.sub(r"\s+", " ", author_text).strip()

    parts = re.split(r",| and ", author_text)
    authors = []
    for p in parts:
        p = clean_line(p)
        if not p:
            continue
        if len(p.split()) > 5:
            continue
        if any(ch.isdigit() for ch in p):
            continue
        authors.append(p)

    deduped = []
    seen = set()
    for a in authors:
        key = a.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(a)

    return {"title": title, "authors": deduped}


def extract_pdf_data(pdf_file) -> Dict[str, Any]:
    pdf_name = getattr(pdf_file, "name", "uploaded.pdf")
    base_name = os.path.splitext(pdf_name)[0]

    file_bytes = pdf_file.read()
    pdf_file.seek(0)

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        pages.append({
            "page_num": i + 1,
            "text": text
        })

    full_text = "\n".join(p["text"] for p in pages).strip()
    first_page_text = pages[0]["text"] if pages else ""

    meta = extract_title_and_authors(first_page_text, base_name)
    return {
        "source": pdf_name,
        "base_name": base_name,
        "title": meta["title"] or base_name,
        "authors": meta["authors"],
        "first_page_text": first_page_text,
        "pages": pages,
        "page_count": len(pages),
        "full_text": full_text,
    }


def chunk_all_pdfs(docs) -> Tuple[List[str], List[dict], List[dict]]:
    """
    Returns:
      chunks: list[str]
      metadatas: list[dict]
      doc_index: list[dict]  # structured per-document metadata
    """
    chunks, metadatas, doc_index = [], [], []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    for doc_id, pdf in enumerate(docs):
        try:
            pdf_data = extract_pdf_data(pdf)
        except Exception as exc:
            st.warning(f"Skipping unreadable file: {getattr(pdf, 'name', 'upload')} ({exc})")
            continue

        if not pdf_data["full_text"].strip():
            st.warning(f"No readable text in {pdf_data['source']}; skipping.")
            continue

        doc_record = {
            "doc_id": doc_id,
            "source": pdf_data["source"],
            "filename": pdf_data["source"],
            "title": pdf_data["title"],
            "authors": pdf_data["authors"],
            "page_count": pdf_data["page_count"],
            "first_page_text": pdf_data["first_page_text"],
        }
        doc_index.append(doc_record)

        for page in pdf_data["pages"]:
            page_num = page["page_num"]
            page_text = page["text"]
            if not page_text.strip():
                continue
            if any(x in page_text.lower() for x in [
                "references", "bibliography", "citation", "related work"
            ]):
                continue

            for chunk_id, chunk in enumerate(splitter.split_text(page_text)):
                if not chunk.strip():
                    continue
                chunks.append(chunk)
                metadatas.append({
                    "doc_id": doc_id,
                    "source": pdf_data["source"],
                    "title": pdf_data["title"],
                    "authors": pdf_data["authors"],
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "is_first_page": page_num == 1,
                })

    return chunks, metadatas, doc_index


# vector store using MiniLM + FAISS
def get_vectorstore(chunks, metadatas):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas)


def build_bm25_index(chunks):
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


def bm25_search(question, bm25, chunks, metas, top_k=5):
    if not bm25 or not chunks or not metas:
        return []
    query_tokens = question.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    docs = []
    for i in top_idx:
        docs.append(Document(page_content=chunks[i], metadata=metas[i]))
    return docs


def hybrid_retrieve(question, vectorstore, bm25, chunks, metas, top_k_dense=4, top_k_bm25=4):
    dense_docs = vectorstore.similarity_search(question, k=top_k_dense) if vectorstore else []
    sparse_docs = bm25_search(question, bm25, chunks, metas, top_k=top_k_bm25)

    merged, seen = [], set()
    for doc in dense_docs + sparse_docs:
        meta = doc.metadata or {}
        key = (meta.get("doc_id"), meta.get("page"), meta.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    return merged


# 🟦 conversation chain using Groq LLaMA-3
def get_conversationchain(vectorstore):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.15,
        max_retries=2
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 15, "lambda_mult": 0.7},
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        memory=memory,
        return_source_documents=True,
    )
    return conversation_chain


def classify_query(question: str) -> str:
    q = question.lower()
    metadata_triggers = [
        "title", "titles", "author", "authors",
        "paper name", "paper names",
        "document name", "document names",
        "which pdf", "which pdfs",
        "uploaded files", "uploaded pdfs",
        "list the papers", "list papers",
        "list documents", "what papers",
        "what documents", "how many pdf"
    ]
    comparison_triggers = ["compare", "difference", "different", "contrast", " vs ", " versus "]
    if any(t in q for t in metadata_triggers):
        return "metadata"
    if any(t in q for t in comparison_triggers):
        return "comparison"
    return "semantic"


def answer_from_doc_index(question: str, doc_index: List[dict]) -> str:
    q = question.lower()

    if not doc_index:
        return "No processed PDFs found."

    if "how many" in q and "pdf" in q:
        return f"{len(doc_index)} PDF(s) uploaded."

    if "author" in q and "title" in q:
        lines = []
        for d in doc_index:
            authors = ", ".join(d["authors"]) if d["authors"] else "Authors not clearly extracted"
            lines.append(f"- {d['title']} — {authors}")
        return "\n".join(lines)

    if "author" in q:
        lines = []
        for d in doc_index:
            authors = ", ".join(d["authors"]) if d["authors"] else "Authors not clearly extracted"
            lines.append(f"- {d['title']}: {authors}")
        return "\n".join(lines)

    if "title" in q or "paper" in q or "document" in q or "pdf" in q or "list" in q:
        return "\n".join(f"- {d['title']}" for d in doc_index)

    return "\n".join(f"- {d['title']}" for d in doc_index)


def summarize_all_docs() -> str:
    docs = st.session_state.get("doc_index", [])
    if not docs:
        return "No documents found."
    lines = [f"- {d['title']}" for d in docs]
    return "Summary should be done per document. Available documents:\n" + "\n".join(lines)


def normalize_simple(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def match_document_from_question(question: str, doc_index: List[dict]) -> Dict[str, Any] | None:
    if not doc_index:
        return None
    nq = normalize_simple(question)
    best = None
    for d in doc_index:
        title_norm = normalize_simple(d["title"])
        if title_norm and title_norm in nq:
            best = d
            break
        # partial: first part before colon
        if ":" in d["title"]:
            short = normalize_simple(d["title"].split(":")[0])
            if short and short in nq:
                best = d
                break
    return best


def get_first_page_chunks_for_doc(vectorstore, target_doc_id: int, limit: int = 2):
    docs = []
    if not vectorstore:
        return docs
    store = getattr(vectorstore, "docstore", None)
    if not store or not hasattr(store, "_dict"):
        return docs
    for doc in store._dict.values():
        meta = doc.metadata or {}
        if meta.get("doc_id") == target_doc_id and meta.get("is_first_page"):
            docs.append(doc)
            if len(docs) >= limit:
                break
    return docs


def get_chunks_for_doc(question: str, vectorstore, target_doc_id: int, k: int = 4, first_page_k: int = 2):
    if not vectorstore:
        return []
    retrieved = hybrid_retrieve(
        question,
        vectorstore,
        st.session_state.get("bm25"),
        st.session_state.get("all_chunks", []),
        st.session_state.get("all_metas", []),
        top_k_dense=6,
        top_k_bm25=6
    )
    filtered = []
    seen = set()
    for doc in retrieved:
        meta = doc.metadata or {}
        if meta.get("doc_id") != target_doc_id:
            continue
        key = (meta.get("doc_id"), meta.get("page"), meta.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        filtered.append(doc)
        if len(filtered) >= k:
            break
    firsts = get_first_page_chunks_for_doc(vectorstore, target_doc_id, limit=first_page_k)
    for doc in firsts:
        meta = doc.metadata or {}
        key = (meta.get("doc_id"), meta.get("page"), meta.get("chunk_id"))
        if key not in seen:
            filtered.append(doc)
            seen.add(key)
    return filtered


def get_top_chunks_for_doc(question, vectorstore, doc_id, k=3, first_page_k=1):
    if not vectorstore:
        return []
    retrieved = hybrid_retrieve(
        question,
        vectorstore,
        st.session_state.get("bm25"),
        st.session_state.get("all_chunks", []),
        st.session_state.get("all_metas", []),
        top_k_dense=6,
        top_k_bm25=6
    )
    out = []
    seen = set()
    for doc in retrieved:
        meta = doc.metadata or {}
        if meta.get("doc_id") != doc_id:
            continue
        key = (meta.get("doc_id"), meta.get("page"), meta.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
        if len(out) >= k:
            break
    firsts = get_first_page_chunks_for_doc(vectorstore, doc_id, limit=first_page_k)
    for doc in firsts:
        meta = doc.metadata or {}
        key = (meta.get("doc_id"), meta.get("page"), meta.get("chunk_id"))
        if key not in seen:
            out.append(doc)
            seen.add(key)
    return out


def build_grouped_context(doc_to_chunks: Dict[int, List]) -> str:
    parts = []
    for doc_id, chunks in doc_to_chunks.items():
        if not chunks:
            continue
        title = chunks[0].metadata.get("title") or chunks[0].metadata.get("source", "")
        parts.append(f"Document: {title}")
        for c in chunks:
            page = c.metadata.get("page")
            label = f"(page {page}) " if page else ""
            parts.append(label + c.page_content)
        parts.append("")  # separator
    return "\n".join(parts)


def build_multi_doc_summary(vectorstore, doc_index):
    if not doc_index or not vectorstore:
        return "Not enough information found in the uploaded documents.", []
    doc_to_chunks = {}
    for d in doc_index:
        doc_to_chunks[d["doc_id"]] = get_top_chunks_for_doc("summary of the document", vectorstore, d["doc_id"], k=3, first_page_k=1)
    all_docs = [c for lst in doc_to_chunks.values() for c in lst]
    if not all_docs:
        return "Not enough information found in the uploaded documents.", []
    context = build_grouped_context(doc_to_chunks)
    prompt = PromptTemplate.from_template(summary_multi_template)
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.15,
        max_retries=2
    )
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": "Provide a 1-2 sentence summary for each document."})
    return to_text(answer), all_docs


def display_label(meta):
    title = meta.get("title") or os.path.splitext(meta.get("source", "PDF"))[0]
    page = meta.get("page")
    if page:
        return f"{title} (page {page})"
    return title


def to_text(resp):
    return resp.content if hasattr(resp, "content") else str(resp)


def answer_with_custom_context(question: str, source_docs: List[Any], mode: str = "normal"):
    if not source_docs:
        return "Not enough information found in the uploaded documents.", []
    if mode == "comparison":
        prompt_tpl = comparison_template
    elif mode == "multi_summary":
        prompt_tpl = summary_multi_template
    else:
        prompt_tpl = qa_template
    context_lines = []
    for d in source_docs:
        meta = d.metadata or {}
        label = display_label(meta)
        context_lines.append(f"{label}\n{d.page_content}")
    context = "\n\n".join(context_lines)
    prompt = PromptTemplate.from_template(prompt_tpl)
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.15,
        max_retries=2
    )
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": question})
    answer_text = answer.content if hasattr(answer, "content") else str(answer)
    return answer_text, source_docs


# handling chat UI
def handle_question(question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDFs first.")
        return

    qlow = question.lower()
    if any(phrase in qlow for phrase in [
        "what's the paper about", "whats the paper about", "whats the papers about",
        "what is this paper about", "what are the papers about"
    ]):
        question = "What is the main idea of the uploaded documents?"
        qlow = question.lower()

    # broad summarize detection
    # broad summarize detection
    summarize_hit = ("summarize" in qlow) or ("summary" in qlow)
    target_doc_for_summary = match_document_from_question(
        question,
        st.session_state.get("doc_index", [])
    ) if summarize_hit else None

    if summarize_hit and not target_doc_for_summary:
        answer_text, used_docs = build_multi_doc_summary(
            st.session_state.get("vectorstore"),
            st.session_state.get("doc_index", [])
        )
        src_labels = sorted({display_label(doc.metadata) for doc in used_docs}) if used_docs else []
        src_note = "<br/><sub>Sources: " + ", ".join(src_labels) + "</sub>" if src_labels else ""
        st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
        st.write(
            bot_template.replace("{{MSG}}", answer_text.replace("\n", "<br/>") + src_note),
            unsafe_allow_html=True
        )
        return

    if any(phrase in qlow for phrase in [
        "what are the papers about",
        "what are the documents about",
        "main idea of the uploaded documents",
        "main idea of the papers",
        "what is the main idea of the uploaded documents"
    ]):
        answer_text, used_docs = build_multi_doc_summary(
            st.session_state.get("vectorstore"),
            st.session_state.get("doc_index", [])
        )
        src_labels = sorted({display_label(doc.metadata) for doc in used_docs}) if used_docs else []
        src_note = "<br/><sub>Sources: " + ", ".join(src_labels) + "</sub>" if src_labels else ""
        st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
        st.write(
            bot_template.replace("{{MSG}}", answer_text.replace("\n", "<br/>") + src_note),
            unsafe_allow_html=True
        )
        return

    query_type = classify_query(question)

    # Abstract routing
    if "abstract" in qlow:
        target = match_document_from_question(question, st.session_state.get("doc_index", []))
        if not target:
            st.write(
                user_template.replace("{{MSG}}", question),
                unsafe_allow_html=True
            )
            st.write(
                bot_template.replace("{{MSG}}", "Please mention which uploaded paper you want the abstract from."),
                unsafe_allow_html=True
            )
            return
        rewritten = f"What is the abstract of the paper titled {target['title']}? Answer only from that paper."
        docs = get_chunks_for_doc(rewritten, st.session_state.get("vectorstore"), target["doc_id"], k=4, first_page_k=2)
        answer, used_docs = answer_with_custom_context(rewritten, docs, mode="normal")
        src_labels = sorted({display_label(d.metadata) for d in used_docs}) if used_docs else []
        src_note = "<br/><sub>Sources: " + ", ".join(src_labels) + "</sub>" if src_labels else ""
        st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", answer.replace("\n", "<br/>") + src_note), unsafe_allow_html=True)
        return

    if query_type == "metadata":
        answer = answer_from_doc_index(question, st.session_state.get("doc_index", []))
        st.write(
            user_template.replace("{{MSG}}", question),
            unsafe_allow_html=True
        )
        st.write(
            bot_template.replace("{{MSG}}", answer.replace("\n", "<br/>")),
            unsafe_allow_html=True
        )
        return

    # Comparison routing
    if query_type == "comparison":
        vectorstore = st.session_state.get("vectorstore")
        doc_to_chunks = {}
        for d in st.session_state.get("doc_index", []):
            doc_to_chunks[d["doc_id"]] = get_top_chunks_for_doc(question, vectorstore, d["doc_id"], k=3, first_page_k=1)
        all_docs = [c for lst in doc_to_chunks.values() for c in lst]
        context = build_grouped_context(doc_to_chunks)
        if not all_docs:
            st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", "Not enough information found in the uploaded documents."), unsafe_allow_html=True)
            return
        prompt = PromptTemplate.from_template(comparison_template)
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.15,
            max_retries=2
        )
        chain = prompt | llm
        answer = chain.invoke({"context": context, "question": question})
        answer_text = to_text(answer)
        src_labels = sorted({display_label(doc.metadata) for doc in all_docs})
        src_note = "<br/><sub>Sources: " + ", ".join(src_labels) + "</sub>" if src_labels else ""
        st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", answer_text.replace("\n", "<br/>") + src_note), unsafe_allow_html=True)
        return

    # Document-specific routing
    target_doc = target_doc_for_summary or match_document_from_question(question, st.session_state.get("doc_index", []))
    if target_doc:
        rewritten = f"Summarize the paper titled {target_doc['title']} in 3 bullet points. Focus on its main idea, contribution, and problem it solves." if summarize_hit else f"{question} Answer only using the paper titled {target_doc['title']}."
        docs = get_chunks_for_doc(rewritten, st.session_state.get("vectorstore"), target_doc["doc_id"], k=4, first_page_k=2)
        answer, used_docs = answer_with_custom_context(rewritten, docs, mode="normal")
        src_labels = sorted({display_label(d.metadata) for d in used_docs}) if used_docs else []
        src_note = "<br/><sub>Sources: " + ", ".join(src_labels) + "</sub>" if src_labels else ""
        st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", to_text(answer).replace("\n", "<br/>") + src_note), unsafe_allow_html=True)
        return

    try:
        response = st.session_state.conversation({'question': question})
    except Exception as exc:
        st.error(f"Failed to generate answer: {exc}")
        return

    st.session_state.chat_history = response.get("chat_history", [])
    sources = response.get("source_documents", [])
    first_page_chunks = [doc for doc in sources if doc.metadata.get("is_first_page")]
    sources = list({id(doc): doc for doc in (sources + first_page_chunks)}.values())

    answer_text = response.get("answer", "").strip()
    if len(answer_text) < 5:
        answer_text = "Not enough information found in the uploaded documents."

    src_labels = sorted({display_label(doc.metadata) for doc in sources}) if sources else []
    src_note = "<br/><sub>Sources: " + ", ".join(src_labels) + "</sub>" if src_labels else ""

    st.write(
        user_template.replace("{{MSG}}", question),
        unsafe_allow_html=True
    )
    formatted_answer = answer_text.replace("\n- ", "<br/>- ").replace("\n", "<br/>")
    st.write(
        bot_template.replace("{{MSG}}", formatted_answer + src_note),
        unsafe_allow_html=True
    )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    # Session state init
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "doc_index" not in st.session_state:
        st.session_state.doc_index = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "bm25" not in st.session_state:
        st.session_state.bm25 = None
    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = []
    if "all_metas" not in st.session_state:
        st.session_state.all_metas = []

    st.header("Chat with multiple PDFs 📚")
    question = st.text_input("Ask a question from your documents:")

    if question:
        if st.session_state.conversation is None:
            st.warning("Please upload and process PDFs first.")
        else:
            handle_question(question)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your PDF(s) and click Process",
            accept_multiple_files=True
        )

        if st.button("Clear chat"):
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.doc_index = []
            st.session_state.vectorstore = None
            st.session_state.bm25 = None
            st.session_state.all_chunks = []
            st.session_state.all_metas = []
            st.success("Chat cleared.")

        if st.button("Process"):
            if not docs:
                st.warning("Please upload at least one PDF before processing.")
                st.stop()

            try:
                with st.spinner("Processing..."):
                    chunks, metas, doc_index = chunk_all_pdfs(docs)
                    if not chunks:
                        st.warning("No readable text found in the uploaded PDFs.")
                        st.stop()

                    vectorstore = get_vectorstore(chunks, metas)
                    bm25_index = build_bm25_index(chunks)
                    st.session_state.conversation = get_conversationchain(vectorstore)
                    st.session_state.chat_history = []
                    st.session_state.doc_index = doc_index
                    st.session_state.vectorstore = vectorstore
                    st.session_state.bm25 = bm25_index
                    st.session_state.all_chunks = chunks
                    st.session_state.all_metas = metas
                st.success("Processing complete. Ask a question!")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                st.stop()

    if st.session_state.get("doc_index"):
        st.sidebar.markdown("### Indexed documents")
        for d in st.session_state["doc_index"]:
            authors = ", ".join(d["authors"]) if d["authors"] else "Unknown authors"
            st.sidebar.markdown(f"**{d['title']}**  \n{authors}  \nPages: {d['page_count']}  \nFile: {d.get('filename', d.get('source',''))}")


if __name__ == '__main__':
    main()
