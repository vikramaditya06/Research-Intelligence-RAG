import os
import requests
import streamlit as st

st.set_page_config(page_title="Research Intelligence RAG", layout="wide")
API = st.sidebar.text_input("API URL", os.getenv("API_URL", "http://localhost:8000"))
st.title("Research Intelligence RAG")
st.caption("Multi-document research assistant with hybrid retrieval, RRF, reranking, parent-context expansion, compression, and citation auditing.")

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("PDF", type=["pdf"])
    if uploaded and st.button("Index PDF"):
        try:
            r = requests.post(f"{API}/documents/upload", files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}, timeout=300)
            if r.ok:
                st.success(r.json())
            else:
                st.error(r.text)
        except requests.RequestException as exc:
            st.error(f"API unavailable: {exc}")

try:
    docs_response = requests.get(f"{API}/documents", timeout=10)
    docs = docs_response.json() if docs_response.ok else []
except requests.RequestException:
    docs = []

st.subheader("Documents")
selected = st.multiselect("Search selected documents (leave empty to search all)", docs, format_func=lambda x: f"{x['filename']}" + (f" ({x['year']})" if x.get('year') else ""))
if docs:
    for d in docs:
        cols = st.columns([6, 1])
        cols[0].caption(f"{d['filename']} — {d.get('title') or 'Untitled'}")
        if cols[1].button("Delete", key=f"del-{d['id']}"):
            r = requests.delete(f"{API}/documents/{d['id']}", timeout=30)
            if r.ok:
                st.rerun()
            st.error(r.text)

question = st.text_area("Research question", placeholder="Compare the approaches and limitations across the uploaded papers.")
hyde = st.checkbox("Enable HyDE", value=False, help="Generates a hypothetical passage for retrieval only; it is never treated as evidence.")
col1, col2 = st.columns(2)
year_from = col1.number_input("Published from", min_value=1900, max_value=2100, value=1900)
year_to = col2.number_input("Published to", min_value=1900, max_value=2100, value=2100)

if st.button("Research") and question.strip():
    ids = [d["id"] for d in selected]
    with st.spinner("Retrieving, reranking, and auditing evidence..."):
        try:
            r = requests.post(f"{API}/query", json={"question": question, "document_ids": ids, "top_k": 8, "use_hyde": hyde, "year_from": int(year_from) if year_from > 1900 else None, "year_to": int(year_to) if year_to < 2100 else None}, timeout=300)
        except requests.RequestException as exc:
            st.error(f"API unavailable: {exc}")
            st.stop()
    if r.ok:
        result = r.json()
        st.markdown("### Answer")
        st.write(result["answer"])
        verification = result["citation_verification"]
        st.markdown("### Citation audit")
        st.write({k: verification.get(k) for k in ["valid", "citation_completeness", "support_rate", "cited", "invalid", "evidence_span_failures"] if k in verification})
        if verification.get("invalid"):
            st.warning(f"Invalid citation IDs: {verification['invalid']}")
        st.markdown("### Retrieved evidence")
        for s in result["sources"]:
            page_start = s.get("evidence_page_start") or s.get("retrieval_page_start") or s.get("page_start") or s.get("parent_page_start", "?")
            page_end = s.get("evidence_page_end") or s.get("retrieval_page_end") or s.get("page_end") or s.get("parent_page_end", page_start)
            page = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
            with st.expander(f"[{s.get('citation_id')}] page {page} — {s.get('section') or s.get('parent_section') or 'Unknown'}"):
                st.write(s.get("evidence", s["text"]))
                if s.get("ranking_source") == "reranker":
                    rank_info = f"rerank_score={float(s.get('rerank_score', 0.0)):.4f}"
                else:
                    rank_info = f"rrf_score={float(s.get('rrf_score', 0.0)):.6f} rrf_rank={s.get('rrf_rank', '?')}"
                st.caption(f"document_id={s.get('document_id', 'unknown')} ranking_source={s.get('ranking_source', 'unknown')} {rank_info}")
        with st.expander("Query transformations"):
            st.write(result["queries"])
        with st.expander("Pipeline timings"):
            st.json(result.get("timings_ms", {}))
    else:
        st.error(r.text)
