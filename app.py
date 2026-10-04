import streamlit as st

from rag import config, pipeline, store

st.set_page_config(page_title="Local RAG", page_icon="📚")
st.title("📚 Local RAG (Ollama)")

config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("Documents")
    chat_model = st.text_input("Chat model", config.CHAT_MODEL)
    uploads = st.file_uploader("Upload", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Index uploaded files", disabled=not uploads):
        for up in uploads:
            path = config.UPLOAD_DIR / up.name
            path.write_bytes(up.getbuffer())
            with st.spinner(f"Indexing {up.name}..."):
                try:
                    n = pipeline.ingest(path)
                    st.success(f"{up.name}: {n} chunks")
                except Exception as e:
                    st.error(f"{up.name}: {e}")
    st.subheader("Indexed")
    for s in store.list_sources():
        st.write(f"• {s}")
    if st.button("Clear index"):
        store.reset()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            with st.expander("Sources"):
                for h in m["hits"]:
                    st.caption(f"{h['source']} p.{h['page']} · score {h['score']:.2f}")
                    st.write(h["text"])

if question := st.chat_input("Ask about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            hits = pipeline.retrieve(question)
            answer = st.write_stream(pipeline.stream_answer(question, hits, chat_model))
            with st.expander("Sources"):
                for h in hits:
                    st.caption(f"{h['source']} p.{h['page']} · score {h['score']:.2f}")
                    st.write(h["text"])
            st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits})
        except Exception as e:
            st.error(f"Error: {e}. Is Ollama running and are the models pulled?")
