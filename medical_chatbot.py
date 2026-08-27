import os

import streamlit as st
from dotenv import load_dotenv

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import Client

DB_FAISS_PATH = "vectorstore/db_faiss"
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=env_path)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

# ---------------------------------------------------------------------------
# CẤU HÌNH TRANG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="\U0001fa7a",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CSS TUỲ CHỈNH (làm đẹp giao diện)
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        /* Font hiện đại */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
        }

        body {
            background: linear-gradient(135deg, #eef4fb 0%, #f7f9fe 45%, #eaf6ee 100%);
            background-attachment: fixed;
        }

        .stApp { background: transparent; }

        /* Che menu tuỳ chọn & footer mặc định của Streamlit.
           KHÔNG ẩn header (stHeader) vì nút thu gọn/mở sidebar nằm ở đó. */
        #MainMenu, footer {
            visibility: hidden; height: 0;
        }

        .block-container {
            /* padding-top đủ lớn để nội dung (badge "Trợ lý y khoá...")
               không bị header Streamlit (chứa nút thu gọn sidebar) che khuất */
            padding-top: 5.5rem;
            padding-bottom: 6rem;
            max-width: 52rem;
        }

        /* ===== Header hero ===== */
        .hero { text-align: center; margin-bottom: 0.4rem; }
        .hero-badge {
            display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.35rem 0.9rem; border-radius: 999px;
            background: rgba(38, 166, 91, 0.12); color: #1b7a43;
            font-weight: 600; font-size: 0.8rem; letter-spacing: 0.3px;
            border: 1px solid rgba(38, 166, 91, 0.25);
        }
        .hero h1 {
            background: linear-gradient(120deg, #0b6e4f, #1d7fc2);
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 800; font-size: 2.5rem; margin: 0.6rem 0 0.2rem;
        }
        .hero p { color: #5b6472; font-size: 1.02rem; margin: 0 auto; max-width: 40rem; }

        /* ===== Message bubbles ===== */
        [data-testid="stChatMessage"] {
            border-radius: 18px; padding: 0.15rem 0.6rem;
            box-shadow: 0 4px 14px rgba(40, 60, 90, 0.06);
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: linear-gradient(135deg, #0e6f4e, #128bc4);
            color: #ffffff; border-bottom-right-radius: 6px;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            /* KHÔNG đặt background/border ở đây: container này bao trùm cả
               avatar + nội dung + expander nguồn => sẽ vẽ hộp chữ nhật.
               Chỉ bo góc nhẹ, để mặc định nền trong suốt. */
            border-bottom-left-radius: 6px;
        }
        [data-testid="stBottom"] > div {
            background: transparent !important;
        }

        /* ===== Chat input ===== */
        [data-testid="stChatInput"] {
            border-radius: 18px !important;
            border: 1px solid rgba(200, 212, 228, 0.9) !important;
            background: #ffffff !important;
            box-shadow: 0 6px 22px rgba(40, 70, 110, 0.10) !important;
            /* Quan trọng: cắt mọi mép của phần tử con theo đúng góc bo
               18px => không còn cạnh vuông/gấp khúc lồi ra ngoài viền. */
            overflow: hidden;
        }
        [data-testid="stChatInput"]:focus-within {
            border: 1px solid #1b8a60 !important;
            box-shadow: 0 6px 22px rgba(38, 166, 91, 0.16) !important;
        }
        /* Dẹp border/shadow lồng nhau của các khối bên trong */
        /* QUAN TRỌNG: div con trực tiếp (mang viền đỏ focus màu primary
           mặc định của Streamlit, bán kính nhỏ) phải bị vô hiệu hoá. */
        [data-testid="stChatInput"] > div {
            border: none !important;
            border-color: transparent !important;
            background: transparent !important;
            border-radius: 18px !important;
        }
        [data-testid="stChatInput"] [data-testid="stChatInputTextArea"] {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        [data-testid="stChatInput"] textarea {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        [data-testid="stChatInput"] button {
            border-radius: 50% !important;
            border: none !important;
        }

        /* ===== Loading (không làm mờ nền) ===== */
        .loading-box {
            display: flex; align-items: center; gap: 0.6rem;
            background: #ffffff; border: 1px solid rgba(210, 222, 236, 0.8);
            border-radius: 14px; padding: 0.7rem 1rem; margin: 0.5rem 0 0.25rem;
            box-shadow: 0 4px 14px rgba(40, 70, 110, 0.08);
            color: #33506e; font-size: 0.92rem; font-weight: 500;
        }
        .loading-box .spinner {
            width: 18px; height: 18px; border-radius: 50%;
            border: 3px solid rgba(30, 130, 90, 0.2);
            border-top-color: #1b8a60;
            animation: spin 0.9s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* ===== Disclaimer ===== */
        .disclaimer {
            display: flex; gap: 0.6rem;
            /* Căn giữa nội dung theo hàng ngang & dọc */
            justify-content: center;
            align-items: center;
            background: #fff8e6;
            border: 1px solid #f0d78a; border-left: 5px solid #e6a700;
            color: #6b5200;
            border-radius: 12px; padding: 0.7rem 0.95rem;
            font-size: 0.86rem; line-height: 1.45;
            margin: 0.75rem auto 0.25rem;
            box-shadow: 0 2px 10px rgba(230, 167, 0, 0.12);
        }
        .disclaimer b { color: #8a6500; }

        /* ===== Sidebar ===== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f4f8fc 0%, #eef4fa 100%);
            border-right: 1px solid rgba(210, 222, 236, 0.8);
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 1.2rem;
        }
        [data-testid="stSidebar"] .stButton button {
            border-radius: 12px; font-weight: 600;
            border: 1px solid rgba(30, 90, 60, 0.35);
            background: #ffffff; color: #0e6f4e;
        }
        [data-testid="stSidebar"] .stButton button:hover {
            border-color: #1b8a60; background: #effaf3;
        }
        [data-testid="stSidebar"] .stExpander {
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid rgba(210, 222, 236, 0.7);
            box-shadow: 0 2px 8px rgba(30, 60, 90, 0.05);
            margin-bottom: 0.5rem;
        }
        .brand { display: flex; align-items: center; gap: 0.7rem; padding: 0.4rem 0 0.8rem; }
        .brand .logo { font-size: 1.7rem; }
        .brand .name {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 800; font-size: 1.05rem; line-height: 1.1;
            color: #0d2436;
        }
        .brand .sub { font-size: 0.72rem; color: #5b7186; font-weight: 500; }

        /* ===== Custom scrollbar ===== */
        ::-webkit-scrollbar { width: 9px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #c3cfdd; border-radius: 6px; }
        ::-webkit-scrollbar-thumb:hover { background: #a8b7c9; }

        /* ===== Sources ===== */
        .source-card {
            background: #eef6f1; border-left: 4px solid #1e9a63;
            border-radius: 10px; padding: 0.6rem 0.9rem; margin: 0.2rem 0;
            font-size: 0.82rem; color: #2c3e50;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# VECTOR STORE (cache bằng resource)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    db = FAISS.load_local(
        DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    return db


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <span class="logo">🩺</span>
                <div>
                    <div class="name">MediChat</div>
                    <div class="sub">Trợ lý y tế · RAG</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        new_chat = st.button(
            "➕ Cuộc trò chuyện mới",
            use_container_width=True,
            help="Xoá toàn bộ lịch sử và bắt đầu lại",
        )
        if new_chat:
            st.session_state.messages = []
            st.rerun()

        st.divider()

        with st.expander("⚙️ Tuỳ chỉnh", expanded=True):
            k = st.slider(
                "Số chặng truy xuất (k)",
                1,
                6,
                3,
                help="Càng lớn càng nhiều ngữ liệu nhưng chậm hơn",
            )
            temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
            max_tokens = st.select_slider(
                "Max tokens", options=[256, 512, 1024, 2048], value=1024
            )

        with st.expander("ℹ️ Giới thiệu"):
            st.markdown(
                """
                **MediChat** trả lời câu hỏi y khoá dựa trên **Bách khoa toàn thư Y học (Gale)**,
                không tạo nội dung ngoài nguồn tài liệu.

                Chạy với **Groq** (`openai/gpt-oss-120b`) + FAISS + `bge-m3`.
                """
            )

        return k, temperature, max_tokens


# ---------------------------------------------------------------------------
# WELCOME / TRẠNG THÁI TRỐNG
# ---------------------------------------------------------------------------
SAMPLE_QUESTIONS = [
    ("🩺", "Triệu chứng bệnh tiểu đường là gì?"),
    ("❤", "Huyết áp cao gây ra biến chứng gì?"),
    ("💊", "Thuốc kháng sinh hoạt động như thế nào?"),
    ("🌡", "Cúm và cảm lạnh khác nhau ra sao?"),
]


def render_welcome():
    st.markdown(
        """
        <div class="hero">
            <span class="hero-badge">✨ Trợ lý y khoa dựa trên tài liệu thật</span>
            <h1>Hỏi đáp Y khoa thông minh</h1>
            <p>Hỏi bất cứ điều gì về sức khoẻ — Câu trả lời được truy xuất từ
            bách khoa toàn thư y học thực tế.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("**📄 Thử với một câu hỏi sau:**")
    cols = st.columns(2)
    for i, (emoji, q) in enumerate(SAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(
                f"{emoji} {q}",
                key=f"sample_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = q
                st.rerun()
    st.markdown(
        """
        <div class="disclaimer">
            <span>⚠️</span>
            <span><b>Chỉ mang tính tham khảo</b> — Không thay thế bác sĩ.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    inject_css()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    k, temperature, max_tokens = render_sidebar()

    # Nếu vừa nhấn câu hỏi mẫu -> tự động gửi
    pending = st.session_state.pop("pending_prompt", None)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Đặt một câu hỏi y khoa…")

    if pending:
        prompt = pending

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            with st.chat_message("user"):
                st.markdown(prompt)
            loading = st.empty()
            loading.markdown(
                '<div class="loading-box"><div class="spinner"></div>'
                "<span>🔍 Đang tra cứu tài liệu y khoa…</span></div>",
                unsafe_allow_html=True,
            )
            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error(
                        "Không thể tải vector store. Hãy chạy `embedded.py` trước."
                    )
                    loading.empty()
                    return

                llm = ChatGroq(
                    model=GROQ_MODEL_NAME,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=GROQ_API_KEY,
                )

                client = Client()
                retrieval_qa_chat_prompt = client.pull_prompt(
                    "langchain-ai/retrieval-qa-chat",
                    dangerously_pull_public_prompt=True,
                )
                combine_docs_chain = create_stuff_documents_chain(
                    llm, retrieval_qa_chat_prompt
                )
                rag_chain = create_retrieval_chain(
                    vectorstore.as_retriever(search_kwargs={"k": k}),
                    combine_docs_chain,
                )

                response = rag_chain.invoke({"input": prompt})
                result = response["answer"]
                source_documents = response["context"]
            finally:
                loading.empty()

            # Hiển thị tin nhắn sau khi có kết quả
            with st.chat_message("assistant"):
                st.markdown(result)
                if source_documents:
                    with st.expander(f"📚 Tham chiếu ({len(source_documents)} nguồn)"):
                        for idx, doc in enumerate(source_documents, 1):
                            snippet = doc.page_content.strip()[:220] + "…"
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <b>✓ Nguồn {idx}</b> — {snippet}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.error(f"⚠️ Đã xảy ra lỗi: {e}")

    if not st.session_state.messages:
        render_welcome()


if __name__ == "__main__":
    main()
