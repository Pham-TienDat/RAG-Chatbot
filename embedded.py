from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load raw PDF
DATA_PATH = "data/"


def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    return documents


documents = load_pdf_file(DATA_PATH)

print("Số documents:", len(documents))

for i, doc in enumerate(documents[:5]):
    print(f"\n--- Document {i + 1} ---")

    print("Metadata:", doc.metadata)

    print("Độ dài nội dung:", len(doc.page_content))

    print("Nội dung:", repr(doc.page_content[:300]))


# Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


text_chunks = create_chunks(documents)

print("Số text_chunks:", len(text_chunks))

if not text_chunks:
    raise ValueError("Không tạo được text chunks. Hãy kiểm tra nội dung PDF.")


# Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return embedding_model


embedding_model = get_embedding_model()

# Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
