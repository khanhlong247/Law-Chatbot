# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma

os.environ["ANONYMIZED_TELEMETRY"] = "False"

DATA_PATH = "data/"
PERSIST_PATH = "chroma_db"
COLLECTION_NAME = "law_docs"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

def main():
    print("Bắt đầu quá trình nạp dữ liệu...")

    documents = []
    def load_docs(glob_pattern, loader_cls, label):
        print(f"Đang tải file {label} từ '{DATA_PATH}'...")
        loader = DirectoryLoader(DATA_PATH, glob=glob_pattern, loader_cls=loader_cls, show_progress=True)
        try:
            return loader.load()
        except Exception as e:
            print(f"Lỗi tải {label}: {e}")
            return []

    documents.extend(load_docs("**/*.pdf", PyPDFLoader, "PDF"))
    documents.extend(load_docs("**/*.docx", Docx2txtLoader, "DOCX"))

    if not documents:
        print("Không có tài liệu.")
        return

    # --- CẢI TIẾN 1: XỬ LÝ SƠ BỘ VÀ GẮN TÊN FILE VÀO CONTENT ---
    # Việc này giúp khi cắt nhỏ, Model vẫn biết đoạn đó thuộc luật nào
    print("Đang tiền xử lý nội dung...")
    for doc in documents:
        # Lấy tên file làm ngữ cảnh (ví dụ: BLHS.docx -> BLHS)
        source_file = os.path.basename(doc.metadata.get('source', ''))
        doc.metadata['source_name'] = source_file # Lưu gọn để dùng sau này
        
        # Thêm tên luật vào đầu nội dung văn bản gốc để ngữ cảnh mạnh hơn
        doc.page_content = f"Tài liệu: {source_file}\n{doc.page_content}"

    print("Đang chia tài liệu...")
    logical_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, # Giảm size xuống để Qwen 1.8B dễ tiêu hóa
        chunk_overlap=150,
        separators=["\n\nĐiều ", "\nĐiều ", "Điều "],
        keep_separator=True
    )
    
    chunks = logical_splitter.split_documents(documents)
    
    final_chunks = []
    for chunk in chunks:
        content = chunk.page_content
        source_name = chunk.metadata.get('source_name', 'Tài liệu')
        
        # --- CẢI TIẾN 2: THÊM PREFIX CHO MODEL E5 ---
        # Model E5 bắt buộc nội dung lưu vào DB phải có "passage: "
        # Đồng thời nhắc lại tên file trong từng chunk nhỏ
        new_content = f"passage: {content}" 
        
        # Cập nhật lại chunk
        chunk.page_content = new_content
        final_chunks.append(chunk)

    print(f"Số lượng chunks sau khi chia: {len(final_chunks)}")

    print(f"Đang tải model embedding '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    print(f"Ghi vào ChromaDB...")
    # Lưu ý: Xóa DB cũ nếu muốn làm mới hoàn toàn
    if os.path.exists(PERSIST_PATH):
        import shutil
        # shutil.rmtree(PERSIST_PATH) # Bỏ comment nếu muốn xóa DB cũ đi làm lại từ đầu
        pass

    vectorstore = Chroma.from_documents(
        documents=final_chunks,            
        embedding=embedding_model,          
        collection_name=COLLECTION_NAME,    
        persist_directory=PERSIST_PATH    
    )

    print("Hoàn tất!")

if __name__ == "__main__":
    main()