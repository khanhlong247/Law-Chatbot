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

    # --- Tải tài liệu ---
    documents = []
    def load_docs(glob_pattern, loader_cls, label):
        print(f"Đang tải file {label} từ '{DATA_PATH}'...")
        loader = DirectoryLoader(
            DATA_PATH,
            glob=glob_pattern,
            loader_cls=loader_cls,
            use_multithreading=True,
            show_progress=True
        )
        try:
            return loader.load()
        except Exception as e:
            print(f"Lỗi khi tải {label}: {e}")
            return []

    documents.extend(load_docs("**/*.pdf", PyPDFLoader, "PDF"))
    documents.extend(load_docs("**/*.docx", Docx2txtLoader, "DOCX"))
    documents.extend(load_docs("**/*.doc", Docx2txtLoader, "DOC"))

    if not documents:
        print(f"Không tìm thấy tài liệu nào trong thư mục '{DATA_PATH}'.")
        print("Vui lòng kiểm tra lại đường dẫn và các file.")
        return

    print(f"Đã tải thành công {len(documents)} tài liệu.")

    print("Đang chia tài liệu thành các mảnh theo 'Điều'...")
    
    logical_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\nĐiều ", "\nĐiều ", "Điều "],
        keep_separator=True
    )
    
    chunks_with_preamble = logical_splitter.split_documents(documents)
    
    final_chunks = []
    for chunk in chunks_with_preamble:
        content = chunk.page_content.lstrip()
        if content.startswith("Điều "):
            chunk.page_content = content
            final_chunks.append(chunk)

    if not final_chunks:
         print("LỖI: Không tìm thấy 'Điều ' nào trong văn bản.")
         print("Vui lòng kiểm tra lại file data hoặc cấu hình 'separators'.")
         return

    print(f"Đã chia thành {len(final_chunks)} mảnh logic.")

    print(f"Đang tải mô hình embedding '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    print(f"Đang ghi dữ liệu vào ChromaDB (collection: {COLLECTION_NAME})...")

    try:
        vectorstore = Chroma.from_documents(
            documents=final_chunks,            
            embedding=embedding_model,          
            collection_name=COLLECTION_NAME,    
            persist_directory=PERSIST_PATH    
        )
    except Exception as e:
        print(f"Lỗi khi ghi vào ChromaDB: {e}")
        return

    print("=" * 60)
    print(f"Hoàn tất! Dữ liệu đã được lưu vào '{PERSIST_PATH}'")
    print("=" * 60)

if __name__ == "__main__":
    main()