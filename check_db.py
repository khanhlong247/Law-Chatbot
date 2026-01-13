from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
PERSIST_PATH = "./chroma_db"
COLLECTION_NAME = "law_docs"

print("--- ĐANG KIỂM TRA DỮ LIỆU ---")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
vectorstore = Chroma(collection_name=COLLECTION_NAME, persist_directory=PERSIST_PATH, embedding_function=embedding_model)

# Test tìm kiếm thô
targets = ["Điều 12", "Điều 13", "Điều 134"]

for t in targets:
    print(f"\n>> Đang tìm: '{t}' trong BLHS.docx...")
    # Tìm 5 kết quả sát nhất
    docs = vectorstore.similarity_search(f"nội dung {t} bộ luật hình sự", k=5, filter={"source_name": "BLHS.docx"})
    
    found = False
    for d in docs:
        content = d.page_content
        # Kiểm tra xem có chuỗi "Điều X" trong text không
        if f"{t}" in content:
            print(f"   [CÓ THẤY] ID: {d.metadata.get('source_name')} | Preview: {content[:100]}...")
            found = True
            break
    
    if not found:
        print(f"   [KHÔNG TÌM THẤY] Có vẻ như '{t}' không tồn tại hoặc bị cắt lỗi.")
        # In thử cái tìm được xem nó ra cái gì
        if docs:
            print(f"   (Kết quả vector trả về thay thế: {docs[0].page_content[:100]}...)")