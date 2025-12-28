# app.py
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

# ... (Phần check path giữ nguyên) ...

MODEL_PATH = "./qwen1_5-1_8b-chat-q8_0.gguf" 
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
PERSIST_PATH = "./chroma_db"
COLLECTION_NAME = "law_docs"

# --- CẢI TIẾN 3: CẤU HÌNH LLM CHỐNG LẶP ---
print(f"Đang tải model LLM từ: {MODEL_PATH}...")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=0,
    n_ctx=4096, 
    temperature=0.1,
    top_p=0.9,
    verbose=False,
    stop=["<|im_end|>", "Người dùng:", "Kết thúc câu trả lời"],
    # Di chuyển repetition_penalty vào model_kwargs
    model_kwargs={
        "repetition_penalty": 1.2 
    }
)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_PATH,
    embedding_function=embedding_model
)

# --- CẢI TIẾN 4: PROMPT CHUẨN CHATML CHO QWEN ---
# Format này giúp model phân biệt rõ đâu là context, đâu là câu hỏi
template = """<|im_start|>system
Bạn là trợ lý luật sư AI chuyên nghiệp. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên CHÍNH XÁC các văn bản pháp luật được cung cấp dưới đây.
Nếu thông tin không có trong văn bản, hãy nói "Tôi không tìm thấy thông tin trong tài liệu được cung cấp".
Không được tự bịa ra điều luật.

Văn bản pháp luật tham khảo:
{context}
<|im_end|>
<|im_start|>user
Câu hỏi: {question}
<|im_end|>
<|im_start|>assistant
Trả lời:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# Hàm filter giữ nguyên (giả sử bạn đã code đúng logic)
# ... (giữ nguyên hàm get_source_filter của bạn) ...
def get_source_filter(query):
    query_lower = query.lower()
    target_files = set()

    # Định nghĩa đường dẫn file (đảm bảo đúng tên file trong folder data)
    f_blds = "BLDS.docx" # Lưu ý: Chroma lưu tên file gốc, không cần đường dẫn đầy đủ 'data/...'
    f_blhs = "BLHS.docx"
    f_bltths = "BLTTHS.docx"
    f_lgtdb = "LGTDB.docx"
    f_anm = "Luật-An-Ninh-Mạng.docx"
    f_nd168 = "ND168.docx"
    f_nd53 = "Nghị-định-53-ND-CP.docx"

    # --- Logic bắt từ khóa ---
    if ("an ninh mạng" in query_lower or "không gian mạng" in query_lower or
        "nghị định 53" in query_lower or "nd-53" in query_lower):
        target_files.add(f_anm)
        target_files.add(f_nd53)
        
    if ("giao thông" in query_lower or "lgtđb" in query_lower or
        "lái xe" in query_lower or "đèn đỏ" in query_lower or "xe máy" in query_lower or
        "nghị định 168" in query_lower or "nd168" in query_lower or "xử phạt" in query_lower):
        target_files.add(f_lgtdb)
        target_files.add(f_nd168)

    if ("hình sự" in query_lower or "blhs" in query_lower or "tội phạm" in query_lower or 
        "tố tụng" in query_lower or "bltths" in query_lower or "khởi tố" in query_lower or 
        "điều tra" in query_lower or "tạm giam" in query_lower or "bị can" in query_lower or "bị cáo" in query_lower or
        "tội " in query_lower): 
        target_files.add(f_blhs)
        target_files.add(f_bltths)
    
    if "dân sự" in query_lower or "blds" in query_lower:
        target_files.add(f_blds)

    # --- SỬA LỖI Ở ĐÂY ---
    if not target_files:
        print("Không phát hiện từ khóa, tìm kiếm toàn bộ tài liệu...")
        return None  # <--- QUAN TRỌNG: Phải trả về None, không được trả về {}
    
    # Lấy tên file từ target_files để filter theo metadata 'source_name' 
    # (Vì ở ingest.py bước trước ta đã lưu metadata['source_name'] = filename)
    
    target_list = list(target_files)
    
    if len(target_list) == 1:
        print(f"Giới hạn tìm kiếm trong file: {target_list[0]}")
        return {"source_name": {"$eq": target_list[0]}} # Dùng source_name thay vì source
    
    print(f"Giới hạn tìm kiếm trong {len(target_list)} file: {target_list}")
    return {"source_name": {"$in": target_list}}

while True:
    query = input("\nCâu hỏi: ").strip()
    if query.lower() in ["exit", "quit"]: break
    if not query: continue

    # --- CẢI TIẾN 5: THÊM PREFIX CHO QUERY ---
    # Model E5 cần "query: " để tìm kiếm ngữ nghĩa
    search_query = f"query: {query}" 
    
    metadata_filter = get_source_filter(query)
    
    print("Đang tìm kiếm...")
    # Tăng k lên 4-5 để có nhiều ngữ cảnh hơn cho model chọn lọc
    docs = vectorstore.similarity_search(
        search_query, 
        k=4, 
        filter=metadata_filter
    )
    
    # Xử lý context và trích xuất nguồn
    context_text = ""
    sources = set()
    
    for i, doc in enumerate(docs):
        # Loại bỏ prefix "passage: " để Qwen đọc dễ hiểu hơn
        clean_content = doc.page_content.replace("passage: ", "")
        source_name = doc.metadata.get('source_name', 'Unknown')
        sources.add(source_name)
        
        # Format lại context cho model dễ đọc
        context_text += f"\n[Tài liệu {i+1} - Nguồn: {source_name}]:\n{clean_content}\n"

    if not context_text:
        print("Không tìm thấy tài liệu liên quan.")
        continue

    formatted_prompt = QA_CHAIN_PROMPT.format(context=context_text, question=query)
    
    print("AI đang suy nghĩ...")
    answer = llm.invoke(formatted_prompt)
    
    print("\n" + "="*40)
    print("TRẢ LỜI:")
    print(answer.strip())
    print("-" * 40)
    # --- CẢI TIẾN 6: GHI NGUỒN TỪ CODE (CHÍNH XÁC HƠN LLM) ---
    print("Nguồn tài liệu tham khảo:", ", ".join(sources))
    print("="*40)