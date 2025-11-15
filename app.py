import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate

os.environ["ANONYMIZED_TELEMETRY"] = "False"

MODEL_PATH = "./qwen1_5-1_8b-chat-q8_0.gguf" 

if not os.path.exists(MODEL_PATH):
    print(f"Không tìm thấy model tại: {MODEL_PATH}")
    print("Vui lòng tải file GGUF của model về thư mục dự án.")
    exit()

print(f"Đang tải model LLM từ: {MODEL_PATH}...")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=0,
    n_batch=512,
    n_ctx=2048,
    temperature=0.7,
    verbose=False,
    stop=["Dưới đây là", "Câu hỏi:", "\nTrả lời:"]
)
print("Tải LLM thành công!")

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
PERSIST_PATH = "./chroma_db"
COLLECTION_NAME = "law_docs"
DATA_PATH = "data" 

print(f"Đang tải mô hình embedding '{EMBEDDING_MODEL}'...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

print(f"Đang tải Vectorstore từ '{PERSIST_PATH}'...")
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_PATH,
    embedding_function=embedding_model
)

def get_source_filter(query):
    query_lower = query.lower()
    target_files = set()

    f_blds = os.path.join(DATA_PATH, "BLDS.docx")
    f_blhs = os.path.join(DATA_PATH, "BLHS.docx")
    f_bltths = os.path.join(DATA_PATH, "BLTTHS.docx")
    f_lgtdb = os.path.join(DATA_PATH, "LGTDB.docx")
    f_anm = os.path.join(DATA_PATH, "Luật-An-Ninh-Mạng.docx")
    f_nd168 = os.path.join(DATA_PATH, "ND168.docx")
    f_nd53 = os.path.join(DATA_PATH, "Nghị-định-53-ND-CP.docx")

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

    if not target_files:
        print("Không phát hiện từ khóa, tìm kiếm toàn bộ 7 tài liệu...")
        return {}
    
    if len(target_files) == 1:
        file_path = target_files.pop()
        print(f"Tìm kiếm trong file: {file_path}")
        return {"source": {"$eq": file_path}}
    
    print(f"Tìm kiếm trong {len(target_files)} file: {target_files}")
    return {"source": {"$in": list(target_files)}}

prompt_template_str = """Dưới đây là một số điều luật trích từ văn bản pháp lý:
---
{context}
---
Dựa vào văn bản trên, hãy trả lời câu hỏi sau: {input}

Trả lời:"""
prompt = ChatPromptTemplate.from_template(prompt_template_str)

while True:
    query = input("\nCâu hỏi: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Tạm biệt!")
        break
    if not query:
        continue

    try:
        print("Đang tìm kiếm tài liệu...")
        
        metadata_filter = get_source_filter(query)

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": metadata_filter
            }
        )
        
        docs = retriever.invoke(query)
        
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        
        if not context:
            context = "Không tìm thấy tài liệu liên quan."
            
        formatted_prompt = prompt.format(context=context, input=query)
        
        print("Đang suy nghĩ...")
        answer = llm.invoke(formatted_prompt)
        
        print("\nTrả lời:\n")
        print(answer) 
        print("-" * 80)

    except Exception as e:
        print(f"Lỗi khi thực thi RAG chain: {e}")
        continue