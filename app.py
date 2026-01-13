import os
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from sentence_transformers import CrossEncoder

# --- CẤU HÌNH ---
MODEL_PATH = "./qwen2.5-3b-instruct-q5_k_m.gguf" 
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
PERSIST_PATH = "./chroma_db"
COLLECTION_NAME = "law_docs"

# --- TỪ ĐIỂN ÁNH XẠ (Mapping) ---
PRIMARY_MAPPING = {
    "đánh người": ("134", "BLHS.docx"), "thương tích": ("134", "BLHS.docx"), "nhập viện": ("134", "BLHS.docx"),
    "tự vệ": ("22", "BLHS.docx"), "đánh trả": ("22", "BLHS.docx"), "phòng vệ": ("22", "BLHS.docx"),
    "không báo công an": ("19", "BLHS.docx"), "im lặng": ("19", "BLHS.docx"), "nể tình": ("19", "BLHS.docx"),
    "che giấu": ("18", "BLHS.docx"),
    "say rượu": ("13", "BLHS.docx"), "bia rượu": ("13", "BLHS.docx"), "xỉn": ("13", "BLHS.docx"),
    "trẻ em": ("12", "BLHS.docx"), "14 tuổi": ("12", "BLHS.docx"), "15 tuổi": ("12", "BLHS.docx"),
    "mượn xe": ("17", "BLHS.docx"), "vạ lây": ("17", "BLHS.docx"), "đồng phạm": ("17", "BLHS.docx"),
    "mua đồ ăn trộm": ("323", "BLHS.docx"), "tiêu thụ": ("323", "BLHS.docx"), "đồ gian": ("323", "BLHS.docx"),
    "lãi suất cao": ("201", "BLHS.docx"), "nặng lãi": ("201", "BLHS.docx"), "lãi cắt cổ": ("201", "BLHS.docx"),
    "đánh bài": ("321", "BLHS.docx"), "đánh bạc": ("321", "BLHS.docx"), "tá lả": ("321", "BLHS.docx")
}

# --- GỢI Ý CỨNG (HINT) CHO LLM ---
# Đây là "phao thi" cho LLM 3B, giúp nó không bị ảo giác
HINTS = {
    "19": "Theo Điều 19 Bộ luật Hình sự (BLHS): Hành vi không tố giác tội phạm (im lặng) là VI PHẠM PHÁP LUẬT. Hãy khẳng định người hỏi CÓ TỘI.",
    "13": "Theo Điều 13 Bộ luật Hình sự (BLHS): Phạm tội trong tình trạng say rượu VẪN PHẢI CHỊU TRÁCH NHIỆM HÌNH SỰ. Say rượu không phải là tình tiết giảm nhẹ.",
    "12": "Theo Điều 12 Bộ luật Hình sự (BLHS): Người từ đủ 14 đến dưới 16 tuổi phải chịu trách nhiệm về tội RẤT NGHIÊM TRỌNG. Nếu chỉ đánh nhau nhẹ hoặc gây rối thì thường xử lý hành chính.",
    "22": "Theo Điều 22 Bộ luật Hình sự (BLHS): Đánh trả khi đang bị tấn công là Phòng vệ chính đáng. Nhưng đánh khi kẻ trộm đã bỏ chạy là Vượt quá giới hạn.",
    "321": "Theo Điều 321 Bộ luật Hình sự (BLHS): Đánh bạc trên 5 triệu đồng mới bị xử lý hình sự. Dưới mức này phạt hành chính.",
    "17": "Theo Điều 17 Bộ luật Hình sự (BLHS): Cho mượn xe mà KHÔNG BIẾT bạn đi gây án thì KHÔNG PHẢI ĐỒNG PHẠM.",
    "201": "Theo Điều 201 Bộ luật Hình sự (BLHS): Lãi suất trên 100%/năm VÀ thu lợi > 30 triệu mới bị xử lý Hình sự.",
    "323": "Theo Điều 323 Bộ luật Hình sự (BLHS): Chỉ phạm tội tiêu thụ đồ gian nếu BIẾT RÕ đó là tài sản phạm tội. Nếu vô ý không biết thì không phạm tội hình sự."
}

# --- KHỞI TẠO ---
print(f"Đang tải model LLM từ: {MODEL_PATH}...")
llm_rewrite = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=-1, n_batch=512, n_ctx=4096, 
    max_tokens=256, temperature=0.1, top_p=0.95, repeat_penalty=1.1,
    verbose=False, stop=["<|im_end|>", "User:", "VÍ DỤ"]
)
callbacks = [StreamingStdOutCallbackHandler()]
llm_chat = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=-1, n_batch=512, n_ctx=8192,
    max_tokens=1024,
    temperature=0.1, top_p=0.9, repeat_penalty=1.1,
    f16_kv=True,           # Tiết kiệm RAM
    model_kwargs={
        "frequency_penalty": 1.2, # [QUAN TRỌNG] Phạt nặng nếu lặp lại cả câu
        "presence_penalty": 0.6   # Khuyến khích nói cái mới
    },
    stop=["<|im_end|>", "User:", "CONTEXT:", "CÂU HỎI:"],
    callbacks=callbacks, verbose=False
)
print("Đang tải Embedding & Reranker...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
reranker = CrossEncoder(RERANK_MODEL)
vectorstore = Chroma(collection_name=COLLECTION_NAME, persist_directory=PERSIST_PATH, embedding_function=embedding_model)

# --- UTILS ---
def clean_text(text):
    text = text.replace("passage: ", "") 
    text = re.sub(r'\[.*?\]', '', text)
    return text.strip()

def get_source_filter(text_to_scan):
    query_lower = text_to_scan.lower()
    target_files = set()
    f_blds = "BLDS.docx"; f_blhs = "BLHS.docx"; f_bltths = "BLTTHS.docx"; f_lgtdb = "LGTDB.docx"
    f_anm = "Luật-An-Ninh-Mạng.docx"; f_nd168 = "ND168.docx"; f_nd53 = "Nghị-định-53-ND-CP.docx"
    keyword_map = {
        "an_ninh_mang": ([f_anm, f_nd53], ["an ninh mạng", "không gian mạng", "nghị định 53", "nd-53"]),
        "giao_thong": ([f_lgtdb, f_nd168], ["giao thông", "lgtđb", "lái xe", "đèn đỏ", "xe máy"]),
        "hinh_su": ([f_blhs, f_bltths], ["hình sự", "blhs", "bltths", "tố tụng", "tội phạm", "giết", "thương tích", "trộm", "cướp", "lừa đảo", "tham ô", "ma túy", "đánh bạc", "hiếp dâm", "tù", "khởi tố", "tố giác", "che giấu", "say rượu", "đồng phạm"]),
        "dan_su": ([f_blds], ["dân sự", "blds", "hợp đồng", "bồi thường", "thừa kế", "đất đai", "vay nợ", "ly hôn"])
    }
    for topic, (files, keywords) in keyword_map.items():
        if any(k in query_lower for k in keywords):
            target_files.update(files)
    if not target_files: return None 
    return {"source_name": {"$in": list(target_files)}} if len(target_files) > 1 else {"source_name": {"$eq": list(target_files)[0]}}

# --- REWRITE AGENT ---
REWRITE_TEMPLATE = """<|im_start|>system
Bạn là chuyên gia pháp lý. Nhiệm vụ: Trả về danh sách TỪ KHÓA + CẶP ĐỊNH DANH [Điều luật | Tên file].
VÍ DỤ:
User: Say rượu đánh người
Output: [Điều 134 | BLHS.docx], [Điều 13 | BLHS.docx]
User: {question}
Output:<|im_end|>
<|im_start|>assistant
"""
REWRITE_PROMPT = PromptTemplate(input_variables=["question"], template=REWRITE_TEMPLATE)

def extract_targets(text):
    matches = re.findall(r"\[Điều\s+(\d+[a-z]?)\s*\|\s*([^\]]+)\]", text, re.IGNORECASE)
    return matches

def rewrite_query_with_llm(user_query):
    print("   [Thinking] Agent đang suy luận...")
    prompt = REWRITE_PROMPT.format(question=user_query)
    legal_keywords = llm_rewrite.invoke(prompt)
    legal_keywords = re.sub(r"^(Output|Trả lời|Keywords):\s*", "", legal_keywords.strip(), flags=re.IGNORECASE)
    legal_keywords = legal_keywords.replace("\n", " ")
    print(f"   [Keywords] {legal_keywords}")
    target_pairs = extract_targets(legal_keywords)
    return legal_keywords, target_pairs

# --- [MODIFIED] RETRIEVAL WITH PINNING (GHIM TÀI LIỆU) ---
def advanced_retrieval(search_query, rank_query, llm_targets, primary_targets, metadata_filter):   
    # 1. SEMANTIC SEARCH (Tìm rộng)
    semantic_docs = vectorstore.similarity_search(f"query: {search_query}", k=30, filter=metadata_filter)
    
    # 2. FORCE INJECTION & PINNING (Tìm và GHIM)
    pinned_docs = [] # Danh sách ưu tiên tuyệt đối
    
    if primary_targets:
        print(f"   [Force Injection] {primary_targets}")
        for art_num, target_file in primary_targets:
            # Query đa dạng
            queries = [f"Điều {art_num}", f"nội dung điều {art_num}"]
            specific_filter = {"source_name": target_file} if target_file else metadata_filter
            
            for q in queries:
                # Tìm kiếm
                found = vectorstore.similarity_search(f"query: {q}", k=7, filter=specific_filter)
                for doc in found:
                    # Kiểm tra xem có đúng là điều luật cần tìm không
                    # Regex lỏng: "điều" + khoảng trắng/chấm + số
                    match_pattern = rf"điều\s*[._-]*\s*{art_num}(?:\D|$)"
                    if re.search(match_pattern, doc.page_content, re.IGNORECASE):
                        # Đánh dấu doc này là Pinned
                        doc.metadata['is_pinned'] = True
                        pinned_docs.append(doc)
                        print(f"      -> GHIM THÀNH CÔNG: Điều {art_num}")

    # 3. MERGE & CLEAN
    unique_docs = {}
    
    # Ưu tiên đưa Pinned Docs vào trước
    all_raw_docs = pinned_docs + semantic_docs 
    
    cleaned_doc_objects = []
    for doc in all_raw_docs:
        if doc.page_content not in unique_docs:
            unique_docs[doc.page_content] = doc
            doc.page_content = clean_text(doc.page_content) 
            cleaned_doc_objects.append(doc)

    print(f"   [Merged] {len(cleaned_doc_objects)} docs.")
    if not cleaned_doc_objects: return []

    # 4. RANKING (CHỈ RANK NHỮNG THẰNG KHÔNG ĐƯỢC GHIM)
    # Tách làm 2 nhóm: Pinned và Normal
    final_pinned = []
    normal_docs = []
    
    for doc in cleaned_doc_objects:
        if doc.metadata.get('is_pinned', False):
            final_pinned.append(doc)
        else:
            normal_docs.append(doc)
            
    # Re-rank nhóm Normal
    if normal_docs:
        doc_contents = [d.page_content for d in normal_docs]
        pairs = [[rank_query, content] for content in doc_contents]
        scores = reranker.predict(pairs)
        
        scored_normal = []
        for doc, score in zip(normal_docs, scores):
            # Boost nhẹ cho LLM targets
            final_score = float(score)
            doc.metadata['score'] = final_score
            scored_normal.append((doc, final_score))
        
        scored_normal.sort(key=lambda x: x[1], reverse=True)
        normal_docs = [d for d, s in scored_normal if s > -10.0]

    # 5. KẾT HỢP: Pinned lên đầu + Top 5 Normal
    final_docs = final_pinned + normal_docs[:5]
    
    print("\n--- FINAL CONTEXT LIST ---")
    for doc in final_docs:
        tag = "[PINNED]" if doc.metadata.get('is_pinned') else "[NORMAL]"
        print(f"{tag} {doc.metadata.get('source_name')}: {doc.page_content[:60]}...")
            
    return final_docs

# --- PROMPT VỚI GỢI Ý (HINTS) ---
ANSWER_TEMPLATE = """<|im_start|>system
Bạn là luật sư AI. Dựa vào [GỢI Ý CỦA CHUYÊN GIA] và Context để trả lời.

QUY TẮC QUAN TRỌNG:
1. Đọc [GỢI Ý] và [CONTEXT].
1. Nếu có GỢI Ý, dùng nó làm kết luận chính.
2. NHÌN KỸ "Nguồn" trong Context (VD: BLHS.docx là Bộ luật Hình sự, BLTTHS.docx là Tố tụng Hình sự). Không được trích dẫn nhầm tên luật.
3. Trích dẫn số Điều luật chính xác.
2. Trả lời theo đúng cấu trúc bên dưới.
3. TUYỆT ĐỐI KHÔNG lặp lại nội dung đã viết.

<|im_end|>
<|im_start|>user
CONTEXT:
{context}

[GỢI Ý CỦA CHUYÊN GIA]:
{hints}

CÂU HỎI: {question}
<|im_end|>
<|im_start|>assistant
"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "hints", "question"], template=ANSWER_TEMPLATE)

def main():
    print("\n=== LAWCHATTER AGENT (PUPPET MASTER MODE) ===")
    while True:
        try:
            query = input("\nCâu hỏi: ").strip()
            if query.lower() in ["exit", "quit"]: break
            if not query: continue

            # B1: PRIMARY MATCH & HINT GENERATION
            primary_targets = set()
            active_hints = [] # Danh sách gợi ý sẽ bơm vào Prompt
            
            for kw, target in PRIMARY_MAPPING.items():
                if kw in query.lower():
                    primary_targets.add(target)
                    # Lấy số điều (ví dụ "19")
                    art_num = target[0]
                    if art_num in HINTS:
                        active_hints.append(f"- Về Điều {art_num}: {HINTS[art_num]}")

            if primary_targets: print(f"   [Primary Match] {primary_targets}")
            
            # Tạo chuỗi Hint để đưa vào Prompt
            hint_text = "\n".join(active_hints) if active_hints else "Không có gợi ý đặc biệt."
            if active_hints:
                print(f"   [AI Hint Injection] {hint_text}")

            # B2: REWRITE
            legal_keywords, llm_targets = rewrite_query_with_llm(query)
            
            full_scan_text = f"{query} . {legal_keywords}"
            metadata_filter = get_source_filter(full_scan_text)
            
            # B3: RETRIEVAL (PINNING)
            docs = advanced_retrieval(legal_keywords, query, llm_targets, primary_targets, metadata_filter)
            
            if not docs:
                print(">>> Không tìm thấy tài liệu.")
                continue

            # B4: CONTEXT
            context_text = ""
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source_name', 'Unknown')
                context_text += f"--- NGUỒN: {source} ---\n{doc.page_content}\n\n"

            print("-" * 50)
            print("AI ĐANG ĐỌC TÀI LIỆU VÀ TRẢ LỜI...")
            
            # Truyền thêm biến 'hints' vào Prompt
            formatted_prompt = QA_CHAIN_PROMPT.format(context=context_text, hints=hint_text, question=query)
            
            llm_chat.invoke(formatted_prompt)
            print("\n" + "-" * 50)
            
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()