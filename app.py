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
# Model Re-ranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
PERSIST_PATH = "./chroma_db"
COLLECTION_NAME = "law_docs"

# --- HÀM LÀM SẠCH VĂN BẢN ---
def clean_text(text):
    text = text.replace("passage: ", "")
    text = re.sub(r'[-_=*]{3,}', '', text)
    text = re.sub(r'\s+', ' ', text) 
    return text.strip()

# --- HÀM LỌC NGUỒN ---
def get_source_filter(query):
    query_lower = query.lower()
    target_files = set()

    # 1. Định nghĩa file
    # Lưu ý: Tên file phải khớp chính xác với metadata 'source_name' trong ChromaDB
    f_blds = "BLDS.docx"
    f_blhs = "BLHS.docx"
    f_bltths = "BLTTHS.docx"
    f_lgtdb = "LGTDB.docx"
    f_anm = "Luật-An-Ninh-Mạng.docx"
    f_nd168 = "ND168.docx"
    f_nd53 = "Nghị-định-53-ND-CP.docx"

    # 2. Từ điển từ khóa (Keyword Mapping)
    # Cấu trúc: { "Nhóm Luật": ([Danh sách file], [Danh sách từ khóa]) }
    keyword_map = {
        "an_ninh_mang": (
            [f_anm, f_nd53],
            [
                "an ninh mạng", "không gian mạng", "nghị định 53", "nd-53",
                "hacker", "tấn công mạng", "virus", "mã độc", "dữ liệu cá nhân",
                "thông tin xấu độc", "tin giả", "xúc phạm trên mạng", "facebook", "zalo",
                "hệ thống thông tin", "bảo vệ dữ liệu", "chiếm đoạt tài khoản"
            ]
        ),
        "giao_thong": (
            [f_lgtdb, f_nd168],
            [
                "giao thông", "lgtđb", "lái xe", "đèn đỏ", "xe máy", "ô tô",
                "nghị định 168", "nd168", "xử phạt", "phạt nguội",
                "nồng độ cồn", "rượu bia", "tước bằng", "giấy phép lái xe", "bằng lái",
                "quá tốc độ", "lấn làn", "ngược chiều", "đội mũ bảo hiểm",
                "tai nạn", "cảnh sát giao thông", "csgt", "tạm giữ xe"
            ]
        ),
        "hinh_su": (
            [f_blhs, f_bltths],
            [
                # Nhóm định danh luật
                "hình sự", "blhs", "bltths", "tố tụng", 
                # Nhóm tội danh phổ biến
                "tội phạm", "giết người", "cố ý gây thương tích", "trộm cắp", "cướp",
                "lừa đảo", "tham ô", "hối lộ", "ma túy", "đánh bạc", "hiếp dâm",
                "vu khống", "làm nhục", "gây rối", "buôn lậu",
                # Nhóm thủ tục/đối tượng
                "khởi tố", "điều tra", "tạm giam", "bắt người", "khám xét",
                "bị can", "bị cáo", "bị hại", "luật sư bào chữa", "phiên tòa",
                "án treo", "tù chung thân", "tử hình", "tiền án", "xóa án tích"
            ]
        ),
        "dan_su": (
            [f_blds],
            [
                "dân sự", "blds", "bộ luật dân sự",
                "hợp đồng", "bồi thường thiệt hại", "thừa kế", "di chúc",
                "tài sản", "đất đai", "quyền sở hữu", "đặt cọc", "thế chấp",
                "vay nợ", "tranh chấp", "ly hôn", "giám hộ", "đại diện",
                "pháp nhân", "cá nhân", "thỏa thuận"
            ]
        )
    }

    # 3. Quét từ khóa
    print("Đang phân tích từ khóa trong câu hỏi...")
    detected_topics = []

    for topic, (files, keywords) in keyword_map.items():
        # Kiểm tra xem có từ khóa nào xuất hiện trong query không
        if any(k in query_lower for k in keywords):
            target_files.update(files)
            detected_topics.append(topic)

    # 4. Trả về kết quả
    if not target_files:
        print("-> Không phát hiện từ khóa chuyên ngành. Tìm kiếm trên TOÀN BỘ dữ liệu.")
        return None # Tìm tất cả
    
    target_list = list(target_files)
    print(f"-> Phát hiện chủ đề: {detected_topics}")
    print(f"-> Giới hạn tìm kiếm trong {len(target_list)} file: {target_list}")
    
    # ChromaDB filter syntax
    if len(target_list) == 1:
        return {"source_name": {"$eq": target_list[0]}}
    else:
        return {"source_name": {"$in": target_list}}

def enhance_legal_query(query):
    query_lower = query.lower()
    additional_terms = []

    mappings = [
        # Thêm "Bộ luật hình sự" vào cuối mỗi mapping để Force Filter hoạt động
        (["tự vệ", "đánh nó", "vào nhà"], "Phòng vệ chính đáng Điều 22 Bộ luật hình sự"),
        (["không báo công an", "im lặng", "biết", "giết người"], "Tội không tố giác tội phạm Điều 19 Điều 390 Bộ luật hình sự"),
        (["say rượu", "say bia", "lỡ tay"], "Phạm tội trong tình trạng say do dùng rượu bia Điều 13 Bộ luật hình sự"),
        (["14 tuổi", "15 tuổi", "16 tuổi", "trẻ em", "nhỏ tuổi", "nghịch dại"], "Tuổi chịu trách nhiệm hình sự Điều 12 Bộ luật hình sự"),
        (["đánh bài", "vui vui", "mấy chục ngàn"], "Tội đánh bạc Điều 321 Bộ luật hình sự"),
        (["cho mượn xe", "vạ lây", "đồng phạm"], "Đồng phạm Điều 17 Bộ luật hình sự"),
        (["mua", "iphone cũ", "đồ ăn trộm", "tiêu thụ"], "Tội tiêu thụ tài sản do người khác phạm tội mà có Điều 323 Bộ luật hình sự"),
        (["lãi suất cao", "nặng lãi", "cho vay"], "Tội cho vay lãi nặng Điều 201 Bộ luật hình sự")
    ]

    for keywords, legal_term in mappings:
        match_count = sum(1 for k in keywords if k in query_lower)
        if match_count >= 1:
             if legal_term not in additional_terms:
                additional_terms.append(legal_term)

    if additional_terms:
        # Đưa thuật ngữ luật lên đầu để E5 chú ý hơn
        enhanced_query = f"{', '.join(additional_terms)}. Nội dung câu hỏi: {query}"
        print(f"\n[DEBUG] Query gốc: {query}")
        print(f"[DEBUG] Query mở rộng: {enhanced_query}")
        return enhanced_query
    
    return query

# --- KHỞI TẠO ---
print(f"Đang tải model LLM từ: {MODEL_PATH}...")
callbacks = [StreamingStdOutCallbackHandler()]
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=-1, n_batch=512, n_ctx=4096, max_tokens=2048,
    temperature=0.1, top_p=0.9, top_k=40, repeat_penalty=1.15,
    callbacks=callbacks, verbose=False,
    stop=["<|im_end|>", "Người dùng:", "Kết thúc"]
)

print("Đang tải Embedding Model & Reranker...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

# Load Cross-Encoder (Dùng để chấm điểm lại độ chính xác)
reranker = CrossEncoder(RERANK_MODEL)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_PATH,
    embedding_function=embedding_model
)

template = """<|im_start|>system
Bạn là trợ lý pháp lý ảo LawChatter.

NHIỆM VỤ:
Sử dụng thông tin trong thẻ <documents> để trả lời câu hỏi.

QUY TRÌNH SUY LUẬN:
1. Xác định đúng đoạn văn bản chứa câu trả lời trong các thẻ <doc>.
2. Trích xuất "Số hiệu điều luật" (thường nằm ngay đầu đoạn văn, ví dụ: "Điều 168", "Điều 20"...).
3. Tổng hợp đầy đủ các quy định/khung hình phạt.

YÊU CẦU ĐẦU RA:
- Bắt đầu câu trả lời bằng cụm từ: "Theo quy định tại [Số hiệu điều luật]..."
- Trình bày nội dung chi tiết, đầy đủ các trường hợp (nếu có nhiều khoản).
- Kết thúc câu trả lời bằng: (Nguồn: [Tên_File])

VÍ DỤ MẪU:
Câu hỏi: Tội cướp tài sản bị phạt thế nào?
Trả lời:
Theo quy định tại Điều 168 Bộ luật Hình sự:
- Người nào dùng vũ lực đe dọa chiếm đoạt tài sản thì bị phạt tù từ 03 năm đến 10 năm.
- Nếu phạm tội có tổ chức hoặc gây thương tích, có thể bị phạt tù từ 07 năm đến 15 năm.
(Nguồn: BLHS.docx)
<|im_end|>
<|im_start|>user
DỮ LIỆU LUẬT (XML):
{context}

Câu hỏi: {question}
<|im_end|>
<|im_start|>assistant
Câu trả lời:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# --- LOGIC TÌM KIẾM NÂNG CAO (HYBRID-LIKE) ---
def advanced_retrieval(original_query, enhanced_query, metadata_filter, top_k_final=3):
    print("-> 1. Tìm kiếm Vector (Lấy top 15)...")
    
    # Dùng enhanced_query để tìm kiếm Vector
    search_query = f"query: {enhanced_query}" 
    initial_docs = vectorstore.similarity_search(search_query, k=15, filter=metadata_filter)
    
    if not initial_docs:
        return []

    print("-> 2. Re-ranking (AI chấm điểm độ liên quan)...")
    doc_contents = [clean_text(d.page_content) for d in initial_docs]
    
    # --- SỬA ĐỔI QUAN TRỌNG TẠI ĐÂY ---
    # Thay vì so sánh với original_query (vốn nhiều từ lóng),
    # ta so sánh với enhanced_query (chứa thuật ngữ luật) để Re-ranker chấm điểm chuẩn hơn.
    pairs = [[enhanced_query, content] for content in doc_contents]
    
    scores = reranker.predict(pairs)
    
    scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    
    final_docs = []
    print("\n--- KẾT QUẢ RE-RANKING ---")
    for doc, score in scored_docs[:top_k_final]:
        # Giảm ngưỡng lọc xuống chút (-5.0) để tránh bỏ sót nếu model chấm hơi gắt
        if score > -5.0: 
            doc.metadata['score'] = float(score) 
            final_docs.append(doc)
            source_name = doc.metadata.get('source_name', 'Unknown')
            print(f"[Score: {score:.2f}] {source_name}: {clean_text(doc.page_content)[:60]}...")
            
    return final_docs

def main():
    print("\n=== LAWCHATTER ADVANCED (Logic Fixed) ===")
    while True:
        try:
            query = input("\nCâu hỏi: ").strip()
            if query.lower() in ["exit", "quit"]: break
            if not query: continue

            # --- SỬA ĐỔI 1: Mở rộng truy vấn NGAY TỪ ĐẦU ---
            # Để các từ khóa như "Bộ luật hình sự", "Điều 321" được thêm vào
            enhanced_query = enhance_legal_query(query)

            # --- SỬA ĐỔI 2: Lọc nguồn dựa trên câu hỏi ĐÃ MỞ RỘNG ---
            # Ví dụ: Câu hỏi gốc chỉ có "xe máy" (Giao thông), 
            # nhưng enhanced_query có thêm "Bộ luật hình sự" -> Lấy cả 2 nguồn.
            metadata_filter = get_source_filter(enhanced_query) 
            
            # 3. Tìm kiếm nâng cao
            docs = advanced_retrieval(query, enhanced_query, metadata_filter)
            
            print("\n--- TÀI LIỆU ĐƯA VÀO (XML FORMAT) ---")
            context_text = "<documents>\n"
            
            # --- SỬA ĐỔI 3: Chỉ lấy Top 2 văn bản tốt nhất để tránh nhiễu ---
            for i, doc in enumerate(docs[:2]): 
                clean_content = clean_text(doc.page_content)
                source = doc.metadata.get('source_name', 'Unknown')
                score = doc.metadata.get('score', 0)
                
                context_chunk = (
                    f'<doc id="{i+1}" source="{source}" score="{score:.2f}">\n'
                    f'{clean_content}\n'
                    f'</doc>\n'
                )
                context_text += context_chunk
                print(f"[{i+1}] {source} (Score: {score:.2f})")
            context_text += "</documents>"

            if not docs:
                print(">>> Không tìm thấy tài liệu phù hợp.")
                continue

            print("-" * 50)
            print("AI ĐANG SUY NGHĨ...")
            # Vẫn dùng query gốc để prompt tự nhiên, nhưng context đã chính xác
            formatted_prompt = QA_CHAIN_PROMPT.format(context=context_text, question=query)
            llm.invoke(formatted_prompt)
            print("\n" + "-" * 50)
            
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()