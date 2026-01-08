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
def advanced_retrieval(query, metadata_filter, top_k_final=3):
    # BƯỚC 1: Lấy rộng
    print("-> 1. Tìm kiếm Vector (Lấy top 15)...")
    search_query = f"query: {query}"
    initial_docs = vectorstore.similarity_search(search_query, k=15, filter=metadata_filter)
    
    if not initial_docs:
        return []

    # BƯỚC 2: Re-ranking
    print("-> 2. Re-ranking (AI chấm điểm độ liên quan)...")
    doc_contents = [clean_text(d.page_content) for d in initial_docs]
    pairs = [[query, content] for content in doc_contents]
    
    scores = reranker.predict(pairs)
    
    # Ghép điểm vào và sắp xếp
    scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    
    # BƯỚC 3: Lọc và LƯU ĐIỂM VÀO METADATA
    final_docs = []
    print("\n--- KẾT QUẢ RE-RANKING ---")
    for doc, score in scored_docs[:top_k_final]:
        if score > -4.0: 
            # --- QUAN TRỌNG: Lưu điểm vào metadata để main() in ra được ---
            doc.metadata['score'] = float(score) 
            final_docs.append(doc)
            
            source_name = doc.metadata.get('source_name', 'Unknown')
            # In thử 60 ký tự đầu
            print(f"[Score: {score:.2f}] {source_name}: {clean_text(doc.page_content)[:60]}...")
            
    return final_docs

def main():
    print("\n=== LAWCHATTER ADVANCED (Rerank Enabled) ===")
    while True:
        try:
            query = input("\nCâu hỏi: ").strip()
            if query.lower() in ["exit", "quit"]: break
            if not query: continue

            metadata_filter = get_source_filter(query)
            
            # Sử dụng hàm tìm kiếm nâng cao
            docs = advanced_retrieval(query, metadata_filter)
            
            print("\n--- TÀI LIỆU ĐƯA VÀO (XML FORMAT) ---")
            context_text = "<documents>\n" # Mở thẻ bao quanh
            
            for i, doc in enumerate(docs):
                clean_content = clean_text(doc.page_content)
                source = doc.metadata.get('source_name', 'Unknown')
                
                # --- CẢI TIẾN: Dùng thẻ XML để cô lập dữ liệu ---
                context_chunk = (
                    f'<doc id="{i+1}" source="{source}">\n'
                    f'{clean_content}\n'
                    f'</doc>\n'
                )
                context_text += context_chunk
                
                # In ra debug gọn hơn
                print(f"[{i+1}] {source} (Score: {doc.metadata.get('score', 'N/A')})")

            context_text += "</documents>"

            print("-" * 50)
            print("AI ĐANG SUY NGHĨ...")
            formatted_prompt = QA_CHAIN_PROMPT.format(context=context_text, question=query)
            llm.invoke(formatted_prompt)
            print("\n" + "-" * 50)
            
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()