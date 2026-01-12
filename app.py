import os
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from sentence_transformers import CrossEncoder

# --- CẤU HÌNH (Giữ nguyên) ---
MODEL_PATH = "./qwen2.5-3b-instruct-q5_k_m.gguf" 
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
PERSIST_PATH = "./chroma_db"
COLLECTION_NAME = "law_docs"

# --- KHỞI TẠO (Giữ nguyên) ---
print(f"Đang tải model LLM từ: {MODEL_PATH}...")
llm_rewrite = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=-1, n_batch=512, n_ctx=4096, 
    max_tokens=256, # Tăng lên chút
    temperature=0.1, # Tăng nhẹ để mềm dẻo hơn
    top_p=0.95, 
    repeat_penalty=1.1,
    verbose=False, 
    stop=["<|im_end|>", "User:", "VÍ DỤ"] # Chỉ dừng khi hết lượt hoặc sang ví dụ mới
)
callbacks = [StreamingStdOutCallbackHandler()]
llm_chat = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=-1, n_batch=512, n_ctx=4096, max_tokens=2048,
    temperature=0.1, top_p=0.9, repeat_penalty=1.2,
    callbacks=callbacks, verbose=False
)
print("Đang tải Embedding & Reranker...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
reranker = CrossEncoder(RERANK_MODEL)
vectorstore = Chroma(collection_name=COLLECTION_NAME, persist_directory=PERSIST_PATH, embedding_function=embedding_model)

# --- CÁC HÀM TIỆN ÍCH ---
def clean_text(text):
    # Xóa prefix passage
    text = text.replace("passage: ", "") 
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[-_=*]{3,}', '', text)
    # Không dùng gộp dòng \s+ lung tung nữa, vì ta cần dấu \n để định vị tiêu đề
    # Chỉ xóa khoảng trắng thừa đầu đuôi
    return text.strip()

# [MỚI] Hàm trích xuất số hiệu điều luật từ chuỗi
def extract_article_numbers(text):
    # Tìm các mẫu: "Điều 123", "Điều 1", "điều 5"
    matches = re.findall(r"(?:Điều|điều)\s+(\d+[a-z]?)", text)
    # Trả về set để loại bỏ trùng lặp (VD: {'134', '13'})
    return set(matches)

# --- HÀM FILTER (Giữ nguyên logic) ---
def get_source_filter(text_to_scan):
    query_lower = text_to_scan.lower()
    target_files = set()
    f_blds = "BLDS.docx"; f_blhs = "BLHS.docx"; f_bltths = "BLTTHS.docx"; f_lgtdb = "LGTDB.docx"
    f_anm = "Luật-An-Ninh-Mạng.docx"; f_nd168 = "ND168.docx"; f_nd53 = "Nghị-định-53-ND-CP.docx"
    
    keyword_map = {
        "an_ninh_mang": ([f_anm, f_nd53], ["an ninh mạng", "không gian mạng", "nghị định 53", "nd-53", "hacker", "virus", "lừa đảo mạng"]),
        "giao_thong": ([f_lgtdb, f_nd168], ["giao thông", "lgtđb", "lái xe", "đèn đỏ", "xe máy", "ô tô", "nghị định 168", "nd168", "nồng độ cồn"]),
        "hinh_su": ([f_blhs, f_bltths], ["hình sự", "blhs", "bltths", "tố tụng", "tội phạm", "giết", "thương tích", "trộm", "cướp", "lừa đảo", "tham ô", "ma túy", "đánh bạc", "hiếp dâm", "tù", "khởi tố", "tố giác", "che giấu"]),
        "dan_su": ([f_blds], ["dân sự", "blds", "hợp đồng", "bồi thường", "thừa kế", "đất đai", "vay nợ", "ly hôn"])
    }
    for topic, (files, keywords) in keyword_map.items():
        if any(k in query_lower for k in keywords):
            target_files.update(files)
    if not target_files: return None 
    return {"source_name": {"$in": list(target_files)}} if len(target_files) > 1 else {"source_name": {"$eq": list(target_files)[0]}}

# --- TEMPLATE MỚI: YÊU CẦU TRẢ VỀ CẶP [ĐIỀU | FILE] ---
REWRITE_TEMPLATE = """<|im_start|>system
Bạn là chuyên gia pháp lý. Nhiệm vụ: Phân tích câu hỏi và trả về danh sách TỪ KHÓA + CẶP ĐỊNH DANH [Điều luật | Tên file].

DANH SÁCH FILE CHUẨN (Chỉ được dùng các tên file này):
- BLHS.docx (Bộ luật Hình sự - Dùng cho tội danh, hình phạt, tù tội)
- BLTTHS.docx (Tố tụng Hình sự - Dùng cho quy trình, bắt bớ, tạm giam)
- LGTDB.docx (Luật Giao thông), ND168.docx (Nghị định 168 - Phạt xe cộ)
- BLDS.docx (Dân sự - Dùng cho bồi thường, thừa kế, hợp đồng)

TỪ ĐIỂN ÁNH XẠ (BẮT BUỘC TUÂN THỦ):
1. "đánh người", "thương tích", "nhập viện" -> Tội cố ý gây thương tích [Điều 134 | BLHS.docx]
2. "tự vệ", "đánh trả trộm" -> Phòng vệ chính đáng [Điều 22 | BLHS.docx]
3. "không báo công an", "im lặng", "nể tình" -> Tội không tố giác tội phạm [Điều 19 | BLHS.docx], Tội che giấu tội phạm [Điều 18 | BLHS.docx]
4. "say rượu", "bia rượu" -> Phạm tội do dùng rượu bia [Điều 13 | BLHS.docx]
5. "trẻ em", "14 tuổi", "15 tuổi" -> Tuổi chịu trách nhiệm hình sự [Điều 12 | BLHS.docx]
6. "mượn xe đi cướp", "vạ lây" -> Đồng phạm [Điều 17 | BLHS.docx]
7. "mua đồ ăn trộm", "tiêu thụ" -> Tội tiêu thụ tài sản do phạm tội mà có [Điều 323 | BLHS.docx]
8. "lãi suất cao", "nặng lãi" -> Tội cho vay lãi nặng [Điều 201 | BLHS.docx]
9. "đánh bài", "tá lả" -> Tội đánh bạc [Điều 321 | BLHS.docx]

YÊU CẦU:
- Trả về danh sách từ khóa và các thẻ [Điều số | Tên file].
- KHÔNG giải thích.

VÍ DỤ 1:
User: Say rượu đánh người gãy tay
Output: Tội cố ý gây thương tích, [Điều 134 | BLHS.docx], Phạm tội trong tình trạng say, [Điều 13 | BLHS.docx]

VÍ DỤ 2:
User: Mua xe máy ăn trộm có bị bắt không
Output: Tội tiêu thụ tài sản do người khác phạm tội mà có, [Điều 323 | BLHS.docx]

VÍ DỤ 3:
User: {question}
Output:<|im_end|>
<|im_start|>assistant
"""
REWRITE_PROMPT = PromptTemplate(input_variables=["question"], template=REWRITE_TEMPLATE)

def extract_targets(text):
    # Regex bắt mẫu: [Điều 123 | Filename.docx]
    # \d+[a-z]? : Bắt số điều (vd: 134 hoặc 134a)
    # [^\]]+    : Bắt tên file (bất cứ ký tự gì không phải dấu đóng ngoặc vuông)
    matches = re.findall(r"\[Điều\s+(\d+[a-z]?)\s*\|\s*([^\]]+)\]", text, re.IGNORECASE)
    return matches

def rewrite_query_with_llm(user_query):
    print("   [Thinking] Agent đang suy luận từ khóa & điều luật...")
    prompt = REWRITE_PROMPT.format(question=user_query)
    legal_keywords = llm_rewrite.invoke(prompt)
    
    # Xóa rác
    legal_keywords = re.sub(r"^(Output|Trả lời|Keywords):\s*", "", legal_keywords.strip(), flags=re.IGNORECASE)
    legal_keywords = legal_keywords.replace("\n", " ")
    
    print(f"   [Keywords Generated] {legal_keywords}")
    
    # Trích xuất cặp đích danh
    target_pairs = extract_targets(legal_keywords)
    if target_pairs:
        print(f"   [Target Pairs] Tìm ưu tiên: {target_pairs}")
        
    return legal_keywords, target_pairs

# --- ADVANCED RETRIEVAL (CÔNG THỨC MỚI) ---
def advanced_retrieval(search_query, rank_query, target_pairs, metadata_filter):
    print(f"   [Retrieval] Đang tìm kiếm với query: {search_query[:50]}...")
    # Tăng K lên 30 để đảm bảo Điều 13 không bị rơi mất ở vòng gửi xe
    docs = vectorstore.similarity_search(f"query: {search_query}", k=30, filter=metadata_filter)
    if not docs: return []

    doc_contents = [clean_text(d.page_content) for d in docs]
    pairs = [[rank_query, content] for content in doc_contents]
    scores = reranker.predict(pairs)
    
    scored_docs = []
    for doc, score in zip(docs, scores):
        final_score = float(score)
        doc_source = doc.metadata.get('source_name', '')
        
        # --- LOGIC BẢO HỘ TARGET ---
        is_target = False
        for article_num, target_file in target_pairs:
            if target_file.lower().replace(".docx", "") in doc_source.lower():
                pattern = rf"(?:^|\n|passage: )Điều\s+{article_num}[\.\s]"
                if re.search(pattern, doc.page_content, re.IGNORECASE):
                    print(f"   >>> BOOST BẢO HỘ (+15.0) cho [Điều {article_num} | {doc_source}]")
                    # CÔNG THỨC MỚI: Reset điểm âm về 0 rồi mới cộng
                    base_score = max(float(score), 0.0)
                    final_score = base_score + 15.0
                    is_target = True
                    break 
        
        # Nếu không phải target thì giữ nguyên điểm gốc
        if not is_target:
            final_score = float(score)

        doc.metadata['score'] = final_score
        scored_docs.append((doc, final_score))
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    final_docs = []
    print("\n--- RE-RANKING TOP 5 (After Protected Boosting) ---")
    for doc, score in scored_docs[:5]:
        # Hạ ngưỡng lọc xuống thấp hơn nữa để an toàn
        if score > -10.0:
            final_docs.append(doc)
            preview = clean_text(doc.page_content)[:100].replace('\n', ' ')
            print(f"[Score: {score:.2f}] {doc.metadata.get('source_name')}: {preview}...")
            
    return final_docs

# --- HÀM TRẢ LỜI (Giữ nguyên) ---
# --- SỬA LẠI TEMPLATE: TÍCH HỢP TƯ DUY CHO 8 TÌNH HUỐNG ---
ANSWER_TEMPLATE = """<|im_start|>system
Bạn là luật sư AI LawChatter. Nhiệm vụ: Tư vấn pháp luật dựa trên dữ liệu <doc>.

HƯỚNG DẪN TƯ DUY PHÁP LÝ THEO TỪNG CHỦ ĐỀ (BẮT BUỘC THAM CHIẾU):

1. [TỰ VỆ/TRỘM VÀO NHÀ]:
   - Áp dụng Điều 22 (Phòng vệ chính đáng).
   - Phân tích: Nếu đánh trộm khi nó đang tấn công -> Phòng vệ (Không tội). Nếu trộm đã bỏ chạy hoặc bị khống chế mà vẫn đánh gây thương tích nặng -> Vượt quá giới hạn (Có tội theo Điều 134 hoặc 126).

2. [KHÔNG TỐ GIÁC/BẠN BÈ]:
   - Áp dụng Điều 19 (Không tố giác tội phạm).
   - Phân tích: "Bạn thân", "Anh em xã hội" KHÔNG thuộc diện miễn trừ. Chỉ có Ông/Bà/Cha/Mẹ/Vợ/Chồng/Con/Anh chị em RUỘT mới được miễn. -> Vẫn phạm tội.

3. [SAY RƯỢU]:
   - Áp dụng Điều 13 (Phạm tội do dùng rượu bia).
   - Phân tích: Luật quy định người say rượu VẪN PHẢI CHỊU trách nhiệm hình sự. Không được coi là tình tiết giảm nhẹ hay miễn trừ.

4. [TRẺ EM (14-16 TUỔI)]:
   - Áp dụng Điều 12 (Tuổi chịu TNHS).
   - Phân tích: Tuổi này chỉ chịu trách nhiệm về tội RẤT NGHIÊM TRỌNG (Giết người, Cướp, Hiếp dâm...). Nếu chỉ là gây rối trật tự hoặc đánh bạc nhỏ -> Thường chỉ phạt hành chính/giáo dục.

5. [ĐÁNH BẠC/ĐÁNH BÀI]:
   - Áp dụng Điều 321 (Tội đánh bạc).
   - Phân tích: Mốc định lượng là 5 TRIỆU ĐỒNG. Nếu tổng tiền > 5 triệu -> Hình sự. Nếu "vui vui" vài chục ngàn (tổng < 5 triệu) và chưa có tiền án -> Phạt hành chính.

6. [ĐỒNG PHẠM/CHO MƯỢN XE]:
   - Áp dụng Điều 17 (Đồng phạm).
   - Phân tích: Cần xét yếu tố "BIẾT RÕ". Nếu cho mượn xe mà "không biết" bạn đi cướp -> Không phải đồng phạm (chỉ bồi thường dân sự). Nếu "biết rõ" mà vẫn cho -> Đồng phạm giúp sức.

7. [TIÊU THỤ ĐỒ GIAN/MUA ĐỒ RẺ]:
   - Áp dụng Điều 323 (Tiêu thụ tài sản...).
   - Phân tích: Cấu thành tội nếu "BIẾT RÕ" là đồ gian. Việc mua giá "rẻ bèo" là dấu hiệu nghi vấn, nhưng nếu người mua chứng minh được mình không biết -> Không phạm tội hình sự (nhưng bị tịch thu đồ).

8. [CHO VAY NẶNG LÃI]:
   - Áp dụng Điều 201.
   - Phân tích: Lãi suất phải gấp 5 lần mức cao nhất (tức >100%/năm) VÀ thu lợi bất chính > 30 triệu đồng mới xử lý hình sự.

CẤU TRÚC TRẢ LỜI:
### BƯỚC 1: CƠ SỞ PHÁP LÝ
[Trích dẫn nguyên văn Điều luật quan trọng nhất từ <doc> (Số hiệu, Nội dung, Mức phạt)]

### BƯỚC 2: PHÂN TÍCH
- Dựa vào HƯỚNG DẪN trên, phân tích hành vi của người hỏi.
- Đối chiếu các yếu tố: Độ tuổi, Số tiền, Ý thức chủ quan (Biết/Không biết), Mối quan hệ.

### BƯỚC 3: KẾT LUẬN
[Khẳng định: Có phạm tội không? Mức phạt dự kiến là gì?]

<|im_end|>
<|im_start|>user
DỮ LIỆU LUẬT (XML):
{context}

Câu hỏi: {question}
<|im_end|>
<|im_start|>assistant
"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=ANSWER_TEMPLATE)

def main():
    print("\n=== LAWCHATTER AGENT (File-Specific Boosting) ===")
    while True:
        try:
            query = input("\nCâu hỏi: ").strip()
            if query.lower() in ["exit", "quit"]: break
            if not query: continue

            # BƯỚC 1: LLM SINH TỪ KHÓA & CẶP [ĐIỀU|FILE]
            legal_keywords, target_pairs = rewrite_query_with_llm(query)
            
            # Lọc nguồn
            full_scan_text = f"{query} . {legal_keywords}"
            metadata_filter = get_source_filter(full_scan_text)
            
            # BƯỚC 2: TÌM KIẾM & BOOSTING CÓ ĐÍCH NHẮM
            docs = advanced_retrieval(legal_keywords, query, target_pairs, metadata_filter)
            
            if not docs:
                print(">>> Không tìm thấy tài liệu phù hợp.")
                continue

            # BƯỚC 3: CONTEXT & ANSWER
            context_text = "<documents>\n"
            for i, doc in enumerate(docs):
                clean_content = clean_text(doc.page_content)
                source = doc.metadata.get('source_name', 'Unknown')
                context_text += f'<doc id="{i+1}" source="{source}">\n{clean_content}\n</doc>\n'
            context_text += "</documents>"

            print("-" * 50)
            print("AI ĐANG SUY NGHĨ...")
            formatted_prompt = QA_CHAIN_PROMPT.format(context=context_text, question=query)
            llm_chat.invoke(formatted_prompt)
            print("\n" + "-" * 50)
            
        except Exception as e:
            print(f"Lỗi hệ thống: {e}")

if __name__ == "__main__":
    main()