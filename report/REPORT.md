# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Dương Quang Đông
**Nhóm:** Nhóm 6
**Ngày:** 04/10/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

> High cosine similarity nghĩa là hai vector embedding có hướng rất giống nhau, tức là hai câu/văn bản có nội dung ngữ nghĩa gần nhau. Giá trị càng gần 1 thì mức độ tương đồng càng cao

**Ví dụ HIGH similarity:**

- **Sentence A:** “The cat is sleeping on the sofa.”
- **Sentence B:** “A kitten is resting on the couch.”
- **Tại sao tương đồng:** Cả hai câu đều nói về con mèo đang nằm nghỉ trên ghế sofa, khác từ nhưng cùng ý nghĩa

**Ví dụ LOW similarity:**

- **Sentence A:** “The cat is sleeping on the sofa.”
- **Sentence B:** “The stock market crashed yesterday.”
- **Tại sao khác:** Hai câu nói về hai chủ đề hoàn toàn khác nhau: một câu về động vật, một câu về tài chính.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

> Cosine similarity đo góc giữa các vector nên tập trung vào ngữ nghĩa thay vì độ lớn vector. Với text embeddings, hướng vector quan trọng hơn khoảng cách tuyệt đối, nên cosine thường phản ánh similarity tốt hơn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> Công thức:
>
> Để tính được số chunks, ta sử dụng công thức:
>
> num_chunks = ceil[(doc_size - overlap)/(chunk_size - overlap)]
>
> Trong đó ceil là hàm trần (làm tròn lên)
>
> Phép tính:
>
> num_chunks = ceil[(10,000 - 50)/(500 - 50)]
>
> = ceil[9,950/450]
> = ceil[22.11]
> = 23 chunks
>
> **Đáp án:** 23 chunks

**Trực quan:** Chunk đầu chiếm ký tự 1–500. Chunk tiếp theo bắt đầu tại 451 (lùi lại 50 để overlap).Mỗi bước tiến thêm 450 ký tự cho đến khi hết tài liệu.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

> Phép tính:
>
> num_chunks = ceil[(10,000 - 100)/(500 - 100)]
>
> = ceil[9,900/400]
>
> = ceil[24.75] = 25 chunks
>
> **Đáp án:** 25 chunks

**Trực quan:** Chunk đầu chiếm ký tự 1–500. Chunk tiếp theo bắt đầu tại 451 (lùi lại 50 để overlap). Mỗi bước tiến thêm 450 ký tự cho đến khi hết tài liệu.

Chunk count **tăng thêm 2** (từ 23 → 25) vì bước tiến mỗi chunk ngắn hơn (400 thay vì 450), cần nhiều chunk hơn để phủ hết tài liệu.

> **Tại sao muốn overlap nhiều hơn?** Overlap đảm bảo các câu nằm ở ranh giới giữa hai chunk không bị cắt đứt ngữ cảnh — khi retrieval trả về một chunk, thông tin liền kề vẫn được giữ lại ở cả chunk trước lẫn sau, giúp model trả lời chính xác hơn.

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Vietnamese cooking recipes

**Tại sao nhóm chọn domain này?**

> Các tài liệu công thức nấu ăn có cấu trúc rất rõ ràng, bao gồm các phần cố định như Giới thiệu, Nguyên liệu và Các bước thực hiện. Cấu trúc này rất phù hợp để đánh giá xem chiến lược chunking có bảo toàn được trọn vẹn ngữ cảnh của một bước nấu hay danh sách nguyên liệu hay không. Đồng thời, nó cho phép tạo ra các benchmark queries thực tế và phong phú.

### Data Inventory
| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Savory Pancakes (Bánh Khọt) | vietnamtourism | 1981 | source, extension, category, difficulty, doc_id, chunk_index |
| 2 | Braised Tofu with Quail Eggs | vietnamtourism | 1210 | source, extension, category, difficulty, doc_id, chunk_index |
| 3 | Duck Porridge & Salad (Cháo Gỏi Vịt) | vietnamtourism | 2470 | source, extension, category, difficulty, doc_id, chunk_index |
| 4 | Grilled Snails with Salt & Chili | vietnamtourism | 1014 | source, extension, category, difficulty, doc_id, chunk_index |
| 5 | Orange Fruit Skin Jam (Mứt Vỏ Cam) | vietnamtourism | 1226 | source, extension, category, difficulty, doc_id, chunk_index |

### Metadata Schema
| Trường metadata | Kiểu   | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|-----------------|--------|---------------|-------------------------------|
| source | string | "Braised_Tofu" | Tên gốc document — dùng để trace kết quả về file nguồn |
| extension | string | ".md" | Loại file — hỗ trợ filter theo định dạng nếu mix .md/.txt |
| category | string | "main_dish", "seafood", "dessert" | Filter theo loại món — VD: chỉ tìm trong dessert hoặc seafood |
| difficulty | string | "easy", "medium", "hard" | Filter theo độ khó — VD: chỉ tìm món dễ nấu |
| doc_id | string | "Orange_Fruit_Skin_Jam" | ID gốc của document trước khi chunk — dùng để delete_document và group chunks cùng nguồn |
| chunk_index | int | 0, 1, 2... | Vị trí chunk trong document — hỗ trợ debug và tái tạo thứ tự nội dung gốc |
---
## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu                     | Strategy                         | Chunk Count | Avg Length | Preserves Context?                                                                                 |
| ---------------------------- | -------------------------------- | ----------- | ---------- | -------------------------------------------------------------------------------------------------- |
| Braised Tofu with Quail Eggs | FixedSizeChunker (`fixed_size`)  | 3           | 436.67     | Trung bình: giữ được độ dài lớn nhưng có cắt giữa câu và giữa bước nấu                             |
| Braised Tofu with Quail Eggs | SentenceChunker (`by_sentences`) | 3           | 402.00     | Khá tốt: bám theo câu nên dễ đọc hơn, nhưng vẫn gom nhiều ý trong cùng một chunk                   |
| Braised Tofu with Quail Eggs | RecursiveChunker (`recursive`)   | 4           | 301.25     | Tốt nhất: tách nhỏ hơn và thường giữ được cụm ý như nguyên liệu hoặc từng bước                     |
| Duck Porridge & Salad        | FixedSizeChunker (`fixed_size`)  | 6           | 453.33     | Trung bình: chunk dài nhưng nhiều chỗ bị cắt dở danh sách nguyên liệu                              |
| Duck Porridge & Salad        | SentenceChunker (`by_sentences`) | 6           | 410.83     | Khá tốt: phần giới thiệu rõ ràng hơn, nhưng tài liệu dài nên vẫn có chunk chứa quá nhiều thông tin |
| Duck Porridge & Salad        | RecursiveChunker (`recursive`)   | 7           | 352.00     | Tốt nhất: chia mượt hơn theo cấu trúc văn bản, phù hợp tài liệu dài và nhiều mục                   |
| Orange Fruit Skin Jam        | FixedSizeChunker (`fixed_size`)  | 3           | 442.00     | Trung bình: số chunk ít nhưng bị cắt giữa quy trình ở các bước sau                                 |
| Orange Fruit Skin Jam        | SentenceChunker (`by_sentences`) | 4           | 305.75     | Tốt: tách được các bước rõ hơn, thuận lợi cho truy xuất theo hành động                             |
| Orange Fruit Skin Jam        | RecursiveChunker (`recursive`)   | 4           | 305.50     | Tốt nhất: cân bằng giữa độ dài và ngữ cảnh, giữ được ranh giới hợp lý giữa các phần                |

Nhìn chung, `fixed_size` tạo ít chunk hơn và độ dài trung bình cao hơn, nhưng thường cắt ngang câu hoặc cắt giữa phần nguyên liệu và quy trình nên ngữ cảnh bị gãy. `by_sentences` cải thiện độ mạch lạc vì tách theo câu, phù hợp khi truy vấn hỏi theo từng bước nấu ăn. `recursive` cho kết quả ổn định nhất trên cả 3 tài liệu vì ưu tiên tách theo separator tự nhiên, nên các chunk ngắn hơn, dễ đọc hơn và bảo toàn ý nghĩa cục bộ tốt hơn cho retrieval.

### Strategy Của Tôi

**Loại:** `RecursiveChunker`

**Mô tả cách hoạt động:**

> Strategy này chunk theo kiểu đệ quy, lần lượt thử các separator theo thứ tự ưu tiên `\n\n`, `\n`, `. `, ` ` rồi cuối cùng mới fallback sang `""` để cắt trực tiếp theo độ dài cố định. Ý tưởng là luôn ưu tiên các ranh giới tự nhiên của văn bản như đoạn, dòng, câu và từ trước khi phải chia cứng theo số ký tự. Sau khi tách thành các mảnh nhỏ hơn, thuật toán còn ghép lại các mảnh thành chunk lớn nhất có thể nhưng không vượt quá `chunk_size`, nên chunk đầu ra vẫn gọn mà không bị vỡ ngữ cảnh quá nhiều. Base case là khi đoạn hiện tại đã ngắn hơn `chunk_size` thì trả về luôn, còn nếu không còn separator phù hợp thì cắt tuần tự theo từng đoạn dài `chunk_size`.

**Tại sao tôi chọn strategy này cho domain nhóm?**

> Domain công thức nấu ăn có cấu trúc khá rõ: tiêu đề, phần nguyên liệu, rồi các bước thực hiện được trình bày theo dòng hoặc theo đoạn ngắn. Recursive chunking tận dụng rất tốt các dấu xuống dòng và ranh giới câu này để giữ nguyên từng cụm ý, thay vì cắt ngang giữa danh sách nguyên liệu hay giữa một bước nấu ăn như fixed-size chunking. So với `SentenceChunker`, nó cũng linh hoạt hơn khi tài liệu có nhiều list nhiều dòng hoặc các đoạn mô tả dài ngắn không đều.

**Code snippet (nếu custom):**

```python
class RecursiveChunker:
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return self._split(text.strip(), self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        # ưu tiên tách theo ranh giới tự nhiên, hết lựa chọn thì mới cắt cứng
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy      | Chunk Count | Avg Length | Retrieval Quality? |
| -------- | ------------- | ----------- | ---------- | ------------------ |
| Braised Tofu with Quail Eggs | best baseline: `SentenceChunker` | 3 | 402.00 | Khá tốt: truy xuất ổn với câu hỏi về nguyên liệu và quy trình, nhưng vẫn có chunk gom nhiều ý nên đôi lúc trả về hơi rộng |
| Braised Tofu with Quail Eggs | **của tôi: `RecursiveChunker`** | 4 | 301.25 | Tốt hơn: chunk gọn hơn, bám sát từng cụm ý nên dễ lấy đúng phần nguyên liệu hoặc từng bước nấu |
| Duck Porridge & Salad | best baseline: `SentenceChunker` | 6 | 410.83 | Khá tốt: giữ được câu hoàn chỉnh nhưng với tài liệu dài vẫn có chunk chứa nhiều thông tin, làm giảm độ sắc nét khi retrieve |
| Duck Porridge & Salad | **của tôi: `RecursiveChunker`** | 7 | 352.00 | Tốt hơn: chia mượt theo cấu trúc văn bản, giúp các truy vấn hỏi quy trình nấu cháo hoặc phần gỏi vịt ra đúng ngữ cảnh hơn |
| Orange Fruit Skin Jam | best baseline: `SentenceChunker` | 4 | 305.75 | Tốt: đã tách rõ các bước chính và đủ dùng cho truy vấn về cách làm hoặc bảo quản |
| Orange Fruit Skin Jam | **của tôi: `RecursiveChunker`** | 4 | 305.50 | Nhỉnh hơn nhẹ: độ dài gần tương đương nhưng ranh giới chunk tự nhiên hơn, nên phần bước làm và lưu trữ ít bị dính vào nhau |

### So Sánh Với Thành Viên Khác

Tất cả chạy trên cùng 5 documents, cùng embedder `all-MiniLM-L6-v2`, cùng 5 benchmark queries.

| Thành viên | Strategy | Chunks | Q1 Top-1 | Q2 Top-1 | Q3 Top-1 | Q4 Top-1 | Q5 Top-1 | Top-3 Relevant |
|-----------|----------|--------|----------|----------|----------|----------|----------|----------------|
| Nguyễn Lê Trung | FixedSizeChunker (300/50) | 32 | Braised_Tofu:2 (0.7012) | Grilled_Snails:3 (0.7107) ✓ | Duck_Porridge:5 (0.6558) ✓ | Orange_Fruit_Skin_Jam:4 (0.4947) △ | Savory_Pancakes:2 (0.6186) ✓ | **5/5** |
| Phạm Anh Dũng | SentenceChunker (3 sentences) | 34 | Braised_Tofu:1 (0.7493) ✓ | Grilled_Snails:2 (0.6763) ✓ | Duck_Porridge:4 (–) ✓ | Orange_Fruit_Skin_Jam:5 (0.4988) ✓ | Savory_Pancakes:1 (0.5978) ✓ | **5/5** |
| Tôi (Dương Quang Đông) | RecursiveChunker (300) | 39 | Braised_Tofu:3 (0.7287) ✓ | Grilled_Snails:3 (0.7001) ✓ | Duck_Porridge:5 (0.7640) ✓ | Orange_Fruit_Skin_Jam:5 (0.5260) ✓ | Savory_Pancakes:4 (0.6530) ✓ | **5/5** |
| Vương Hoàng Giang | CustomRecipeChunker (by header) | 39 | Braised_Tofu:1 (0.7420) ✓ | Grilled_Snails:5 (0.6438) △ | Duck_Porridge:5 (0.7667) ✓ | Orange_Fruit_Skin_Jam:6 (0.5260) ✓ | Savory_Pancakes:1 (0.6275) ✓ | **5/5** |

> ✓ = top-1 đúng document | △ = top-1 sai nhưng có trong top-3

**So sánh chi tiết từng strategy:**

| Strategy | Điểm mạnh | Điểm yếu |
|----------|-----------|----------|
| **FixedSizeChunker** | Đơn giản, nhất quán; overlap 50 giữ ngữ cảnh biên; chunk lớn dễ chứa đủ 1 bước nấu | Cắt qua ranh giới câu và section; Q4 top-1 không chứa tên món |
| **SentenceChunker** | Giữ trọn câu; Q1 top-1 chính xác nhất (0.7493) vì chunk Ingredients nguyên vẹn | Gom nhầm nội dung từ nhiều section; Q2 không lấy đúng "Step 5 Making sauce" |
| **RecursiveChunker** (của tôi)| Score Q3 cao nhất (0.7640); tôn trọng `\n\n` → thường giữ đúng paragraph | Tạo nhiều chunk nhỏ hơn → đôi khi phân mảnh bước nấu dài |
| **CustomRecipeChunker** | Chunk hoàn toàn khớp cấu trúc recipe; Ingredients chunk độc lập, mỗi Step là 1 unit | Q2 top-1 lấy Step 4 (grill snails) thay vì Step 5 (making sauce) vì Step 5 quá ngắn → score thấp |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> `RecursiveChunker` cho kết quả retrieval tốt nhất tổng thể — score Q3 cao nhất (0.7640), 5/5 top-1 đúng document, và tôn trọng ranh giới tự nhiên của văn bản recipe (paragraph breaks). `CustomRecipeChunker` có thiết kế lý tưởng về mặt ngữ nghĩa nhưng gặp vấn đề với các step quá ngắn (Step 5 của Grilled Snails chỉ 1 câu → chunk yếu về ngữ nghĩa embedding). `FixedSizeChunker` (strategy của tôi) vẫn đạt 5/5 nhờ overlap nhưng cần tăng `chunk_size` lên ~400 để tránh cắt giữa section quan trọng.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

> _Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?_
> Tôi dùng `re.split(r"(?<=[.!?])(?:\\s+|\\n+)")` để tách câu dựa trên các dấu kết thúc câu như `.`, `!`, `?`, sau đó yêu cầu phải có khoảng trắng hoặc xuống dòng ngay sau dấu câu. Sau khi tách, tôi `strip()` từng câu và bỏ các phần rỗng để xử lý trường hợp văn bản có nhiều khoảng trắng hoặc nhiều dòng trống liên tiếp. Cuối cùng, các câu được gom lại theo từng nhóm `max_sentences_per_chunk` để tạo ra các chunk dễ đọc và vẫn giữ được mạch nội dung.

**`RecursiveChunker.chunk` / `_split`** — approach:

> _Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?_
> `chunk()` chỉ xử lý đầu vào rỗng rồi gọi helper `_split()` trên chuỗi đã `strip()`. Trong `_split()`, thuật toán thử lần lượt các separator theo thứ tự `\n\n`, `\n`, `. `, ` `, `""`; nếu tách được nhiều mảnh thì nó cố gắng ghép các mảnh lại thành chunk lớn nhất có thể nhưng không vượt quá `chunk_size`, còn nếu một mảnh vẫn quá dài thì tiếp tục đệ quy với separator chi tiết hơn. Base case là khi đoạn hiện tại rỗng thì trả `[]`, khi độ dài đã nhỏ hơn hoặc bằng `chunk_size` thì trả luôn `[current_text]`, và khi không còn separator phù hợp thì fallback sang cắt trực tiếp theo từng đoạn dài `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:

> _Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?_
> Khi thêm tài liệu, tôi embed `doc.content` bằng `embedding_fn`, chuẩn hóa metadata rồi lưu thành record gồm `id`, `content`, `metadata`, và `embedding`; nếu môi trường có ChromaDB thì ghi vào collection, còn không thì fallback sang list in-memory. Khi search, tôi cũng embed câu query rồi so sánh với các embedding đã lưu. Ở nhánh in-memory, tôi dùng dot product qua helper `_dot()` để chấm điểm similarity, sau đó sort giảm dần và trả về top-k kết quả theo một schema thống nhất.

**`search_with_filter` + `delete_document`** — approach:

> _Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?_
> Với `search_with_filter`, tôi filter theo metadata trước rồi mới chạy similarity search trên tập record còn lại, vì cách này giúp thu hẹp đúng không gian tìm kiếm và tránh trả về chunk sai domain. Nếu dùng ChromaDB thì filter được đẩy xuống ngay trong `query(..., where=metadata_filter)`, còn ở nhánh in-memory tôi duyệt qua `_store` để giữ lại những record có metadata khớp rồi mới gọi `_search_records()`. Với `delete_document`, tôi xóa theo `id` hoặc theo `metadata["doc_id"]` để có thể remove cả một document gốc lẫn toàn bộ các chunk con của nó.

### KnowledgeBaseAgent

**`answer`** — approach:

> _Viết 2-3 câu: prompt structure? Cách inject context?_
> Hàm `answer()` trước hết gọi `store.search(question, top_k)` để lấy các chunk liên quan nhất, sau đó ghép chúng thành một khối `Context` theo dạng `[Source: doc_id]` kèm nội dung chunk để giữ dấu vết nguồn. Prompt được dựng rất đơn giản: phần hướng dẫn "Answer the question using the context below", rồi đến `Context`, `Question`, và cuối cùng là `Answer:` để model sinh câu trả lời. Trong implementation hiện tại, `answer()` trả về chuỗi trả lời để giữ tương thích với test của lab, còn phần thông tin debug như `top_results` được tách sang `answer_with_details()`. Nếu không truyền `llm_fn`, agent tự dùng OpenAI `gpt-4o-mini` qua `OPENAI_API_KEY`.

### Test Results

```
# Paste output of: pytest tests/ -v
```
```
(venv) PS E:\VinAI\Ex7\Day-07-Lab-Data-Foundations> pytest tests/ -v
====================================== test session starts =======================================
platform win32 -- Python 3.14.3, pytest-9.0.3, pluggy-1.6.0 -- E:\VinAI\Ex7\Day-07-Lab-Data-Foundations\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: E:\VinAI\Ex7\Day-07-Lab-Data-Foundations
plugins: anyio-4.13.0
collected 42 items                                                                                

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED       [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED         [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED          [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED               [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED     [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED      [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED    [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                      [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED      [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                 [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED             [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                       [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED  [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED  [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                      [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED        [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED          [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED     [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED       [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED        [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                 [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED           [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED       [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED  [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED      [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED            [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED      [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

======================================= 42 passed in 2.01s =======================================
```
**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán    | Actual Score | Đúng? |
| ---- | ---------- | ---------- | ---------- | ------------ | ----- |
| 1    | The cat is sleeping on the sofa. | A kitten is resting on the couch. | high | 0.6712 | Yes |
| 2    | Python is a programming language. | Python is used to build software applications. | high | 0.8173 | Yes |
| 3    | Vietnamese mini savory pancakes use shrimp. | Banh khot includes fresh and dried shrimp. | high | 0.4939 | Yes |
| 4    | The stock market crashed yesterday. | A duck porridge recipe uses grilled onion and ginger. | low | -0.1284 | Yes |
| 5    | Orange fruit skin jam is a dessert. | The dessert should be cooled and stored in a jar. | high | 0.4562 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**

> Kết quả làm tôi thấy bất ngờ nhất là cặp 5 chỉ đạt `0.4562` dù hai câu rõ ràng cùng nói về món `Orange fruit skin jam` và cách bảo quản của nó. Điều này cho thấy embedding không chỉ nhìn vào cùng một chủ đề tổng quát mà còn rất nhạy với mức độ trùng khớp về chi tiết diễn đạt và mục tiêu thông tin giữa hai câu. Ngược lại, cặp 2 có score rất cao vì cả hai câu đều bám sát cùng khái niệm trung tâm là Python trong ngữ cảnh lập trình phần mềm.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | "What ingredients are needed for braised tofu with quail eggs?" |"100-200g fried tofu slices, 15-20 quail eggs, spring onion, shallot, salt, fish sauce, sugar, pepper, soy sauce, and Maggi's seasoning powder." |
| 2 | "How do you make the dipping sauce for grilled snails?"| "Mix salt, pepper, lemon juice, and sugar together. Serve with Vietnamese mint herb." |
| 3 | "What is the process for making duck porridge?" | "Boil duck with ginger and grilled onion. Roast sticky rice separately, then cook in broth until soft. Season and top with fried purple onion and pepper."|
| 4 | "Which dish is a dessert and how is it stored?" | "Orange Fruit Skin Jam (Mut Vo Cam). After cooking, wait to cool, then store in a jar and use day by day." |
| 5 | "Which dishes require shrimp as an ingredient?" | "Vietnamese Mini Savory Pancakes (Banh Khot) require fresh shrimps (10 pieces, boiled and cut in half) and dried shrimp (100g, ground well)." |

### Kết Quả Của Tôi

| #   | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
| --- | ----- | ------------------------------- | ----- | --------- | ---------------------- |
| 1   | What ingredients are needed for braised tofu with quail eggs? | `Braised_Tofu chunk#3`: phần ingredients của món đậu hũ kho trứng cút, chứa tofu, quail eggs, spring onion, shallot, fish sauce, sugar, pepper, soy sauce | 0.7287 | Yes | Agent trả lời đúng trọng tâm, nêu được danh sách nguyên liệu chính và bám sát context của tài liệu Braised Tofu |
| 2   | How do you make the dipping sauce for grilled snails? | `Grilled_Snails chunk#3`: bước pha hỗn hợp muối, đường, Maggi, dầu và ớt để phết/nêm cho ốc nướng | 0.7001 | Yes | Agent lấy đúng phần hướng dẫn liên quan đến sauce/seasoning của món ốc nướng và trả lời khá sát với gold answer |
| 3   | What is the process for making duck porridge? | `Duck_Porridge chunk#5`: bước làm cháo vịt sau khi luộc vịt, lấy gừng và hành nướng ra rồi cho gạo nếp vào nấu mềm | 0.7640 | Yes | Agent tóm tắt đúng quy trình chính: luộc vịt, nấu cháo với broth và hoàn thiện món ăn từ tài liệu Duck Porridge |
| 4   | Which dish is a dessert and how is it stored? | `Orange_Fruit_Skin_Jam chunk#5`: phần kết nói rõ đây là dessert, để nguội rồi cho vào hũ dùng dần | 0.5260 | Yes | Agent xác định đúng món tráng miệng là Orange Fruit Skin Jam và nêu đúng cách bảo quản trong hũ sau khi nguội |
| 5   | Which dishes require shrimp as an ingredient? | `Savory_Pancakes chunk#4`: phần nguyên liệu có dried shrimp; top-2 còn có `Savory_Pancakes chunk#3` chứa fresh shrimps | 0.6530 | Yes | Agent trả lời đúng rằng món Bánh Khọt cần shrimp, và context top đầu cũng hỗ trợ rõ cả dried shrimp lẫn fresh shrimps |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

> Tôi học được từ các thành viên khác cách chia nhỏ bài toán chunking theo từng bước rõ ràng thay vì chỉ code theo cảm tính. Ngoài ra, cách mọi người tổ chức metadata và suy nghĩ về Vector Store cũng giúp tôi hiểu rõ hơn vì sao retrieval không chỉ phụ thuộc vào embedding mà còn phụ thuộc vào cách chuẩn bị dữ liệu.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

> Qua phần demo của nhóm khác, tôi thấy việc trình bày thêm sơ đồ workflow giúp người nghe hiểu nhanh toàn bộ pipeline từ chunking, embedding, retrieval đến answer generation. Tôi cũng học được rằng một phần demo tốt không chỉ nằm ở code chạy đúng mà còn ở cách trực quan hóa kết quả và giải thích vì sao strategy đó hoạt động hiệu quả.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

> Nếu làm lại, tôi sẽ thử custom một chunking strategy bám sát cấu trúc recipe hơn, ví dụ tách riêng phần giới thiệu, nguyên liệu và từng bước nấu ăn. Tôi cũng muốn kết hợp thêm metadata chi tiết hơn như loại món, nguyên liệu chính hoặc giai đoạn chế biến để retrieval chính xác hơn khi gặp các câu hỏi cụ thể.

---

## Tự Đánh Giá

| Tiêu chí                    | Loại    | Điểm tự đánh giá |
| --------------------------- | ------- | ---------------- |
| Warm-up                     | Cá nhân | 5 / 5              |
| Document selection          | Nhóm    | 10 / 10             |
| Chunking strategy           | Nhóm    | 15 / 15             |
| My approach                 | Cá nhân | 10 / 10             |
| Similarity predictions      | Cá nhân | 5 / 5              |
| Results                     | Cá nhân | 10 / 10             |
| Core implementation (tests) | Cá nhân | 30 / 30             |
| Demo                        | Nhóm    | 0 / 5              |
| **Tổng**                    |         | **85 / 100**        |
