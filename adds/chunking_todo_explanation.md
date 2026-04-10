# Chunking TODO Explanation

Mình đã hoàn thành các `TODO` trong `src/chunking.py` ở 4 phần chính:

## 1. `SentenceChunker.chunk`

- Dùng `re.split(r"(?<=[.!?])(?:\\s+|\\n+)")` để tách câu theo dấu kết thúc câu như `.`, `!`, `?`.
- Mỗi câu được `strip()` để bỏ khoảng trắng dư.
- Sau đó gom lại theo từng nhóm `max_sentences_per_chunk` câu để tạo ra từng chunk.

## 2. `RecursiveChunker.chunk` và `_split`

- `chunk()` chỉ làm nhiệm vụ kiểm tra đầu vào rỗng và gọi helper `_split()`.
- `_split()` xử lý theo thứ tự separator ưu tiên: `\n\n`, `\n`, `. `, `" "`, rồi cuối cùng là `""`.
- Nếu đoạn hiện tại đã ngắn hơn `chunk_size` thì trả về luôn.
- Nếu không còn separator phù hợp, hoặc separator là chuỗi rỗng, hàm fallback sang cắt trực tiếp theo từng đoạn dài `chunk_size`.
- Khi tách được nhiều mảnh, hàm sẽ cố gắng ghép các mảnh lại thành chunk lớn nhất có thể nhưng không vượt `chunk_size`.

## 3. `compute_similarity`

- Implement đúng công thức cosine similarity:

```python
dot(a, b) / (||a|| * ||b||)
```

- Nếu một trong hai vector có độ lớn bằng 0 thì trả về `0.0` để tránh chia cho 0.

## 4. `ChunkingStrategyComparator.compare`

- Gọi 3 strategy có sẵn:
  - `FixedSizeChunker`
  - `SentenceChunker`
  - `RecursiveChunker`
- Với mỗi strategy, mình trả về:
  - `count`: số chunk
  - `avg_length`: độ dài trung bình
  - `chunks`: danh sách chunk thực tế

## Lý do chọn cách này

- Code ngắn, dễ đọc, đúng với mục tiêu bài lab.
- Đủ ổn cho test hiện tại.
- Có fallback rõ ràng khi text không có separator phù hợp.
