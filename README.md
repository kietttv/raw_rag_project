```
raw_rag_project/
├── .env                      # Biến môi trường (API Key của Gemini, DB URI,...)
├── Dockerfile                # Cấu hình đóng gói container
├── requirements.txt # Quản lý thư viện
└── src/                      # Mã nguồn chính
    ├── __init__.py
    ├── main.py               # Điểm khởi chạy ứng dụng (FastAPI app, CLI, hoặc Streamlit)
    ├── config.py             # Quản lý cấu hình hệ thống (dùng Pydantic BaseSettings)
    │
    ├── api/                  # Lớp giao diện API (Controllers / Routers)
    │   ├── __init__.py
    │   ├── v1/
    │   │   ├── endpoints/
    │   │   │   ├── chat.py   # API endpoint xử lý câu hỏi RAG
    │   │   │   └── ingest.py # API endpoint tải lên và xử lý tài liệu mới
    │   │   └── router.py
    │
    ├── core/                 # Lô-gic cốt lõi của RAG (Pipelines)
    │   ├── __init__.py
    │   ├── ingestion.py      # Xử lý tài liệu: Loader, Advanced Chunking, Embedding
    │   ├── retriever.py      # Tìm kiếm: Vector search, Hybrid search (BM25)
    │   ├── reranker.py       # Sắp xếp lại kết quả (Cross-Encoder Re-ranker)
    │   └── generator.py      # Giao tiếp với Gemini, Context Caching, Prompt Assembly
    │
    ├── services/             # Kết nối các dịch vụ ngoài (External Clients)
    │   ├── __init__.py
    │   ├── vector_db.py      # Khởi tạo kết nối Vector Database (Chroma/Pinecone/Milvus)
    │   ├── cache_service.py  # Quản lý Redis Semantic Cache
    │   └── tracing.py        # Tích hợp LLMOps (Langfuse / Arize Phoenix)
    │
    ├── guardrails/           # Lớp bảo mật và kiểm duyệt nội dung
    │   ├── __init__.py
    │   ├── input_guard.py    # Kiểm tra prompt injection, từ khóa độc hại
    │   └── output_guard.py   # Kiểm tra hallucination, ẩn thông tin nhạy cảm (PII)
    │
    ├── models/               # Định nghĩa kiểu dữ liệu (Pydantic Schemas)
    │   ├── __init__.py
    │   └── schemas.py        # Khung request/response cho API
    │
    └── utils/                # Các hàm tiện ích chung
        ├── __init__.py
        └── logger.py         # Cấu hình ghi log chuẩn (Structured Logging)
```