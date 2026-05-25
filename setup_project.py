import os

# Danh sách các thư mục cần tạo
folders = [
    "configs", 
    "data/raw", "data/processed", "data/embeddings",
    "notebooks", 
    "src/data_pipeline", "src/features", "src/models", 
    "src/training", "src/evaluation", 
    "pipelines"
]

# Danh sách các file cần tạo
files = [
    "configs/model_config.yaml", "configs/data_config.yaml",
    "src/__init__.py", "src/data_pipeline/crawler.py", "src/data_pipeline/stock_data.py", 
    "src/data_pipeline/preprocessor.py", "src/features/bert_extractor.py", 
    "src/features/graph_builder.py", "src/models/bert_module.py", 
    "src/models/gnn_module.py", "src/models/lstm_module.py", "src/models/fusion_model.py",
    "src/training/trainer.py", "src/training/custom_loss.py",
    "src/evaluation/metrics.py", "src/evaluation/backtest.py",
    "pipelines/run_pipeline.py", "pipelines/infer.py",
    ".gitignore", "requirements.txt", "README.md"
]

# 1. Tạo folder và thêm file .gitkeep để Github nhận diện folder rỗng
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, ".gitkeep"), "w") as f:
        pass

# 2. Tạo các file code trắng
for file in files:
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass

print("Đã tạo xong toàn bộ framework!")