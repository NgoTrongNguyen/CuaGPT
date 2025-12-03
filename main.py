from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from infer import LoadCheckpoint
import torch
from Model import GPT
from Data import CharTokenizer

# Khởi tạo ứng dụng FastAPI
app = FastAPI()
# cấu hình
block_size = 256
d_model = 256
n_layers = 10
n_heads = 4
dropout = 0.1
batch_size = 32
epochs = 10
lr = 3e-4

# Cho phép FE gửi request từ bất kỳ origin nào
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ["http://127.0.0.1:5500"] nếu bạn serve FE ở port này
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, OPTIONS, PUT, DELETE...
    allow_headers=["*"],
)

# Load mô hình và tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
l_checkpoint = "epoch53.pt"
model, tok = LoadCheckpoint(l_checkpoint, block_size, d_model, n_layers, n_heads, dropout)
model.to(device).eval()

# Định nghĩa cấu trúc dữ liệu nhận từ frontend
class InputData(BaseModel):
    text: str


@app.post("/process")
def process_signal(data: InputData):
    prompt = data.text
    ids = torch.tensor([tok.encode(prompt)], device=device)
    gen = model.generate(ids, max_new_tokens=450, temperature=0.8, top_k=50)
    processed_text = tok.decode(gen[0].tolist())
    # Trả kết quả về cho frontend
    return {"result": processed_text}