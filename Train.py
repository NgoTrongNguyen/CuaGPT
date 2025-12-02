import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from Data import LoadCorpus, CharTokenizer, MakeDataset
from Model import GPT
#from reference import box


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# cấu hình
block_size = 256
d_model = 256
n_layers = 4
n_heads = 4
dropout = 0.08
batch_size = 32
epochs = 15
lr = 3e-4

#Dữ liệu
text = LoadCorpus("C:\VSCode Project\LLMs\corpus.txt")
tok = CharTokenizer(text)
X, Y = MakeDataset(text, block_size=block_size)
dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#Mô hình
model = GPT(tok.char_size, d_model, n_layers, n_heads, dropout, block_size).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

#Huấn luyện
for epoch in range(epochs):
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    for xb, yb in pbar:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)              # B, T, V
        loss = loss_fn(logits.view(-1, tok.char_size), yb.view(-1))
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        pbar.set_postfix(loss=float(loss.item()))
    # lưu checkpoint mỗi epoch
    torch.save({
        "model": model.state_dict(),
        "tok": tok.stoi
    }, f"epoch{epoch+1}.pt")

# Sinh văn bản
model.eval()
start = "Giới thiệu về one punch man"
start_ids = torch.tensor([tok.encode(start)], device=device)
gen = model.generate(start_ids, max_new_tokens=300, temperature=0.8, top_k=50)
print(tok.decode(gen[0].tolist()))