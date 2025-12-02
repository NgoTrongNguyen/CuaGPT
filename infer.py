import torch
from Model import GPT
from Data import CharTokenizer

# Thư viện tạo giao diện

import torch, tkinter as tk
import random
from tkinter import scrolledtext
from tkinter import font

# cấu hình
block_size = 256
d_model = 256
n_layers = 4
n_heads = 4
dropout = 0.07
batch_size = 32
epochs = 10
lr = 3e-4

def LoadCheckpoint(path, block_size, d_model, n_layers, n_heads, dropout):
    checkpoint = torch.load(path, map_location = "cpu")
    stoi = checkpoint["tok"]
    # tạo tokenizer giả từ stoi
    class T(CharTokenizer):
        def __init__(self):
            self.stoi = stoi
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.char_size = len(self.stoi)
    tok = T()
    model = GPT(tok.char_size, d_model, n_layers, n_heads, dropout, block_size)
    model.load_state_dict(checkpoint["model"])
    return model, tok

def send_message():
    message = entry.get()
    prompt = message
    if message:
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "Bạn: " + message + "\n", "to")
        chat_window.config(state=tk.DISABLED)
        entry.delete(0, tk.END)
        chat_window.update_idletasks()
    
    ids = torch.tensor([tok.encode(prompt)], device=device)
    gen = model.generate(ids, max_new_tokens=450, temperature=0.8, top_k=50)
    if message:
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "ChatBot: " + f"{tok.decode(gen[0].tolist())}" + "\n", "to")
        chat_window.config(state=tk.DISABLED)




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    l_checkpoint = input("Select:" )
    model, tok = LoadCheckpoint(l_checkpoint, block_size, d_model, n_layers, n_heads, dropout)
    model.to(device).eval()
        

    #Tạo cửa sổ
    root = tk.Tk()
    root.title("Chatbox")
    root.geometry("900x500")

    #Tạo scroll
    chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED)
    chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    #Tạo frame
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10, fill=tk.X)

    #Tạo entry
    entry = tk.Entry(frame)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

    #Tạo button
    send_button = tk.Button(frame, text="Gửi", command=send_message)
    send_button.pack(side=tk.RIGHT)

    #Tạo bind keyboard
    entry.bind("<Return>", lambda event: send_message())

    #Tạo font
    Font_mau = font.Font(family = "Arial", size = 14, weight = "bold")
    chat_window.tag_configure("to", font=Font_mau)

    #Khởi tạo
    root.mainloop()


 
