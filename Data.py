#Tokenizer ký tự (chuyển sang ID để ánh xạ)

import numpy as np
import os
import re
from pathlib import Path


def LoadCorpus(path = "corpus.txt"):
    text = Path(path).read_text(encoding="utf-8")

    #Xóa ký tự đặc biệt 
    clean = re.sub(r"[!@#$%&*()]", "", text)

    return clean

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text))) #Tách các ký tự
        self.stoi = {ch: i for i, ch in enumerate(chars)} #Mã hóa ký tự -> số
        self.itos = {i:ch for ch,i in self.stoi.items()} #Giải mã số -> ký tụ
        self.char_size = len(self.stoi) #Số ký tự sử dụng trong câu

    def encode(self, s): #Hàm mã hóa
        return [self.stoi[ch] for ch in s if ch in self.stoi]
    
    def decode(self, ids): #Hàm giải mã
        return "".join(self.itos[i] for i in ids)
    
def MakeDataset(text, block_size=256):
    ids = np.array(CharTokenizer(text).encode(text),dtype=np.int64) #Mã hóa thành chuỗi 64-bits

    # tạo cặp (input, target) dạng dịch phải 1 ký tự

    # Dùng để huấn luyện dự đoán ký tự tiếp theo
    X, Y= [], []
    for i in range(0, len(ids) - block_size):
        x = ids[i:i+block_size]
        y = ids[i+1:i+block_size+1]
        X.append(x)
        Y.append(y)
    return np.stack(X), np.stack(Y)
