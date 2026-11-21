# data/prepare_shards.py
import numpy as np
from tokenizers import Tokenizer

def tokenize_to_bin(txt_path="data/raw/corpus_all.txt", tok_path="tokenizer/out/tokenizer.json", out_bin="data/train.bin"):
    tok = Tokenizer.from_file(tok_path)
    all_ids = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_ids.extend(tok.encode(line).ids)
    arr = np.array(all_ids, dtype=np.uint32)
    arr.tofile(out_bin)
    print(f"[+] تم تحويل {len(arr):,} رمز (Token) وحفظها في {out_bin}")

if __name__ == "__main__":
    tokenize_to_bin()
