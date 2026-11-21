# tokenizer/train_tokenizer.py
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

SPECIAL_TOKENS = [
    "<|pad|>", "<|bos|>", "<|eos|>",
    "<|sys|>", "<|usr|>", "<|ast|>",
    "<|thought|>", "</|thought|>",
    "<|env_ish|>",        # بيئة iSH (Alpine x86 على iOS - حزم apk و ash shell)
    "<|env_segfault|>",   # خوادم Segfault.net المؤقتة (SSH Root + Tor + /sec)
    "<|env_kali|>",       # Kali Linux (أدوات الاختبار السيبراني الكاملة)
    "<|env_ubuntu|>",     # Ubuntu Server/Desktop (إدارة الأنظمة والخدمات)
    "<|py_exec|>",        # تنفيذ سكربت بايثون
    "<|web_search|>",     # بحث الإنترنت العادي
    "<|tor_search|>",     # بحث الشبكة المظلمة (.onion)
    "<|obs|>",            # مخرجات التنفيذ أو البحث (Stdout/Stderr)
]

def build_custom_tokenizer(corpus_files: list[str], output_dir: str = "tokenizer/out", vocab_size: int = 65536):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"[*] تدريب المُرمّز من الصفر على {len(corpus_files)} ملف بيانات...")
    tokenizer.train(files=corpus_files, trainer=trainer)

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    save_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(save_path)
    print(f"[+] تم حفظ المُرمّز بنجاح في: {save_path}")

if __name__ == "__main__":
    # مرر مسارات ملفاتك النصية المجمعة من الزاحف
    build_custom_tokenizer(["data/raw/corpus_all.txt"])
