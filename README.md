# black
cybercore_zero/
├── config.py                  # إعدادات النموذج (300M للتجربة / 1.5B - 7B للإنتاج)
├── tokenizer/
│   └── train_tokenizer.py     # بناء مُرمّز BPE من الصفر مخصص للبايثون والطرفيات
├── model/
│   └── architecture.py        # المعمارية الكاملة من الصفر (PyTorch الصافي)
├── data/
│   ├── harvester.py           # زاحف الإنترنت العادي + شبكة Tor (.onion)
│   └── prepare_shards.py      # تنظيف وترميز وتقطيع البيانات لملفات ثنائية (.bin)
├── train/
│   └── pretrain.py            # محرك التدريب الموزع من الصفر (DDP + BF16 + FlashAttn)
├── agent/
│   └── self_evolve.py         # نظام الوكيل، التنفيذ عبر البيئات الـ 4، والتعلم الذاتي المستمر
└── export/
    └── export_gguf.py         # تصدير الأوزان الخام وتحويلها وضغطها إلى GGUF تلقائياً
