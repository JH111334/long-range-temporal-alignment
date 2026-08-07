# ADR 0003: Phân Tầng Phần Cứng

## Quyết định

Core chạy thuần Python trên CPU. Máy tham chiếu có 16 GB RAM và NVIDIA GeForce RTX 2050 4 GB VRAM. Chỉ đưa model vào pipeline sau một resource probe có peak RAM/VRAM.

Qwen2.5-0.5B-Instruct là candidate đầu tiên cho tokenizer/frozen probe. LoRA chỉ chạy sau khi probe đạt; các model 3B–13B của paper thuộc Tier 3 và cần GPU ngoài máy cục bộ.

## Hệ quả

Repository không tự tải model, không cam kết model vừa VRAM và không dùng kết quả CPU fixture để thay cho thí nghiệm LLM.
