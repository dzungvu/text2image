from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time


print(f'{time.time()} - start load model')
model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

inputs = [
    "vi: Trương Tấn Sang sinh ngày 21 tháng 1 năm 1949 tại huyện Châu Thành, tỉnh Hậu Giang, Việt Nam. Ông là một chính trị gia người Việt Nam, từng giữ chức Tổng Bí thư Đảng Cộng sản Việt Nam, và từng là Chủ tịch nước nước Cộng hòa xã hội chủ nghĩa Việt Nam.",
    ]

# Check if MPS is available and use it, otherwise fall back to CPU
print(f'{time.time()}')
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"{time.time()} - Using device: {device}")
model.to(device)

# Tokenize inputs and move to the appropriate device
inputs = tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to(device)

# Generate translations
outputs = model.generate(inputs, max_length=512)
print(f"{time.time()} - result")
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])