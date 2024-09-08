from transformers import pipeline

# Load XLM-R model for multilingual zero-shot classification
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

# Vietnamese text to classify
text = "Ai là người kéo cờ Tổ quốc trong lễ Tuyên ngôn Độc lập?"

# Define candidate labels in Vietnamese or English
candidate_labels = ["lịch sử", "địa lý", "toán học", "khoa học", "văn học"]

# Perform zero-shot classification
result = classifier(text, candidate_labels)

# Output the result
print(result)
