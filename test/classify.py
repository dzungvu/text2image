from transformers import pipeline
import torch

# Check if a GPU is available and set the device accordingly
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load pre-trained zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# Text to classify
text = "triangle, circle and square are basic shapes in geometry"

# Define candidate labels (these are the specific fields you're interested in)
candidate_labels = ["history", "geography", "math", "science", "literature", "social"]

# Perform zero-shot classification
result = classifier(text, candidate_labels)

# Output the result
print(result)