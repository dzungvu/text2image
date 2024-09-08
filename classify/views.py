from django.shortcuts import render
from django.http import JsonResponse
import torch
import json
from transformers import pipeline

# Check if a GPU is available and set the device accordingly
device = None

# Load pre-trained zero-shot classification model
classifier = None


def classify_text(request):
    global device, classifier  # Declare them as global to modify the module-level variables
    if request.method == 'POST':
        if device is None or classifier is None:
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
        
        input_text = request.POST.get('text')
        selected_options = json.loads(request.POST.get('selected_options'))
        
        candidate_labels = [option['label'] for option in selected_options]
        classification_result = classifier(input_text, candidate_labels)
        
        # Extract the label with the highest score
        highest_score_index = classification_result['scores'].index(max(classification_result['scores']))
        highest_score_label = classification_result['labels'][highest_score_index]
        highest_score_id = selected_options[highest_score_index]['id']
        
        return JsonResponse({'result': highest_score_label, 'id': highest_score_id})
    return render(request, 'classify.html')