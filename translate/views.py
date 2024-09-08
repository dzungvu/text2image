from django.shortcuts import render

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def translate_text(request):
	translated_text = None
	if request.method == 'POST':
		text = request.POST.get('text')
		# Here you can add your translation logic
		translated_text = f"Translated: {text}"  # Placeholder for actual translation logic
	return render(request, 'translate.html', {'translated_text': translated_text})


model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# Check if MPS is available and use it, otherwise fall back to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

@csrf_exempt
def translate_api(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        inputs = [f'vi: {text}']
        if text:
            # Tokenize inputs and move to the appropriate device
            inputs = tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to(device)
            # Generate translations
            outputs = model.generate(inputs, max_length=512)
            translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return JsonResponse({'translated_text': translated_text}, status=200)
        return JsonResponse({'error': 'No text provided'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)