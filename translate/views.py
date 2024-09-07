from django.shortcuts import render

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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

@csrf_exempt
def translate_api(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        text_in_array = [text]
        if text:
             # Perform the translation
            inputs = tokenizer([text], return_tensors="pt", padding=True)
            outputs = model.generate(inputs.input_ids, max_length=512)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return JsonResponse({'translated_text': translated_text}, status=200)
        return JsonResponse({'error': 'No text provided'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)