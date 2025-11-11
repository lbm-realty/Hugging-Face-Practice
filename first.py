from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("Hugging Face makes machine learning fun!")
print(result)
