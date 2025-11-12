from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer

classifier = pipeline("sentiment-analysis")
result = classifier("Lenovo sucks")

dataset = load_dataset("ag_news")
# print(dataset["train"][0])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokens = tokenizer("I love Hugging Face!")
# print(tokens)

generator = pipeline("text-generation")
generated_answer = generator("I am failing my classes because ")
# print(generated_answer)

qa = pipeline("question-answering")
generated_ans_qa = qa({
    "question": "Where was Lebron James born", 
    "context": "Lebron James is an Indian footballer who scored 30 goals in one season"
    })
# print(generated_ans_qa)