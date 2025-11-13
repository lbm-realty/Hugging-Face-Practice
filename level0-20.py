from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer
from datatrove.pipeline.readers import ParquetReader
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter

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
print(generated_ans_qa)


# limit determines how many documents will be streamed (remove for all)
# this will fetch the Portuguese data
data_reader = ParquetReader("hf://datasets/HuggingFaceFW/finewiki/data/ptwiki", limit=1) 
# for document in data_reader():
    # do something with document
    # print(document)
    # return

# OR for a processing pipeline:
# pipeline_exec = LocalPipelineExecutor(
#     pipeline=[
#         ParquetReader("hf://datasets/HuggingFaceFW/finewiki/data/ptwiki", limit=3),
#         LambdaFilter(lambda doc: "hugging" in doc.text),
#         JsonlWriter("some-output-path")
#     ],
#     tasks=10
# )
# pipeline_exec.run()