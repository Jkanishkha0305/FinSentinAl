# Text Summarization

```python
model = "facebook/bart-large-cnn"
llm = HuggingFaceHub(
    repo_id=model,
    model_kwargs={"temperature": 0.9},
    huggingfacehub_api_token=""
)

from langchain import LLMChain, PromptTemplate

prompt = PromptTemplate(
    input_variables=["text"],
    template="Explain the text{text}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
```


## Sentiment Analysis
```python
from transformers import pipeline

sentiment_task = pipeline(
    "sentiment-analysis",
    model="yiyanghkust/finbert-tone",
    tokenizer="yiyanghkust/finbert-tone"
)
```

## NER: Name Entity Recognition

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
classification = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-ner")

token_classification = pipeline('ner', tokenizer=tokenizer, model=classification)
t = token_classification(df['summary'][0])
```
