# Large Language Models with Retrieval Augmented Agents among Collaborators for Medical Reasoning


We build collaborative LLM agents that combine intrinsic LLM knowledge with RAG-sourced medical data to produce consistent, reliable answers through collaborative, training-free reasoning.


## Requirements

Install all required python dependencies:

```
pip install -r requirements.txt
```

## Data

We evaluate our framework on two benchmark datasets MedQA and PubMedQA. Both the dataset is located in datasets folder.


## Implementations
Input your own openai api key in env variable OPENAI_API_KEY which will be used by OpenAI client. From the root folder, run the below to execute code.
```
python run.py
```

