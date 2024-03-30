This script generates embeddings of various sizes (`368, 512, 768, 1024`) for the [WikiAnswers](https://huggingface.co/datasets/embedding-data/WikiAnswers) dataset and publishes them to HuggingFace.

The models used for generating the embeddings are,
```python
{
    384: "all-MiniLM-L6-v2",
    512: "distiluse-base-multilingual-cased-v1",
    768: "all-distilroberta-v1",
    1024: "llmrails/ember-v1",
}
```

All the generated `embedding-datasets` can be found [here](https://huggingface.co/nikhilchigali).

