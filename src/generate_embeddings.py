from argparse import ArgumentParser
from functools import partial
from logging import getLogger
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

logger = getLogger(__name__)


def run(model_name: str, embed_dim: int) -> dict:
    """
    Generate embeddings for the given model name and embedding dimension.
    Args:
        model_name: The name of the model to use for generating embeddings.
        embed_dim: The dimension of the embeddings to generate.
    Returns:
        A dictionary containing the embeddings for the given model and embedding dimension.
    """
    dataset = load_dataset(
        "nikhilchigali/wikianswers_small", cache_dir="data/wikianswers_small/"
    )
    logger.info(f"Loaded dataset with {len(dataset)} examples")

    model = SentenceTransformer(model_name)
    logger.info(f"Loaded model: {model_name}")

    def embed(model, item):
        return {
            "sentence": item["sentence"],
            "cluster": item["cluster"],
            f"embedding_{embed_dim}": model.encode(item["sentence"]),
        }

    embed_partial = partial(embed, model)
    logger.info(f"Generating embeddings for {len(dataset)} examples...")

    return dataset.map(
        embed_partial,
        batched=True,
        batch_size=64,
        num_proc=4,
    )


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--model_name", type=str, required=True)
    arg_parser.add_argument("--embed_dim", type=int, required=True)
    args = arg_parser.parse_args()
    logger.info(
        f"Generating embeddings for model: {args.model_name} with embedding dimension: {args.embed_dim}"
    )
    embedded_dataset = run(args.model_name, args.embed_dim)
    logger.info(f"Saving embeddings to disk")
    embedded_dataset.save_to_disk(f"data/wikianswers_embeddings_{args.embed_dim}")
