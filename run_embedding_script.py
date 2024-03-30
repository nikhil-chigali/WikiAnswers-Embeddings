from lightning_sdk import Studio, Machine

embedding_model_mappings = {
    384: "all-MiniLM-L6-v2",
    512: "distiluse-base-multilingual-cased-v1",
    768: "all-distilroberta-v1",
    1024: "llmrails/ember-v1",
}

studio = Studio()
studio.install_plugin("jobs")
job_plugin = studio.installed_plugins["jobs"]


def run():
    for embed_dim, model_name in embedding_model_mappings.items():
        job = job_plugin.create_job(
            name=f"Generate embeddings for {model_name} with embedding size: {embed_dim}",
            command=f"python src/generate_embeddings.py --model_name {model_name} --embed_dim {embed_dim}",
            machine=Machine.A10G,
        )
        job.run()


if __name__ == "__main__":
    run()
