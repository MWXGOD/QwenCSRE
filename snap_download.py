from huggingface_hub import snapshot_download


# snapshot_download(repo_id="Qwen/Qwen-Audio", local_dir = "cache/Qwen-Audio", ignore_patterns=["tf_model.h5", "flax_model.msgpack", "*.safetensors"], resume_download=True)

snapshot_download(repo_id="Qwen/Qwen2-Audio-7B", local_dir = "cache/Qwen2-Audio-7B", ignore_patterns=["tf_model.h5", "flax_model.msgpack", "*.safetensors"], resume_download=True)
