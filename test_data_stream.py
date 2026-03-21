from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

model = Qwen2AudioForConditionalGeneration.from_pretrained("cache/Qwen2-Audio-7B")
processor = AutoProcessor.from_pretrained("cache/Qwen2-Audio-7B")

prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=processor.feature_extractor.sampling_rate)  

inputs = processor(text=prompt, audio=audio, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_length=30)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]