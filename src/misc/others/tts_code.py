import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

prompt = "ನಿಮ್ಮ ಇನ್‌ಪುಟ್ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ಸೇರಿಸಿ, ಸಾಮ್ರಾಜ್ಯದಲ್ಲಿ ಅತ್ಯುನ್ನತಸ್ಥಾನವನ್ನು ಗಳಿಸಿದೆ."
description = "Anu speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality"

description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)


prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("indic_tts_out_1.wav", audio_arr, model.config.sampling_rate)

