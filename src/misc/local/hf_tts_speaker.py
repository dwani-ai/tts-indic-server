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
'''
prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("indic_tts_out_3.wav", audio_arr, model.config.sampling_rate)

prompt2 = "ನಿಮ್ಮ ಇನ್‌ಪುಟ್ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ಸೇರಿಸಿ, ಸಾಮ್ರಾಜ್ಯದಲ್ಲಿ ಅತ್ಯುನ್ನತಸ್ಥಾನವನ್ನು ಗಳಿಸಿದೆ. ಇದರ ಕರ್ತೃವಿಗೆ ಆದಿಕವಿ ಆ ಎಂದೂ ಸಾರ್ಧಕವಾದ ವಿಶೇಷಣಗಳು ಬರಾಲಾ ಪ್ರಸಿದ್ಧಿಗೆ ಇಲ್ಲಿನ ಬಂದಿವೆ. ಶ್ರೇಯಸ್ಸಿನ ಸಂಪಾದನೆಗೂ ನಂಬಿಕೆಯಾಗಿದೆ. ಸೀತಾ, ಕಧಾವಸ್ತುವು ರಸಜ್ಞ್ಞ ತೆಯನ್ನು ಸಹೃದಯನ  ಭಾರತದ ಸಹಾಚು ಇವದ್ಮಡಿಸುವುದಲ್ಲದೆ, ಜನರೆಲ್ಲರ ಆತನ ನ್ಯಾಯವಾದ್‌ ವಾಲ್ಮೀಕಿಮಹರ್ಷಿಗಳಂ ಈ ಮಹಾಕಾವ್ಯದಲ್ಲಿ ಪ್ರಧಾನಪಾತ್ರಗಳಾದ"

prompt_input_ids = tokenizer(prompt2, return_tensors="pt").to(device)

generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("indic_tts_out_3_2.wav", audio_arr, model.config.sampling_rate)
'''
prompt3 = """ರಾಮಾಯಣವು ಭಾರತದ ಎರಡು ಮಹಾಕಾವ್ಯಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ. ಇದನ್ನು ಆದಿಕವಿ ವಾಲ್ಮೀಕಿ ರಚಿಸಿದರು ಎಂದು ನಂಬಲಾಗಿದೆ
ಈ ಮಹಾಕಾವ್ಯವು ಏಳು ಕಾಂಡಗಳಾಗಿ ವಿಭಜಿಸಲ್ಪಟ್ಟಿದೆ: ಬಾಲಕಾಂಡ, ಅಯೋಧ್ಯಾಕಾಂಡ, ಅರಣ್ಯಕಾಂಡ, ಕಿಷ್ಕಿಂಧಾಕಾಂಡ, ಸುಂದರಕಾಂಡ, ಯುದ್ಧಕಾಂಡ ಮತ್ತು ಉತ್ತರಕಾಂಡ.
ರಾಮಾಯಣದ ಕಥೆಯು ಮುಖ್ಯವಾಗಿ ಅಯೋಧ್ಯೆಯ ಸೂರ್ಯವಂಶದ ರಾಜಪುತ್ರ ರಾಮ, ಆತನ ಪತ್ನಿ ಸೀತೆ ಹಾಗೂ ಸೀತೆಯನ್ನು ಅಪಹರಿಸಿದ ರಾವಣನ ಸಂಹಾರದ ಕುರಿತಾಗಿದೆ.
ಬಾಲಕಾಂಡದಲ್ಲಿ ರಾಮನ ಜನನ, ಬಾಲ್ಯ ಮತ್ತು ಸೀತಾ ಸ್ವಯಂವರದ ಕಥೆ ಇದೆ.
ಅಯೋಧ್ಯಾಕಾಂಡದಲ್ಲಿ ಕೈಕೇಯಿಯ ವರದಿಂದ ರಾಮನಿಗೆ ವನವಾಸ ದೊರೆಯುತ್ತದೆ.
ಕಿಷ್ಕಿಂಧಾಕಾಂಡದಲ್ಲಿ ರಾಮ ಸುಗ್ರೀವ ಮತ್ತು ಹನುಮಂತನನ್ನು ಭೇಟಿಯಾಗುತ್ತಾನೆ
ರಾಮಾಯಣವು ಒಳ್ಳೆಯದರ ಜಯದ ಕಥೆಯಾಗಿದೆ. ಇದು ಧರ್ಮ, ಕರ್ತವ್ಯ ಮತ್ತು ನೀತಿಯ ಪಾಠಗಳನ್ನು ಕಲಿಸುತ್ತದೆ
ರಾಮನನ್ನು ಆದರ್ಶ ಪುತ್ರ, ಪತಿ ಮತ್ತು ರಾಜನೆಂದು ಪರಿಗಣಿಸಲಾಗಿದೆ
ರಾಮಾಯಣದ ಘಟನೆಗಳನ್ನು ಆಚರಿಸುವ ಎರಡು ಪ್ರಮುಖ ಭಾರತೀಯ ಹಬ್ಬಗಳೆಂದರೆ ದಸರಾ ಮತ್ತು ದೀಪಾವಳಿ
ರಾಮಾಯಣವು ಭಾರತೀಯ ಸಂಸ್ಕೃತಿ ಮತ್ತು ಸಾಹಿತ್ಯದ ಮೇಲೆ ಅಗಾಧ ಪ್ರಭಾವ ಬೀರಿದೆ.
"""

prompt_input_ids = tokenizer(prompt3, return_tensors="pt").to(device)

generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("indic_tts_out_3_3.wav", audio_arr, model.config.sampling_rate)

