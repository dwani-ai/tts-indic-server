
description = "Anu speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality"

Trigger Chunks

 curl -X 'POST'   'http://localhost:7860/v1/audio/speech'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "input": "ನಿಮ್ಮ ಇನ್‌ಪುಟ್ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ಸೇರಿಸಿ, ಸಾಮ್ರಾಜ್ಯದಲ್ಲಿ ಅತ್ಯುನ್ನತಸ್ಥಾನವನ್ನು ಗಳಿಸಿದೆ. ಇದರ ಕರ್ತೃವಿಗೆ ಆದಿಕವಿ ಆ ಎಂದೂ ಸಾರ್ಧಕವಾದ ವಿಶೇಷಣಗಳು ಬರಾಲಾ ಪ್ರಸಿದ್ಧಿಗೆ ಇಲ್ಲಿನ ಬಂದಿವೆ. ಶ್ರೇಯಸ್ಸಿನ ಸಂಪಾದನೆಗೂ ನಂಬಿಕೆಯಾಗಿದೆ. ಸೀತಾ, ಕಧಾವಸ್ತುವು ರಸಜ್ಞ್ಞ ತೆಯನ್ನು ಸಹೃದಯನ  ಭಾರತದ ಸಹಾಚು ಇವದ್ಮಡಿಸುವುದಲ್ಲದೆ, ಜನರೆಲ್ಲರ ಆತನ ನ್ಯಾಯವಾದ್‌ ವಾಲ್ಮೀಕಿಮಹರ್ಷಿಗಳಂ ಈ ಮಹಾಕಾವ್ಯದಲ್ಲಿ ಪ್ರಧಾನಪಾತ್ರಗಳಾದ ಶ್",  "voice": "Anu speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality",
  "model": "ai4bharat/indic-parler-tts",
  "response_format": "mp3",
  "speed": 1
}' -o test.mp3


Single run

 curl -X 'POST'   'http://localhost:7860/v1/audio/speech'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "input": "ನಿಮ್ಮ ಇನ್‌ಪುಟ್ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ಸೇರಿಸಿ, ಸಾಮ್ರಾಜ್ಯದಲ್ಲಿ ಅತ್ಯುನ್ನತಸ್ಥಾನವನ್ನು ಗಳಿಸಿದೆ. ಇದರ ಕರ್ತೃವಿಗೆ ಆದಿಕವಿ",  "voice": "Anu speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality",
  "model": "ai4bharat/indic-parler-tts",
  "response_format": "mp3",
  "speed": 1
}' -o test.mp3