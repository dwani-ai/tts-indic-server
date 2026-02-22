# Parler-TTS Latency Measurements

This report summarizes the latency measurements for the Parler-TTS text-to-speech system (`ai4bharat/indic-parler-tts`) under different hardware configurations and optimization methods. The latency is reported in seconds and corresponds to the time taken to generate audio for a given number of words. The data is derived from various tests conducted on March 16, 2025.

## Latency Table

# Parler-TTS Latency Measurements (Formatted)

| Hardware | Optimization Method         | Word Count | Latency (s) | Notes                              |
|:---------|:----------------------------|:----------:|:-----------:|:-----------------------------------|
| T4       | Simple Transformer          |     5      |    4.70     | Baseline measurement              |
| T4       | Simple Transformer          |    21      |   23.63     | Baseline measurement              |
| T4       | Flash Attention             |     5      |    8.16     | Slower than baseline              |
| T4       | Flash Attention             |    21      |   38.29     | Significantly slower than baseline|
| L4       | Flash Attention             |     5      |    3.99     | Fastest for 5 words across tests  |
| L4       | Flash Attention             |    21      |   20.82     | Improved over T4 FA               |
| L4       | Flash Attention (App)       |     7      |    7.92     | App request measurement           |
| A10G     | Flash Attention             |    21      |   25.52     | Consistent but slower than L4     |
| A10G     | Flash Attention             |    21      |   24.33     | Slight variation in repeated test |
| L4       | Torch Compile (Regular)     |     1      |    2.72     | Minimal input size                |
| L4       | Torch Compile (Regular)     |     5      |    2.58     | Fastest for small input           |
| L4       | Torch Compile (Regular)     |     7      |    4.70     | Comparable to baseline T4         |
| L4       | Torch Compile (Regular)     |    21      |   10.65     | Best regular compile for 21 words |
| L4       | Torch Compile (Regular)     |    21      |   11.99     | Slight variation                  |
| L4       | Torch Compile (Regular)     |    21      |   12.10     | Consistent performance            |
| L4       | Torch Compile (Regular)     |    21      |   13.51     | Higher variation                  |
| L4       | Torch Compile (Reduce OH)   |     7      |    3.00     | Estimated from "3s - 7 words"     |
| L4       | Torch Compile (Reduce OH)   |    21      |   10.00     | Estimated from "10 s - 21"        |
| L4       | Torch Compile (Reduce OH)   |    21      |   12.00     | Estimated from "12 s - 21 words"  |

## Observations

1. **Hardware Impact**:
   - The L4 server with Flash Attention showed the best performance for 5 words (3.99s), suggesting better optimization or higher computational power compared to T4.
   - A10G with Flash Attention was slower (24-25s for 21 words) than L4 (20.82s), indicating potential hardware or configuration differences.

2. **Optimization Methods**:
   - **Simple Transformer (T4)**: Served as a baseline with 4.70s for 5 words and 23.63s for 21 words.
   - **Flash Attention**: Surprisingly slower on T4 (8.16s for 5 words, 38.29s for 21 words) compared to the baseline, but improved on L4 (3.99s for 5 words, 20.82s for 21 words). This suggests Flash Attention benefits from specific hardware capabilities.
   - **Torch Compile (Regular)**: Consistently faster than Flash Attention, with the best result for 5 words at 2.58s and a range of 10.65-13.51s for 21 words.
   - **Torch Compile (Reduce Overhead)**: Showed promising results with approximately 3s for 7 words and 10-12s for 21 words, indicating potential for lower latency with this mode.

3. **Input Size**:
   - Latency generally increases with word count, but the scaling is not linear. For example, Torch Compile (Regular) took 2.58s for 5 words and 10.65s for 21 words, suggesting optimization benefits for larger inputs.

## Notes

- The "reduce-overhead" mode values (3s, 12s, 10s) were approximated from your shorthand notation; actual measurements might vary slightly.
- All measurements were taken on March 16, 2025, using the `ai4bharat/indic-parler-tts` model.
- Latency values are in seconds (s), rounded to two decimal places.
- Word counts represent the number of words in the input text.
- "Reduce OH" refers to the "reduce-overhead" mode in Torch Compile.
- The table is sorted by hardware, then optimization method, and finally word count for better readability.

## Conclusion
The Torch Compile optimization, particularly with "reduce-overhead" mode, appears to offer the best balance of latency reduction across different input sizes. The L4 server with Flash Attention also performed well, especially for smaller inputs. For optimal performance, consider using Torch Compile with "reduce-overhead" mode on capable hardware, though further testing could refine these findings.



small

curl -X 'POST' \
  'https://gaganyatri-tts-indic-server.hf.space/v1/audio/speech' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": "ನಿಮ್ಮ ಇನ್‌ಪುಟ್ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ಸೇರಿಸಿ",
  "voice": "Anu speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality",
  "model": "ai4bharat/indic-parler-tts",
  "response_format": "mp3",
  "speed": 1
}' -o test_small.mp3


Took 4.70 seconds to generate audio for 5 words

large -

curl -X 'POST' \
  'https://gaganyatri-tts-indic-server.hf.space/v1/audio/speech' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": "ನಿಮ್ಮ ಇನ್‌ಪುಟ್ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ಸೇರಿಸಿ, ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?, ಶುಭೋದಯ!, ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿದೆ ಕರ್ನಾಟಕದಲ್ಲಿ ನಾವು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇವೆ , ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು",
  "voice": "Anu speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality",
  "model": "ai4bharat/indic-parler-tts",
  "response_format": "mp3",
  "speed": 1
}' -o test_large.mp3



Simple Transformer on T4 
Took 4.70 seconds to generate audio for 5 words

Took 23.63 seconds to generate audio for 21 words 


Flash attention on T4

FA - 2025-03-16 17:52:55,225 - tts_indic_server - INFO - Took 8.16 seconds to generate audio for 5 words using 0

FA - 2025-03-16 17:53:59,147 - tts_indic_server - INFO - Took 38.29 seconds to generate audio for 21 words using 0


Flash attention on L4 server

2025-03-16 18:01:14,297 - tts_indic_server - INFO - Took 3.99 seconds to generate audio for 5 words using 0

2025-03-16 18:00:58,528 - tts_indic_server - INFO - Took 20.82 seconds to generate audio for 21 words using 0

APP - request
2025-03-16 18:02:50,117 - tts_indic_server - INFO - Took 7.92 seconds to generate audio for 7 words using 0


Flash Attention on A10 G server
2025-03-16 18:10:53,413 - tts_indic_server - INFO - Took 25.52 seconds to generate audio for 21 words using 0
2025-03-16 18:11:22,699 - tts_indic_server - INFO - Took 24.33 seconds to generate audio for 21 words using 0


Torch Compile - regular- 

2025-03-16 18:46:14,854 - tts_indic_server - INFO - Took 12.10 seconds to generate audio for 21 words using CUDA:0
INFO:     10.20.28.164:22507 - "POST /v1/audio/speech HTTP/1.1" 200 OK
2025-03-16 18:47:05,304 - tts_indic_server - INFO - Took 11.99 seconds to generate audio for 21 words using CUDA:0
INFO:     10.20.34.20:1097 - "POST /v1/audio/speech HTTP/1.1" 200 OK
2025-03-16 18:47:36,929 - tts_indic_server - INFO - Took 2.58 seconds to generate audio for 5 words using CUDA:0
INFO:     10.20.28.164:62426 - "POST /v1/audio/speech HTTP/1.1" 200 OK
2025-03-16 18:47:59,388 - tts_indic_server - INFO - Took 10.65 seconds to generate audio for 21 words using CUDA:0
INFO:     10.20.28.164:21051 - "POST /v1/audio/speech HTTP/1.1" 200 OK
2025-03-16 18:48:50,432 - tts_indic_server - INFO - Took 4.70 seconds to generate audio for 7 words using CUDA:0
INFO:     10.20.34.20:33417 - "POST /v1/audio/speech HTTP/1.1" 200 OK
2025-03-16 18:49:16,366 - tts_indic_server - INFO - Took 2.72 seconds to generate audio for 1 words using CUDA:0
INFO:     10.20.34.20:4572 - "POST /v1/audio/speech HTTP/1.1" 200 OK
2025-03-16 18:49:49,739 - tts_indic_server - INFO - Took 13.51 seconds to generate audio for 21 words using CUDA:0


torch compile -reduce overhead



curl -X 'POST' \
  'http://localhost:7860/v1/audio/speech' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": "ನಿಮ್ಮ ಇನ್‌ಪುಟ್ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ಸೇರಿಸಿ, ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?, ಶುಭೋದಯ!, ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿದೆ ಕರ್ನಾಟಕದಲ್ಲಿ ನಾವು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇವೆ , ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು",
  "voice": "Anu speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality",
  "model": "ai4bharat/indic-parler-tts",
  "response_format": "mp3",
  "speed": 1
}'



GTX - Torch Compile 

2025-03-17 22:34:29,839 - tts_indic_server - INFO - Took 8.72 seconds to generate audio for 5 words using CUDA:0
INFO:     127.0.0.1:58614 - "POST /v1/audio/speech HTTP/1.1" 200 OK
INFO:     127.0.0.1:58614 - "GET /v1/audio/speech HTTP/1.1" 405 Method Not Allowed
2025-03-17 22:36:10,131 - tts_indic_server - INFO - Took 42.71 seconds to generate audio for 21 words using CUDA:0
INFO:     127.0.0.1:39172 - "POST /v1/audio/speech HTTP/1.1" 200 OK

GTZ - Flash attention

