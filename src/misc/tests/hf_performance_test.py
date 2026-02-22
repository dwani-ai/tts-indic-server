import subprocess
import time

def run_transcription(device_type):
    start_time = time.time()
    result = subprocess.run(["python", "src/hf_asr.py", "--device", device_type], capture_output=True, text=True)
    end_time = time.time()
    execution_time = end_time - start_time
    transcribed_text = result.stdout.split("\n")[-2]  # Assuming the transcribed text is the second last line
    return transcribed_text, execution_time

def main():
    # Measure execution time for CUDA
    if torch.cuda.is_available():
        cuda_text, cuda_time = run_transcription("cuda")
        print(f"CUDA execution time: {cuda_time:.4f} seconds")
        print(f"Transcribed text (CUDA): {cuda_text}")

    # Measure execution time for CPU
    cpu_text, cpu_time = run_transcription("cpu")
    print(f"CPU execution time: {cpu_time:.4f} seconds")
    print(f"Transcribed text (CPU): {cpu_text}")

if __name__ == "__main__":
    main()