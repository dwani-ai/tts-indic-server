from locust import HttpUser, task, between
import os

class TranscribeUser(HttpUser):
    wait_time = between(1, 5)  # Wait time between tasks

    @task
    def transcribe_audio(self):
        audio_file_path = "./../../kannada_sample_1.wav"
        with open(audio_file_path, 'rb') as audio_file:
            files = {'file': ('kannada_query_infer.wav', audio_file, 'audio/x-wav')}
            headers = {
                'accept': 'application/json'
            }
            response = self.client.post("http://localhost:8000/transcribe/", files=files, headers=headers)
            if response.status_code == 200:
                print("Success:", response.json())
            else:
                print("Failed:", response.status_code, response.text)

    @task
    def transcribe_batch(self):
        batch_files = [
            "./../../kannada_sample_1.wav",
            "./../../kannada_sample_2.wav"
        ]
        files = []
        for i, file_path in enumerate(batch_files):
            with open(file_path, 'rb') as audio_file:
                files.append(('file', (f'kannada_query_infer_{i}.wav', audio_file, 'audio/x-wav')))

        headers = {
            'accept': 'application/json'
        }
        response = self.client.post("http://localhost:8000/transcribe_batch/", files=files, headers=headers)
        if response.status_code == 200:
            print("Batch Success:", response.json())
        else:
            print("Batch Failed:", response.status_code, response.text)