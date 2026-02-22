import gradio as gr
import json
import logging
import spaces  # Import spaces before any CUDA-related operations
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load voice descriptions from JSON file
def load_voice_descriptions(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return []

# Initialize the model and tokenizers
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# Function to generate audio locally
@spaces.GPU
def generate_audio_locally(input_text, voice_description_id):
    try:
        # Find the selected voice description
        selected_description = next((desc for desc in voice_descriptions if desc['userdomain_voice'] == voice_description_id), None)

        if selected_description:
            voice_description = selected_description['voice_description']
            output_file_name = selected_description['output_file_name']
        else:
            logger.error(f"Voice description not found for ID: {voice_description_id}")
            return f"Error: Voice description not found"

        # Prepare the input
        description_input_ids = description_tokenizer(voice_description, return_tensors="pt").to(device)
        prompt_input_ids = tokenizer(input_text, return_tensors="pt").to(device)
        # Generate the audio
        generation = model.generate(
            input_ids=description_input_ids.input_ids,
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask
        )

        audio_arr = generation.cpu().numpy().squeeze()

        # Save the audio file
        sf.write(output_file_name, audio_arr, model.config.sampling_rate)
        logger.info(f"Audio file saved to: {output_file_name}")

        # Return the path to the saved audio file
        return output_file_name
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return f"Error: {e}"

# Load voice descriptions from JSON file
voice_descriptions = load_voice_descriptions('voice_description_indian.json')

# Extract IDs and descriptions for dropdown menu
dropdown_choices = [(desc['userdomain_voice'], f"{desc['userdomain_voice']}: {desc['voice_description'][:50]}...") for desc in voice_descriptions]

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### Text-to-Speech Generator")

    with gr.Row():
        input_text = gr.Textbox(label="Enter Text", placeholder="Type your text here...")

    with gr.Row():
        voice_dropdown = gr.Dropdown(
            choices=[choice[0] for choice in dropdown_choices],
            label="Select Voice Description",
            type="value",
            value=dropdown_choices[0][0] if dropdown_choices else None,
            interactive=True,
        )

    with gr.Row():
        output_audio = gr.Audio(label="Generated Audio")

    submit_button = gr.Button("Generate Audio")

    def process_request(input_text, voice_description_id):
        return generate_audio_locally(input_text, voice_description_id)

    submit_button.click(
        process_request,
        inputs=[input_text, voice_dropdown],
        outputs=[output_audio]
    )

# Launch the Gradio demo
try:
    demo.launch()
except Exception as e:
    logger.error(f"Failed to launch Gradio demo: {e}")