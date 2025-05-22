import torch
import torchaudio
import gradio as gr
from os import getenv

import re
import numpy as np
from pathlib import Path
from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device


from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device

CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None

SPEAKER_EMBEDDING = None
SPEAKER_AUDIO_PATH = None


def load_model_if_needed(model_choice: str):
    global CURRENT_MODEL_TYPE, CURRENT_MODEL
    if CURRENT_MODEL_TYPE != model_choice:
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            torch.cuda.empty_cache()
        print(f"Loading {model_choice} model...")
        CURRENT_MODEL = Zonos.from_pretrained(model_choice, device=device)
        CURRENT_MODEL.requires_grad_(False).eval()
        CURRENT_MODEL_TYPE = model_choice
        print(f"{model_choice} model loaded successfully!")
    return CURRENT_MODEL


def update_ui(model_choice):
    """
    Dynamically show/hide UI elements based on the model's conditioners.
    We do NOT display 'language_id' or 'ctc_loss' even if they exist in the model.
    """
    model = load_model_if_needed(model_choice)
    cond_names = [c.name for c in model.prefix_conditioner.conditioners]
    print("Conditioners in this model:", cond_names)

    text_update = gr.update(visible=("espeak" in cond_names))
    language_update = gr.update(visible=("espeak" in cond_names))
    speaker_audio_update = gr.update(visible=("speaker" in cond_names))
    prefix_audio_update = gr.update(visible=True)
    emotion1_update = gr.update(visible=("emotion" in cond_names))
    emotion2_update = gr.update(visible=("emotion" in cond_names))
    emotion3_update = gr.update(visible=("emotion" in cond_names))
    emotion4_update = gr.update(visible=("emotion" in cond_names))
    emotion5_update = gr.update(visible=("emotion" in cond_names))
    emotion6_update = gr.update(visible=("emotion" in cond_names))
    emotion7_update = gr.update(visible=("emotion" in cond_names))
    emotion8_update = gr.update(visible=("emotion" in cond_names))
    vq_single_slider_update = gr.update(visible=("vqscore_8" in cond_names))
    fmax_slider_update = gr.update(visible=("fmax" in cond_names))
    pitch_std_slider_update = gr.update(visible=("pitch_std" in cond_names))
    speaking_rate_slider_update = gr.update(visible=("speaking_rate" in cond_names))
    dnsmos_slider_update = gr.update(visible=("dnsmos_ovrl" in cond_names))
    speaker_noised_checkbox_update = gr.update(visible=("speaker_noised" in cond_names))
    unconditional_keys_update = gr.update(
        choices=[name for name in cond_names if name not in ("espeak", "language_id")]
    )

    return (
        text_update,
        language_update,
        speaker_audio_update,
        prefix_audio_update,
        emotion1_update,
        emotion2_update,
        emotion3_update,
        emotion4_update,
        emotion5_update,
        emotion6_update,
        emotion7_update,
        emotion8_update,
        vq_single_slider_update,
        fmax_slider_update,
        pitch_std_slider_update,
        speaking_rate_slider_update,
        dnsmos_slider_update,
        speaker_noised_checkbox_update,
        unconditional_keys_update,
    )


def generate_audio(
    model_choice,
    text,
    language,
    speaker_audio,
    prefix_audio,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    e8,
    vq_single,
    fmax,
    pitch_std,
    speaking_rate,
    dnsmos_ovrl,
    speaker_noised,
    cfg_scale,
    top_p,
    top_k,
    min_p,
    linear,
    confidence,
    quadratic,
    seed,
    randomize_seed,
    unconditional_keys,
    progress=gr.Progress(),
):
    """
    Generates audio based on the provided UI parameters.
    We do NOT use language_id or ctc_loss even if the model has them.
    """
    selected_model = load_model_if_needed(model_choice)

    speaker_noised_bool = bool(speaker_noised)
    fmax = float(fmax)
    pitch_std = float(pitch_std)
    speaking_rate = float(speaking_rate)
    dnsmos_ovrl = float(dnsmos_ovrl)
    cfg_scale = float(cfg_scale)
    top_p = float(top_p)
    top_k = int(top_k)
    min_p = float(min_p)
    linear = float(linear)
    confidence = float(confidence)
    quadratic = float(quadratic)
    seed = int(seed)
    max_new_tokens = 86 * 30

    # This is a bit ew, but works for now.
    global SPEAKER_AUDIO_PATH, SPEAKER_EMBEDDING

    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    torch.manual_seed(seed)

    if speaker_audio is not None and "speaker" not in unconditional_keys:
        if speaker_audio != SPEAKER_AUDIO_PATH:
            print("Recomputed speaker embedding")
            wav, sr = torchaudio.load(speaker_audio)
            SPEAKER_EMBEDDING = selected_model.make_speaker_embedding(wav, sr)
            SPEAKER_EMBEDDING = SPEAKER_EMBEDDING.to(device, dtype=torch.bfloat16)
            SPEAKER_AUDIO_PATH = speaker_audio

    audio_prefix_codes = None
    if prefix_audio is not None:
        wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        wav_prefix = selected_model.autoencoder.preprocess(wav_prefix, sr_prefix)
        wav_prefix = wav_prefix.to(device, dtype=torch.float32)
        audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))

    emotion_tensor = torch.tensor(list(map(float, [e1, e2, e3, e4, e5, e6, e7, e8])), device=device)

    vq_val = float(vq_single)
    vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)

    cond_dict = make_cond_dict(
        text=text,
        language=language,
        speaker=SPEAKER_EMBEDDING,
        emotion=emotion_tensor,
        vqscore_8=vq_tensor,
        fmax=fmax,
        pitch_std=pitch_std,
        speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_noised=speaker_noised_bool,
        device=device,
        unconditional_keys=unconditional_keys,
    )
    conditioning = selected_model.prepare_conditioning(cond_dict)

    estimated_generation_duration = 30 * len(text) / 400
    estimated_total_steps = int(estimated_generation_duration * 86)

    def update_progress(_frame: torch.Tensor, step: int, _total_steps: int) -> bool:
        progress((step, estimated_total_steps))
        return True

    codes = selected_model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        batch_size=1,
        sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear, conf=confidence, quad=quadratic),
        callback=update_progress,
    )

    wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
    sr_out = selected_model.autoencoder.sampling_rate
    if wav_out.dim() == 2 and wav_out.size(0) > 1:
        wav_out = wav_out[0:1, :]
    return (sr_out, wav_out.squeeze().numpy()), seed

##### Plugin code Start #####

def batch_generate_audio(
    model_choice: str,
    text_dir: str,
    output_dir: str,
    language: str = "en-us",
    speaker_audio_path: str = None,
    cfg_scale: float = 2.0,
    min_p: float = 0.15,
    seed: int = 420,
    # Fixed emotion parameters matching Gradio defaults
    e1: float = 1.0,    # Happiness
    e2: float = 0.05,   # Sadness
    e3: float = 0.05,   # Disgust
    e4: float = 0.05,   # Fear
    e5: float = 0.05,   # Surprise
    e6: float = 0.05,   # Anger
    e7: float = 0.1,    # Other
    e8: float = 0.2,    # Neutral
):
    """Batch generate audio from text files with consistent tone"""

    title_emotion_map = { 
        "e1": 0.7,   # Happiness (adds warmth and enthusiasm)
        "e2": 0.02,  # Sadness (keep minimal to avoid a dull tone)
        "e3": 0.01,  # Disgust (not needed for emphasis)
        "e4": 0.03,  # Fear (keep low unless dramatic effect is needed)
        "e5": 0.3,   # Surprise (helps add excitement)
        "e6": 0.02,  # Anger (keep minimal unless intensity is required)
        "e7": 0.1,   # Other (neutral balance)
        "e8": 0.6,   # Neutral (reduce slightly to allow other emotions to shine)
    }
    
    normal_emotion_map = { 
        "e1": 0.5,   # Happiness
        "e2": 0.05,  # Sadness
        "e3": 0.01,  # Disgust
        "e4": 0.05,  # Fear
        "e5": 0.1,   # Surprise
        "e6": 0.01,  # Anger
        "e7": 0.1,   # Other
        "e8": 0.8,    # Neutral
    }
    
    # Load model once
    model = load_model_if_needed(model_choice)
    device = model.device

    # Force deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process speaker embedding once if provided
    speaker_embedding = None
    if speaker_audio_path:
        wav, sr = torchaudio.load(speaker_audio_path)
        speaker_embedding = model.make_speaker_embedding(wav, sr)
        speaker_embedding = speaker_embedding.to(device, dtype=torch.bfloat16)

    
    # Fixed emotion tensor
    normal_emotion_tensor = torch.tensor(
        [normal_emotion_map[f"e{i+1}"] for i in range(8)],
        device=device,
        dtype=torch.float32
    )

    title_emotion_tensor = torch.tensor(
        [title_emotion_map[f"e{i+1}"] for i in range(8)],
        device=device,
        dtype=torch.float32
    )
    
    # Process text files
    text_files = sorted(list(Path(text_dir).glob("*.txt")))
    for text_file in text_files:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()[:500]

        # Prepend silence to the text if needed
        # if the text start with # or $ then prepend silence of 750 ms
        # if the text start with #, the add silence of 750 ms at the end
        # Remove the leading #, $ and whitespace
        silence_start = 0
        silence_end = 0
        is_title = False
        if text.startswith("#"):
            is_title = True
            silence_start = 500
            silence_end = 1000
            text = text[1:].lstrip()
            text = text.upper() + "!"
        elif text.startswith("$"):
            silence_start = 750
            text = text[1:].lstrip()
        
        # Create silence audio
        if (silence_start > 0):
            wav_file_name = modify_filename(f"{text_file.stem}", 0)
            output_file = output_path / f"{wav_file_name}.wav"
            create_silent_wav(silence_start, output_file, model.autoencoder.sampling_rate)
        
        # Append silence to the text if needed
        if (silence_end > 0):
            wav_file_name = modify_filename(f"{text_file.stem}", 9)
            output_file = output_path / f"{wav_file_name}.wav"
            create_silent_wav(silence_end, output_file, model.autoencoder.sampling_rate)
        
        emotion_tensor = title_emotion_tensor if is_title else normal_emotion_tensor
        textStack = []
        textStack.append(text)
        seg = 1

        while len(textStack) > 0:
            text = textStack.pop()
            # Create consistent conditioning
            cond_dict = make_cond_dict(
                text=text,
                language=language,
                speaker=speaker_embedding,
                emotion=emotion_tensor,
                device=device,
                # Only disable these if your model supports them
                unconditional_keys=["vqscore_8", "dnsmos_ovrl"]
            )

            conditioning = model.prepare_conditioning(cond_dict)

            # Deterministic generation
            with torch.no_grad():
                codes, step, remaining_steps  = model.generate(
                    prefix_conditioning=conditioning,
                    max_new_tokens=86 * 30,
                    cfg_scale=cfg_scale,
                    batch_size=1,
                    sampling_params=dict(min_p=min_p),
                )
                print(f"Generated {text_file.stem} with {step} steps.")
                if step >= 86 * 27:
                    print(f"Steps exceeds the max step, divide text and generate more files.")
                    text1, text2 = divide_string(text)
                    textStack.append(text2)
                    textStack.append(text1)
                    # break and continue to the next text file
                    continue

            # Audio processing and saving (keep previous fix)
            wav_out = model.autoencoder.decode(codes).cpu().detach()
            if wav_out.dim() == 3: wav_out = wav_out.squeeze(0)
            if wav_out.dim() == 1: wav_out = wav_out.unsqueeze(0)
            if wav_out.dim() == 2 and wav_out.size(0) > 1:
                wav_out = wav_out[0:1, :]

            wav_file_name = modify_filename(f"{text_file.stem}", seg)
            seg += 1
            output_file = output_path / f"{wav_file_name}.wav"
            torchaudio.save(str(output_file), wav_out, model.autoencoder.sampling_rate)
            print(text)

    return f"Processed {len(text_files)} files with consistent tone to {output_dir}"

def divide_string(input_string):
    """
    Divides a string into two parts.
    - Splits on paragraphs (double newlines) if present.
    - Otherwise, splits while keeping sentences intact.
    
    :param input_string: The string to be divided.
    :return: A tuple containing two parts (part1, part2).
    """
    # Check if the input contains paragraphs (double newlines)
    paragraphs = input_string.split("\n\n")
    if len(paragraphs) > 1:
        # Split into two parts based on paragraphs
        midpoint = len(paragraphs) // 2
        part1 = "\n\n".join(paragraphs[:midpoint])
        part2 = "\n\n".join(paragraphs[midpoint:])
        return part1, part2

    # If no paragraphs, split based on sentences
    # Use regex to match sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', input_string)
    if len(sentences) > 1:
        midpoint = len(sentences) // 2
        part1 = " ".join(sentences[:midpoint])
        part2 = " ".join(sentences[midpoint:])
        return part1, part2

    # If neither paragraphs nor sentences can be used, split in the middle
    length = len(input_string)
    midpoint = (length + 1) // 2
    part1 = input_string[:midpoint]
    part2 = input_string[midpoint:]
    return part1, part2

def modify_filename(file_name, segment):
    """
    Modifies the given file name by replacing the sixth character with the given segment number.

    :param file_name: The original file name (e.g., "002341_22_Another_File").
    :param segment: The segment number (int) to replace the sixth character.
    :return: The modified file name as a string.
    """
    if len(file_name) < 6:
        raise ValueError("File name must have at least six characters.")
    
    # Replace only the sixth character with the segment and return the result
    new_file_name = file_name[:5] + str(segment) + file_name[6:]
    return new_file_name


def create_silent_wav(duration_ms, file_path, sampling_rate=16000):
    """
    Creates a silent WAV file with the specified duration and saves it to the given file path.

    :param duration_ms: Duration of the silent audio in milliseconds.
    :param file_path: Path to save the silent WAV file.
    :param sampling_rate: Sampling rate of the audio in Hz (default: 16000).
    """
    # Calculate the number of samples
    num_samples = int((duration_ms / 1000) * sampling_rate)
    
    # Create a silent audio tensor (1 channel, num_samples)
    silent_audio = torch.zeros((1, num_samples), dtype=torch.float32)
    
    # Save the silent audio as a WAV file
    torchaudio.save(file_path, silent_audio, sampling_rate)

##### Plugin code End #####
def build_interface():
    supported_models = []
    if "transformer" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-transformer")

    if "hybrid" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-hybrid")
    else:
        print(
            "| The current ZonosBackbone does not support the hybrid architecture, meaning only the transformer model will be available in the model selector.\n"
            "| This probably means the mamba-ssm library has not been installed."
        )

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(
                    choices=supported_models,
                    value=supported_models[0],
                    label="Zonos Model Type",
                    info="Select the model variant to use.",
                )
                text = gr.Textbox(
                    label="Text to Synthesize",
                    value="Zonos uses eSpeak for text to phoneme conversion!",
                    lines=4,
                    max_length=500,  # approximately
                )
                language = gr.Dropdown(
                    choices=supported_language_codes,
                    value="en-us",
                    label="Language Code",
                    info="Select a language code.",
                )
            prefix_audio = gr.Audio(
                value="assets/silence_100ms.wav",
                label="Optional Prefix Audio (continue from this audio)",
                type="filepath",
            )
            with gr.Column():
                speaker_audio = gr.Audio(
                    label="Optional Speaker Audio (for cloning)",
                    type="filepath",
                )
                speaker_noised_checkbox = gr.Checkbox(label="Denoise Speaker?", value=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Conditioning Parameters")
                dnsmos_slider = gr.Slider(1.0, 5.0, value=4.0, step=0.1, label="DNSMOS Overall")
                fmax_slider = gr.Slider(0, 24000, value=24000, step=1, label="Fmax (Hz)")
                vq_single_slider = gr.Slider(0.5, 0.8, 0.78, 0.01, label="VQ Score")
                pitch_std_slider = gr.Slider(0.0, 300.0, value=45.0, step=1, label="Pitch Std")
                speaking_rate_slider = gr.Slider(5.0, 30.0, value=15.0, step=0.5, label="Speaking Rate")

            with gr.Column():
                gr.Markdown("## Generation Parameters")
                cfg_scale_slider = gr.Slider(1.0, 5.0, 2.0, 0.1, label="CFG Scale")
                seed_number = gr.Number(label="Seed", value=420, precision=0)
                randomize_seed_toggle = gr.Checkbox(label="Randomize Seed (before generation)", value=True)

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### NovelAi's unified sampler")
                    linear_slider = gr.Slider(-2.0, 2.0, 0.5, 0.01, label="Linear (set to 0 to disable unified sampling)", info="High values make the output less random.")
                    #Conf's theoretical range is between -2 * Quad and 0.
                    confidence_slider = gr.Slider(-2.0, 2.0, 0.40, 0.01, label="Confidence", info="Low values make random outputs more random.")
                    quadratic_slider = gr.Slider(-2.0, 2.0, 0.00, 0.01, label="Quadratic", info="High values make low probablities much lower.")
                with gr.Column():
                    gr.Markdown("### Legacy sampling")
                    top_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Top P")
                    min_k_slider = gr.Slider(0.0, 1024, 0, 1, label="Min K")
                    min_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Min P")

        with gr.Accordion("Advanced Parameters", open=False):
            gr.Markdown(
                "### Unconditional Toggles\n"
                "Checking a box will make the model ignore the corresponding conditioning value and make it unconditional.\n"
                'Practically this means the given conditioning feature will be unconstrained and "filled in automatically".'
            )
            with gr.Row():
                unconditional_keys = gr.CheckboxGroup(
                    [
                        "speaker",
                        "emotion",
                        "vqscore_8",
                        "fmax",
                        "pitch_std",
                        "speaking_rate",
                        "dnsmos_ovrl",
                        "speaker_noised",
                    ],
                    value=["emotion"],
                    label="Unconditional Keys",
                )

            gr.Markdown(
                "### Emotion Sliders\n"
                "Warning: The way these sliders work is not intuitive and may require some trial and error to get the desired effect.\n"
                "Certain configurations can cause the model to become unstable. Setting emotion to unconditional may help."
            )
            with gr.Row():
                emotion1 = gr.Slider(0.0, 1.0, 1.0, 0.05, label="Happiness")
                emotion2 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Sadness")
                emotion3 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Disgust")
                emotion4 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Fear")
            with gr.Row():
                emotion5 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Surprise")
                emotion6 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Anger")
                emotion7 = gr.Slider(0.0, 1.0, 0.1, 0.05, label="Other")
                emotion8 = gr.Slider(0.0, 1.0, 0.2, 0.05, label="Neutral")

        with gr.Column():
            generate_button = gr.Button("Generate Audio")
            output_audio = gr.Audio(label="Generated Audio", type="numpy", autoplay=True)

        model_choice.change(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )

        # On page load, trigger the same UI refresh
        demo.load(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )

        # Generate audio on button click
        generate_button.click(
            fn=generate_audio,
            inputs=[
                model_choice,
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                top_p_slider,
                min_k_slider,
                min_p_slider,
                linear_slider,
                confidence_slider,
                quadratic_slider,
                seed_number,
                randomize_seed_toggle,
                unconditional_keys,
            ],
            outputs=[output_audio, seed_number],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    share = getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)
