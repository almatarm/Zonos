##### batch_process.py #####
import argparse
from gradio_interface import batch_generate_audio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process text files into audio')
    parser.add_argument('--model', type=str, default="Zyphra/Zonos-v0.1-transformer",
                       help='Model name from Hugging Face Hub')
    parser.add_argument('--text-dir', type=str, required=True,
                       help='Directory containing text files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save audio files')
    parser.add_argument('--speaker-audio', type=str, default=None,
                       help='Path to speaker audio for voice cloning')
    parser.add_argument('--language', type=str, default="en-us",
                       help='Language code for synthesis')
    parser.add_argument('--seed', type=int, default=420,
                       help='Random seed for reproducibility')

    args = parser.parse_args()


    emotion_map = { 
        "e1": 0.5,   # Happiness
        "e2": 0.05,  # Sadness
        "e3": 0.01,  # Disgust
        "e4": 0.05,  # Fear
        "e5": 0.1,   # Surprise
        "e6": 0.01,  # Anger
        "e7": 0.1,   # Other
        "e8": 0.8,    # Neutral
    }
    
    neutral_emotion_map = {
        "e1": 0.0,
        "e2": 0.0,
        "e3": 0.0,
        "e4": 0.0,
        "e5": 0.0,
        "e6": 0.0,
        "e7": 0.0,
        "e8": 1.0,  # Neutral
    }

    result = batch_generate_audio(
        model_choice=args.model,
        text_dir=args.text_dir,
        output_dir=args.output_dir,
        language=args.language,
        speaker_audio_path=args.speaker_audio,
        seed=args.seed,
        **emotion_map
    )

    print(result)