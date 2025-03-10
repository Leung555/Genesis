import whisper

class SpeechRecognizer:
    def __init__(self, model_name="tiny"):
        self.model = whisper.load_model(model_name)

    def transcribe_audio(self, audio_file):
        """Transcribes audio to text."""
        result = self.model.transcribe(audio_file)
        return result["text"].lower()
