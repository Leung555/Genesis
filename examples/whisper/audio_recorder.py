import pyaudio
import wave

class AudioRecorder:
    def __init__(self, format=pyaudio.paInt16, channels=1, rate=16000, chunk=1024, record_seconds=3, temp_filename="temp.wav"):
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.record_seconds = record_seconds
        self.temp_filename = temp_filename
        self.audio = pyaudio.PyAudio()

    def record_audio(self):
        """Records speech and saves it to temp.wav."""
        print("Listening...")
        stream = self.audio.open(format=self.format, channels=self.channels,
                                 rate=self.rate, input=True, frames_per_buffer=self.chunk)
        frames = []

        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("Processing...")
        stream.stop_stream()
        stream.close()

        with wave.open(self.temp_filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
