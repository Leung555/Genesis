from scene_manager import SceneManager
from robot_agent import RobotAgent
from audio_recorder import AudioRecorder
from speech_recognizer import SpeechRecognizer

Activate_speech_recognition = False

class RobotAssistant:
    def __init__(self):
        # Initialize components
        self.scene_manager = SceneManager()
        self.robot_agent = RobotAgent(self.scene_manager)
        if Activate_speech_recognition:
            self.audio_recorder = AudioRecorder()
            self.speech_recognizer = SpeechRecognizer()

        # Build the scene
        self.scene_manager.build()

    def run(self):
        """Main loop for interacting with the robot."""
        try:
            while True:
                text_command = ''
                # Record audio command from user
                if Activate_speech_recognition:
                    self.audio_recorder.record_audio()
                    text_command = self.speech_recognizer.transcribe_audio(self.audio_recorder.temp_filename)
                print(f"User said: {text_command}")

                # Parse and map the command to an action
                action = self.robot_agent.parse_command_to_action(text_command)

                # Get the current state of the robot after executing the action
                observation = self.scene_manager.step(action)
                print(f"Robot state after action: {observation}")

        except KeyboardInterrupt:
            print("Stopping...")
            self.audio_recorder.audio.terminate()

# Run the Robot Assistant
if __name__ == "__main__":
    assistant = RobotAssistant()
    assistant.run()
