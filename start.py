import os
import json
import time
import threading
import wave
from datetime import datetime
from typing import Optional

import keyboard
import sounddevice as sd
import wave
import numpy as np
import pyperclip
import pyautogui
from openai import OpenAI
from dotenv import load_dotenv

import config

# Load environment variables
load_dotenv()

class VoiceDictationTool:
    def __init__(self):
        """Initialize the voice dictation tool."""
        self.client = self._initialize_openai_client()
        self.is_recording = False
        self.audio_data = []
        self.recording_thread = None
        self.temp_audio_file = None
        
        # Locked recording state
        self.is_locked_recording = False
        self.locked_start_time = None
        
        # Ensure temp directory exists
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        
        # Initialize recordings file
        self._initialize_recordings_file()
        
        # Test microphone availability
        self._test_microphone()
        
        print("🎙️ Voice Dictation Tool initialized!")
        print(f"📋 Normal Recording: {config.HOTKEY}")
        print("🔊 Press and hold Ctrl+Alt to record")
        print("🔒 Press Ctrl+Shift+Alt to start/stop locked recording")
        print("🔄 While recording, add Shift to transition to locked mode")
    
    def _initialize_openai_client(self) -> OpenAI:
        """Initialize OpenAI client with API key from environment."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "❌ OPENAI_API_KEY not found in environment variables.\n"
                "Please create a .env file with your OpenAI API key:\n"
                "OPENAI_API_KEY=your-api-key-here"
            )
        
        return OpenAI(api_key=api_key)
    
    def _initialize_recordings_file(self):
        """Initialize the recordings JSON file if it doesn't exist."""
        if not os.path.exists(config.RECORDINGS_FILE):
            with open(config.RECORDINGS_FILE, 'w') as f:
                json.dump([], f, indent=2)
    
    def _test_microphone(self):
        """Test microphone availability and permissions."""
        try:
            print("🧪 Testing microphone...")
            
            # Query available devices
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            
            print(f"🎤 Found {len(input_devices)} input device(s)")
            
            if len(input_devices) == 0:
                print("❌ No input devices found! Please check your microphone connection.")
                return
            
            # Get default input device
            default_device = sd.query_devices(kind='input')
            print(f"🎤 Default input: {default_device['name']}")
            print(f"📊 Sample rate: {default_device['default_samplerate']} Hz")
            print(f"🔢 Channels: {default_device['max_input_channels']}")
            
            # Test recording for 0.1 seconds
            test_duration = 0.1
            test_recording = sd.rec(
                int(test_duration * config.RECORDING_SETTINGS['sample_rate']),
                samplerate=config.RECORDING_SETTINGS['sample_rate'],
                channels=config.RECORDING_SETTINGS['channels'],
                dtype='float64'
            )
            sd.wait()  # Wait for recording to complete
            
            # Check if we got any data
            if len(test_recording) > 0:
                max_amplitude = float(max(abs(test_recording.max()), abs(test_recording.min())))
                print(f"✅ Microphone test successful! Max amplitude: {max_amplitude:.4f}")
                if max_amplitude < 0.001:
                    print("⚠️ Warning: Very low audio level detected. Speak louder or check microphone settings.")
            else:
                print("❌ Microphone test failed - no audio data received")
                
        except Exception as e:
            print(f"❌ Microphone test error: {e}")
            print("💡 Try running as administrator or check microphone permissions")
    
    def _save_recording_to_history(self, transcription: str, audio_file: str = None):
        """Save transcription to recordings history."""
        try:
            print(f"💾 Saving to history: '{transcription[:50]}{'...' if len(transcription) > 50 else ''}'")
            
            # Load existing recordings
            with open(config.RECORDINGS_FILE, 'r') as f:
                recordings = json.load(f)
            
            # Add new recording (without audio file path to save space)
            recording_entry = {
                'timestamp': datetime.now().isoformat(),
                'transcription': transcription
            }
            recordings.append(recording_entry)
            
            # Save back to file
            with open(config.RECORDINGS_FILE, 'w') as f:
                json.dump(recordings, f, indent=2)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Force OS to write to disk
            
            print(f"✅ Saved to recordings.json (total entries: {len(recordings)})")
            
            # Verify the save worked by reading it back
            try:
                with open(config.RECORDINGS_FILE, 'r') as f:
                    verify_recordings = json.load(f)
                print(f"🔍 Verified: File contains {len(verify_recordings)} entries")
            except Exception as verify_error:
                print(f"⚠️ Could not verify save: {verify_error}")
                
        except Exception as e:
            print(f"❌ Error saving to recordings history: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_temp_filename(self) -> str:
        """Generate a unique temporary filename for audio recording."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return os.path.join(
            config.TEMP_DIR, 
            f"{config.TEMP_FILE_PREFIX}{timestamp}.{config.AUDIO_FORMAT}"
        )
    
    def _start_recording(self):
        """Start audio recording in a separate thread."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.audio_data = []
        self.temp_audio_file = self._generate_temp_filename()
        
        print("🔴 Recording started...")
        
        def record_audio():
            """Record audio in separate thread."""
            try:
                # Check if we have audio devices
                devices = sd.query_devices()
                print(f"🎤 Available audio devices: {len(devices)}")
                
                # Get default input device
                default_device = sd.query_devices(kind='input')
                print(f"🎤 Default input device: {default_device['name']}")
                
                # Start recording with a more robust approach
                max_minutes = float(os.getenv('MAX_RECORDING_MINUTES', '1'))  # Default 1 minute
                duration = max_minutes * 60  # Convert to seconds
                recording = sd.rec(
                    int(config.RECORDING_SETTINGS['sample_rate'] * duration),
                    samplerate=config.RECORDING_SETTINGS['sample_rate'],
                    channels=config.RECORDING_SETTINGS['channels'],
                    dtype='float64',
                    device=None  # Use default device
                )
                
                # Wait while recording, checking every 100ms
                start_time = time.time()
                while self.is_recording and (time.time() - start_time) < duration:
                    time.sleep(0.1)  # Check every 100ms
                
                # Stop recording
                sd.stop()
                
                # Calculate actual recording length based on time
                actual_duration = time.time() - start_time
                actual_samples = int(actual_duration * config.RECORDING_SETTINGS['sample_rate'])
                
                if actual_samples > 0 and actual_samples <= len(recording):
                    self.audio_data = recording[:actual_samples]
                    print(f"🎵 Recorded {actual_duration:.2f} seconds (max: {max_minutes:.1f} min)")
                else:
                    print(f"⚠️ Invalid recording length: {actual_samples} samples")
                    self.audio_data = []
                
            except Exception as e:
                print(f"❌ Recording error: {e}")
                import traceback
                traceback.print_exc()
                self.is_recording = False
        
        self.recording_thread = threading.Thread(target=record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def _stop_recording(self):
        """Stop audio recording and process the audio."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        print("🟡 Recording stopped. Processing...")
        
        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
        
        # Save audio data to file
        try:
            if len(self.audio_data) > 0:
                # Check if recording meets minimum duration
                duration = self._get_recording_duration()
                if duration < config.MIN_RECORDING_SECONDS:
                    print(f"⚠️ Recording too short ({duration:.2f}s < {config.MIN_RECORDING_SECONDS}s). Skipping transcription.")
                    self._cleanup_temp_file()
                    return
                
                print(f"🎵 Recording duration: {duration:.2f} seconds")
                
                # Save as WAV file using built-in wave module
                self._save_wav_file(
                    self.temp_audio_file,
                    self.audio_data,
                    config.RECORDING_SETTINGS['sample_rate'],
                    config.RECORDING_SETTINGS['channels']
                )
                
                # Process the transcription
                self._process_transcription()
            else:
                print("⚠️ No audio data recorded.")
                
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
    
    def _process_transcription(self):
        """Send audio to OpenAI Whisper API and handle the response."""
        try:
            print("🤖 Transcribing audio...")
            
            # Open and send audio file to Whisper API
            with open(self.temp_audio_file, 'rb') as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=config.API_SETTINGS['model'],
                    file=audio_file,
                    response_format=config.API_SETTINGS['response_format'],
                    prompt=config.API_SETTINGS['prompt']  # Use the user-defined prompt
                )
            
            # Extract transcription text
            transcription = response.strip() if isinstance(response, str) else response.text.strip()
            
            # Apply custom vocabulary corrections
            corrected_transcription = self._apply_vocabulary_corrections(transcription)
            
            if corrected_transcription:
                if corrected_transcription != transcription:
                    print(f"🔧 Original: '{transcription}'")
                    print(f"✅ Corrected: '{corrected_transcription}'")
                else:
                    print(f"✅ Transcription: '{corrected_transcription}'")
                
                # Save to history (without audio file path since we're deleting it)
                self._save_recording_to_history(corrected_transcription, None)
                
                # Copy to clipboard and paste
                self._insert_text(corrected_transcription)
                
            else:
                print("⚠️ No speech detected in recording.")
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
        
        finally:
            # Clean up temporary file immediately
            self._cleanup_temp_file()
    
    def _insert_text(self, text: str):
        """Insert text at current cursor position."""
        try:
            # Copy text to clipboard
            pyperclip.copy(text)
            print("📋 Text copied to clipboard")
            
            # Small delay to ensure clipboard is ready
            time.sleep(0.1)
            
            # Paste text using Ctrl+V
            pyautogui.hotkey('ctrl', 'v')
            print("✅ Text pasted successfully!")
            
        except Exception as e:
            print(f"❌ Error inserting text: {e}")
            print(f"💾 Text saved to clipboard: '{text}'")
    
    def _save_wav_file(self, filename: str, audio_data, sample_rate: int, channels: int):
        """Save audio data to WAV file using built-in wave module (lightweight)."""
        try:
            # Convert float64 audio data to int16 for WAV format
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
                
        except Exception as e:
            print(f"❌ Error saving WAV file: {e}")
            raise
    
    def _apply_vocabulary_corrections(self, text: str) -> str:
        """Apply custom vocabulary corrections to improve transcription accuracy."""
        corrected_text = text
        
        # Apply case-insensitive corrections
        for correct_term, alternatives in config.CUSTOM_VOCABULARY.items():
            for alternative in alternatives:
                # Case-insensitive replacement while preserving surrounding text
                import re
                pattern = r'\b' + re.escape(alternative) + r'\b'
                corrected_text = re.sub(pattern, correct_term, corrected_text, flags=re.IGNORECASE)
        
        return corrected_text
    
    def _get_recording_duration(self) -> float:
        """Calculate the duration of recorded audio in seconds."""
        if len(self.audio_data) == 0:
            return 0.0
        
        sample_rate = config.RECORDING_SETTINGS['sample_rate']
        num_samples = len(self.audio_data)
        return num_samples / sample_rate
    
    def _cleanup_temp_file(self):
        """Clean up temporary audio file."""
        try:
            if self.temp_audio_file and os.path.exists(self.temp_audio_file):
                os.remove(self.temp_audio_file)
                self.temp_audio_file = None
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up temp file: {e}")
    
    def _toggle_locked_recording(self):
        """Toggle locked recording on/off."""
        if self.is_locked_recording:
            self._stop_locked_recording()
        else:
            self._start_locked_recording()
    
    def _start_locked_recording(self):
        """Start locked recording mode."""
        if self.is_recording:
            print("⚠️ Already recording - finish current recording first")
            return
            
        print("🔒 LOCKED RECORDING STARTED - Press Ctrl+Shift+Alt again to stop")
        self.is_locked_recording = True
        self.locked_start_time = time.time()
        self._start_recording()
    
    def _stop_locked_recording(self):
        """Stop locked recording mode."""
        if not self.is_locked_recording:
            return
            
        duration = time.time() - self.locked_start_time if self.locked_start_time else 0
        print(f"🔓 Locked recording stopped ({duration:.1f}s) - Processing...")
        self.is_locked_recording = False
        self.locked_start_time = None
        self._stop_recording()
    
    def _transition_to_locked_recording(self):
        """Transition from normal recording to locked recording mode."""
        if not self.is_recording or self.is_locked_recording:
            return
            
        print("🔄 Transitioning to LOCKED RECORDING - you can now release the keys")
        self.is_locked_recording = True
        self.locked_start_time = time.time()  # Mark when we transitioned to locked mode
    
    def _on_hotkey_press(self):
        """Handle hotkey press event."""
        if not self.is_locked_recording:
            self._start_recording()
    
    def _on_hotkey_release(self):
        """Handle hotkey release event."""
        if not self.is_locked_recording:
            self._stop_recording()
    
    def start_listening(self):
        """Start listening for global hotkeys."""
        try:
            print("👂 Listening for hotkeys...")
            print("Press Ctrl+C to exit")
            
            # Register hotkey callbacks
            keyboard.on_press_key('ctrl', lambda _: None)  # Enable modifier detection
            keyboard.on_press_key('alt', lambda _: None)   # Enable modifier detection
            keyboard.on_press_key('shift', lambda _: None) # Enable shift detection
            
            # Use a custom handler for press and release
            def hotkey_handler(event):
                if event.event_type == keyboard.KEY_DOWN:
                    # Check for locked recording toggle (Ctrl+Shift+Alt)
                    if (keyboard.is_pressed('ctrl') and 
                        keyboard.is_pressed('shift') and 
                        keyboard.is_pressed('alt')):
                        
                        # If we're already recording normally, transition to locked
                        if self.is_recording and not self.is_locked_recording:
                            self._transition_to_locked_recording()
                        # Otherwise, toggle locked recording as usual
                        else:
                            self._toggle_locked_recording()
                    
                    # Check for normal recording (Ctrl+Alt, but NOT Shift)
                    elif (keyboard.is_pressed('ctrl') and 
                          keyboard.is_pressed('alt') and 
                          not keyboard.is_pressed('shift')):
                        if not self.is_recording:
                            self._on_hotkey_press()
                            
                elif event.event_type == keyboard.KEY_UP:
                    # Only stop normal recording on key release (not locked recording)
                    if self.is_recording and not self.is_locked_recording:
                        # Check if either ctrl or alt was released
                        if not (keyboard.is_pressed('ctrl') and keyboard.is_pressed('alt')):
                            self._on_hotkey_release()
            
            # Hook all key events to detect our combination
            keyboard.hook(hotkey_handler)
            
            # Keep the program running
            keyboard.wait()
            
        except KeyboardInterrupt:
            print("\n👋 Exiting Voice Dictation Tool...")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            # Cleanup
            keyboard.unhook_all()
            self._cleanup_temp_file()

def main():
    """Main entry point."""
    try:
        # Initialize and start the voice dictation tool
        tool = VoiceDictationTool()
        tool.start_listening()
        
    except Exception as e:
        print(f"❌ Failed to start Voice Dictation Tool: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you have created a .env file with your OPENAI_API_KEY")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check microphone permissions")

if __name__ == "__main__":
    main() 