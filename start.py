import os
import json
import time
import threading
import wave
import random
from datetime import datetime
from typing import Optional

# Replace keyboard with pynput for Mac compatibility
from pynput import keyboard
from pynput.keyboard import Key, Listener
import sounddevice as sd
import wave
import numpy as np
import pyperclip
# Remove pyautogui and use pynput for more reliable Mac pasting
from openai import OpenAI
from dotenv import load_dotenv

import config

# Load environment variables
load_dotenv()

# Prevent Python from showing in dock on macOS
import sys
if sys.platform == 'darwin':  # macOS only
    try:
        import AppKit
        # Initialize the app if needed
        app = AppKit.NSApplication.sharedApplication()
        # Tell macOS this is a background application (LSUIElement)
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    except (ImportError, AttributeError):
        # AppKit not available or initialization failed, continue anyway
        pass

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
        
        # Streaming transcription state
        self.streaming_chunks = []  # Track chunks being transcribed
        self.background_threads = []  # Active transcription threads
        self.current_chunk_data = []  # Current recording buffer
        self.chunk_start_time = None  # When current chunk started
        self.chunk_counter = 0  # Chunk numbering
        self.streaming_lock = threading.Lock()  # Thread safety
        
        # Ensure temp directory exists
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        
        # Initialize recordings file
        self._initialize_recordings_file()
        
        # Test microphone availability
        self._test_microphone()
        
        print("🎙️ Voice Dictation Tool initialized!")
        print(f"📋 Normal Recording: Cmd+Option")
        print("🔊 Press and hold Cmd+Option to record")
        print("🔒 Press Cmd+Option+Shift to start/stop locked recording")
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
        
        # Reset streaming state
        with self.streaming_lock:
            self.streaming_chunks = []
            self.current_chunk_data = []
            self.chunk_start_time = None
            self.chunk_counter = 0
        
        print("🔴 Recording started...")
        
        # Choose recording method based on streaming settings
        if config.STREAMING_SETTINGS['enabled']:
            print("🔄 Streaming transcription enabled")
            self.recording_thread = threading.Thread(target=self._record_with_streaming)
        else:
            self.recording_thread = threading.Thread(target=self._record_traditional)
        
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def _record_traditional(self):
        """Traditional recording method (original approach)."""
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
    
    def _record_with_streaming(self):
        """Streaming recording method with background transcription."""
        try:
            # Check if we have audio devices
            devices = sd.query_devices()
            print(f"🎤 Available audio devices: {len(devices)}")
            
            # Get default input device
            default_device = sd.query_devices(kind='input')
            print(f"🎤 Default input device: {default_device['name']}")
            
            # Streaming recording parameters
            chunk_interval = config.STREAMING_SETTINGS['chunk_interval_seconds']
            sample_rate = config.RECORDING_SETTINGS['sample_rate']
            block_size = int(sample_rate * 0.1)  # 100ms blocks
            
            # Start with empty buffers
            self.audio_data = []
            self.current_chunk_data = []
            self.chunk_start_time = time.time()
            
            print(f"🔄 Streaming mode: {chunk_interval}s chunks, transcribing in background")
            
            # Define callback for continuous recording
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"⚠️ Audio callback status: {status}")
                
                if self.is_recording:
                    # Add to both overall recording and current chunk
                    chunk_data = indata.copy()
                    self.audio_data.extend(chunk_data)
                    self.current_chunk_data.extend(chunk_data)
            
            # Start input stream
            with sd.InputStream(
                callback=audio_callback,
                samplerate=sample_rate,
                channels=config.RECORDING_SETTINGS['channels'],
                dtype='float32',  # Use float32 for streaming compatibility
                blocksize=block_size
            ):
                print("🎙️ Streaming recording active...")
                
                # Main recording loop
                while self.is_recording:
                    current_time = time.time()
                    chunk_duration = current_time - self.chunk_start_time
                    
                    # Check if we should save current chunk and start transcription
                    if chunk_duration >= chunk_interval:
                        self._process_streaming_chunk()
                        
                        # Start new chunk (with overlap)
                        overlap_samples = int(config.STREAMING_SETTINGS['overlap_seconds'] * sample_rate)
                        if len(self.current_chunk_data) > overlap_samples:
                            # Keep overlap for context
                            self.current_chunk_data = self.current_chunk_data[-overlap_samples:]
                        else:
                            self.current_chunk_data = []
                        
                        self.chunk_start_time = current_time
                    
                    # Check every 100ms
                    time.sleep(0.1)
                
                # Process final chunk when recording stops
                if len(self.current_chunk_data) > 0:
                    self._process_streaming_chunk()
                    
        except Exception as e:
            print(f"❌ Streaming recording error: {e}")
            import traceback
            traceback.print_exc()
            self.is_recording = False
    
    def _process_streaming_chunk(self):
        """Process a completed streaming chunk."""
        if not self.current_chunk_data:
            return
        
        chunk_duration = len(self.current_chunk_data) / config.RECORDING_SETTINGS['sample_rate']
        
        # Only transcribe chunks that meet minimum duration
        if chunk_duration >= config.STREAMING_SETTINGS['min_chunk_seconds']:
            chunk_data = np.array(self.current_chunk_data, dtype='float32')
            
            self._start_background_transcription(
                chunk_data,
                self.chunk_counter,
                self.chunk_start_time
            )
            
            self.chunk_counter += 1
        else:
            print(f"⚠️ Skipping short chunk ({chunk_duration:.1f}s < {config.STREAMING_SETTINGS['min_chunk_seconds']}s)")
    
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
                # Convert to numpy array and ensure float64 format for consistency
                audio_array = np.array(self.audio_data)
                if audio_array.dtype != 'float64':
                    audio_array = audio_array.astype('float64')
                
                self._save_wav_file(
                    self.temp_audio_file,
                    audio_array,
                    config.RECORDING_SETTINGS['sample_rate'],
                    config.RECORDING_SETTINGS['channels']
                )
                
                # Choose processing method based on streaming vs file size
                if config.STREAMING_SETTINGS['enabled'] and self.streaming_chunks:
                    self._process_streaming_transcription()
                elif self._is_large_file():
                    self._process_large_transcription()
                else:
                    self._process_transcription()
            else:
                print("⚠️ No audio data recorded.")
                
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
    
    def _call_openai_with_retry(self):
        """Call OpenAI API with exponential backoff retry logic."""
        max_retries = 3
        base_delay = 1
        max_delay = 60
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"🔄 Retry attempt {attempt}/{max_retries}")
                
                # Make the API call
                with open(self.temp_audio_file, 'rb') as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model=config.API_SETTINGS['model'],
                        file=audio_file,
                        response_format=config.API_SETTINGS['response_format'],
                        prompt=config.API_SETTINGS['prompt']
                    )
                
                # Success - return the response
                return response
                
            except Exception as e:
                error_message = str(e).lower()
                
                # Check if it's a retryable error
                retryable_errors = [
                    'connection error', 'network error', 'timeout', 
                    'rate limit', 'server error', '502', '503', '504',
                    'internal server error', 'bad gateway', 'service unavailable',
                    'gateway timeout', 'connection reset'
                ]
                
                is_retryable = any(err in error_message for err in retryable_errors)
                
                if attempt < max_retries and is_retryable:
                    # Exponential backoff with jitter
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0.1, 0.9) * delay
                    wait_time = delay + jitter
                    
                    print(f"❌ API error (attempt {attempt + 1}): {e}")
                    print(f"⏳ Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    # Either max retries reached or non-retryable error
                    if attempt >= max_retries:
                        print(f"❌ API failed after {max_retries + 1} attempts: {e}")
                    else:
                        print(f"❌ Non-retryable API error: {e}")
                    
                    # Re-raise the exception to be handled by calling method
                    raise e
        
        # This should never be reached, but just in case
        raise Exception("Unexpected error in retry logic")
    
    def _start_background_transcription(self, chunk_data, chunk_index, start_time):
        """Start background transcription for a chunk."""
        if not config.STREAMING_SETTINGS['enabled']:
            return
        
        # Check if we're at max concurrent transcriptions
        with self.streaming_lock:
            active_threads = [t for t in self.background_threads if t.is_alive()]
            self.background_threads = active_threads  # Clean up dead threads
            
            if len(active_threads) >= config.STREAMING_SETTINGS['max_concurrent_transcriptions']:
                print(f"⏳ Max concurrent transcriptions ({config.STREAMING_SETTINGS['max_concurrent_transcriptions']}) reached, queueing chunk {chunk_index}")
                return
        
        # Create chunk metadata
        chunk_info = {
            'index': chunk_index,
            'start_time': start_time,
            'duration': len(chunk_data) / config.RECORDING_SETTINGS['sample_rate'],
            'filename': None,
            'transcription': None,
            'status': 'transcribing',
            'error': None
        }
        
        # Add to tracking list
        with self.streaming_lock:
            self.streaming_chunks.append(chunk_info)
        
        # Start background thread
        thread = threading.Thread(
            target=self._transcribe_chunk_background,
            args=(chunk_data, chunk_info),
            daemon=True
        )
        
        with self.streaming_lock:
            self.background_threads.append(thread)
        
        thread.start()
        
        if config.STREAMING_SETTINGS['progress_feedback']:
            print(f"🔄 Started background transcription for chunk {chunk_index} ({chunk_info['duration']:.1f}s)")
    
    def _transcribe_chunk_background(self, chunk_data, chunk_info):
        """Transcribe a chunk in the background."""
        try:
            # Save chunk to temporary file
            chunk_filename = f"{self.temp_audio_file}_stream_chunk_{chunk_info['index']:03d}.wav"
            chunk_info['filename'] = chunk_filename
            
            # Save chunk as WAV file (convert float32 to float64 for consistency)
            chunk_data_float64 = chunk_data.astype('float64')
            self._save_wav_file(
                chunk_filename,
                chunk_data_float64,
                config.RECORDING_SETTINGS['sample_rate'],
                config.RECORDING_SETTINGS['channels']
            )
            
            # Transcribe using our retry logic
            temp_backup = self.temp_audio_file
            self.temp_audio_file = chunk_filename
            
            try:
                response = self._call_openai_with_retry()
                transcription = response.strip() if isinstance(response, str) else response.text.strip()
                
                # Apply vocabulary corrections
                corrected_transcription = self._apply_vocabulary_corrections(transcription)
                
                # Update chunk info
                with self.streaming_lock:
                    chunk_info['transcription'] = corrected_transcription
                    chunk_info['status'] = 'completed'
                
                if config.STREAMING_SETTINGS['progress_feedback']:
                    print(f"✅ Chunk {chunk_info['index']} transcribed: '{corrected_transcription[:50]}{'...' if len(corrected_transcription) > 50 else ''}'")
                
            finally:
                self.temp_audio_file = temp_backup
                # Clean up chunk file
                try:
                    if os.path.exists(chunk_filename):
                        os.remove(chunk_filename)
                except Exception as e:
                    print(f"⚠️ Could not delete chunk file: {e}")
                    
        except Exception as e:
            # Update chunk with error
            with self.streaming_lock:
                chunk_info['error'] = str(e)
                chunk_info['status'] = 'failed'
            
            if config.STREAMING_SETTINGS['progress_feedback']:
                print(f"❌ Chunk {chunk_info['index']} failed: {e}")
    
    def _wait_for_background_transcriptions(self):
        """Wait for all background transcriptions to complete."""
        if not config.STREAMING_SETTINGS['enabled']:
            return
        
        print("⏳ Waiting for background transcriptions to complete...")
        
        # Wait for all threads to complete
        with self.streaming_lock:
            threads_to_wait = list(self.background_threads)
        
        for thread in threads_to_wait:
            if thread.is_alive():
                thread.join(timeout=30)  # 30 second timeout per thread
        
        # Clean up thread list
        with self.streaming_lock:
            self.background_threads = []
        
        print("✅ All background transcriptions completed")
    
    def _merge_streaming_chunks(self):
        """Merge all streaming chunks into final transcription."""
        if not self.streaming_chunks:
            return ""
        
        # Sort chunks by index
        with self.streaming_lock:
            sorted_chunks = sorted(self.streaming_chunks, key=lambda x: x['index'])
        
        transcriptions = []
        failed_chunks = []
        
        for chunk in sorted_chunks:
            if chunk['status'] == 'completed' and chunk['transcription']:
                transcriptions.append({
                    'text': chunk['transcription'],
                    'chunk_index': chunk['index']
                })
            else:
                failed_chunks.append(chunk['index'])
                transcriptions.append({
                    'text': "[CHUNK_FAILED]",
                    'chunk_index': chunk['index']
                })
        
        if failed_chunks:
            print(f"⚠️ Some chunks failed: {failed_chunks}")
        
        # Use existing merge logic
        return self._merge_chunk_transcriptions(transcriptions)
    
    def _process_streaming_transcription(self):
        """Process streaming transcription results."""
        transcription_successful = False
        
        try:
            print("🔄 Processing streaming transcription results...")
            
            # Wait for all background transcriptions to complete
            self._wait_for_background_transcriptions()
            
            # Merge all streaming chunks
            full_transcription = self._merge_streaming_chunks()
            
            if full_transcription and full_transcription.strip():
                # Clean up any chunk failure markers if we have mostly good content
                if "[CHUNK_FAILED]" in full_transcription:
                    failed_count = full_transcription.count("[CHUNK_FAILED]")
                    total_chunks = len(self.streaming_chunks)
                    
                    if failed_count < total_chunks / 2:  # Less than half failed
                        print(f"⚠️ {failed_count}/{total_chunks} chunks failed, but continuing with partial transcription")
                        # Replace failed markers with pauses
                        full_transcription = full_transcription.replace("[CHUNK_FAILED]", "[...]")
                    else:
                        print(f"❌ Too many chunks failed ({failed_count}/{total_chunks}), falling back to traditional transcription")
                        return self._process_transcription()
                
                print(f"✅ Streaming transcription complete: '{full_transcription[:100]}{'...' if len(full_transcription) > 100 else ''}'")
                
                # Save to history and insert text
                self._save_recording_to_history(full_transcription, None)
                self._insert_text(full_transcription)
                
                transcription_successful = True
                
            else:
                print("❌ Streaming transcription produced no results, falling back to traditional method")
                return self._process_transcription()
                
        except Exception as e:
            print(f"❌ Streaming transcription error: {e}")
            print("🔄 Falling back to traditional transcription...")
            return self._process_transcription()
        
        finally:
            # Clean up based on success
            if transcription_successful:
                self._delete_temp_file()
                print("🚀 Streaming transcription completed successfully!")
            else:
                self._cleanup_temp_file()
            
            # Reset streaming state
            with self.streaming_lock:
                self.streaming_chunks = []
                self.current_chunk_data = []
    
    def _is_large_file(self):
        """Check if the recording is considered 'large' and should be chunked."""
        duration = self._get_recording_duration()
        
        # Check file size if temp file exists
        file_size_mb = 0
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            file_size_mb = os.path.getsize(self.temp_audio_file) / (1024 * 1024)
        
        # Use configuration settings for thresholds
        duration_threshold = config.LARGE_FILE_SETTINGS['duration_threshold_seconds']
        size_threshold = config.LARGE_FILE_SETTINGS['size_threshold_mb']
        
        is_long = duration > duration_threshold
        is_big = file_size_mb > size_threshold
        
        if is_long or is_big:
            print(f"📏 Large file detected: {duration:.1f}s, {file_size_mb:.1f}MB")
            return True
        
        return False
    
    def _split_audio_into_chunks(self):
        """Split the audio file into smaller chunks for processing."""
        chunk_duration_seconds = config.LARGE_FILE_SETTINGS['chunk_duration_seconds']
        overlap_seconds = config.LARGE_FILE_SETTINGS['chunk_overlap_seconds']
        
        try:
            import wave
            import numpy as np
            
            print(f"✂️ Splitting audio into {chunk_duration_seconds}s chunks...")
            
            # Read the original audio file
            with wave.open(self.temp_audio_file, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                total_frames = wav_file.getnframes()
                
                # Read all audio data
                audio_data = wav_file.readframes(total_frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            chunk_frames = int(chunk_duration_seconds * sample_rate)
            overlap_frames = int(overlap_seconds * sample_rate)
            
            chunks = []
            start = 0
            chunk_index = 0
            
            while start < len(audio_array):
                end = min(start + chunk_frames, len(audio_array))
                chunk_data = audio_array[start:end]
                
                # Create chunk filename
                base_name = os.path.splitext(self.temp_audio_file)[0]
                chunk_filename = f"{base_name}_chunk_{chunk_index:03d}.wav"
                
                # Save chunk to file
                with wave.open(chunk_filename, 'wb') as chunk_file:
                    chunk_file.setnchannels(channels)
                    chunk_file.setsampwidth(sample_width)
                    chunk_file.setframerate(sample_rate)
                    chunk_file.writeframes(chunk_data.tobytes())
                
                chunk_duration = len(chunk_data) / sample_rate
                chunks.append({
                    'filename': chunk_filename,
                    'duration': chunk_duration,
                    'start_time': start / sample_rate
                })
                
                print(f"📄 Created chunk {chunk_index + 1}: {chunk_duration:.1f}s")
                
                # Move start position (with overlap for context)
                start = end - overlap_frames if end < len(audio_array) else end
                chunk_index += 1
            
            print(f"✅ Split into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"❌ Error splitting audio: {e}")
            return []
    
    def _process_large_transcription(self):
        """Process large audio files by chunking them."""
        transcription_successful = False
        chunk_files = []
        
        try:
            print("🔄 Processing large file with chunking...")
            
            # Split the audio into chunks
            chunks = self._split_audio_into_chunks()
            
            if not chunks:
                print("❌ Failed to split audio. Falling back to normal processing.")
                return self._process_transcription()
            
            # Process each chunk
            transcriptions = []
            total_chunks = len(chunks)
            
            for i, chunk_info in enumerate(chunks):
                chunk_file = chunk_info['filename']
                chunk_files.append(chunk_file)  # Track for cleanup
                
                print(f"🎵 Processing chunk {i + 1}/{total_chunks} ({chunk_info['duration']:.1f}s)...")
                
                try:
                    # Transcribe this chunk using our retry logic
                    temp_audio_backup = self.temp_audio_file
                    self.temp_audio_file = chunk_file  # Temporarily use chunk file
                    
                    response = self._call_openai_with_retry()
                    
                    # Restore original temp file path
                    self.temp_audio_file = temp_audio_backup
                    
                    # Extract transcription text
                    chunk_text = response.strip() if isinstance(response, str) else response.text.strip()
                    
                    transcriptions.append({
                        'text': chunk_text,
                        'start_time': chunk_info['start_time'],
                        'chunk_index': i
                    })
                    
                    print(f"✅ Chunk {i + 1}/{total_chunks} completed: '{chunk_text[:50]}{'...' if len(chunk_text) > 50 else ''}'")
                    
                except Exception as e:
                    print(f"❌ Chunk {i + 1} failed: {e}")
                    transcriptions.append({
                        'text': "[CHUNK_FAILED]",
                        'start_time': chunk_info['start_time'],
                        'chunk_index': i
                    })
            
            # Combine transcriptions
            full_transcription = self._merge_chunk_transcriptions(transcriptions)
            
            if full_transcription and "[CHUNK_FAILED]" not in full_transcription:
                # Apply vocabulary corrections
                corrected_transcription = self._apply_vocabulary_corrections(full_transcription)
                
                if corrected_transcription != full_transcription:
                    print(f"🔧 Applied vocabulary corrections")
                
                print(f"✅ Complete transcription: '{corrected_transcription[:100]}{'...' if len(corrected_transcription) > 100 else ''}'")
                
                # Save to history and insert text
                self._save_recording_to_history(corrected_transcription, None)
                self._insert_text(corrected_transcription)
                
                transcription_successful = True
            else:
                print("❌ Large file transcription failed or incomplete")
                
        except Exception as e:
            print(f"❌ Large transcription processing error: {e}")
        
        finally:
            # Clean up chunk files
            for chunk_file in chunk_files:
                try:
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
                except Exception as e:
                    print(f"⚠️ Could not delete chunk file {chunk_file}: {e}")
            
            # Handle original file based on success
            if transcription_successful:
                self._delete_temp_file()
            else:
                self._cleanup_temp_file()
    
    def _merge_chunk_transcriptions(self, transcriptions):
        """Merge chunk transcriptions with basic overlap handling."""
        if not transcriptions:
            return ""
        
        # Sort by chunk index to ensure correct order
        transcriptions.sort(key=lambda x: x['chunk_index'])
        
        merged_parts = []
        
        for i, trans_info in enumerate(transcriptions):
            text = trans_info['text'].strip()
            
            if text == "[CHUNK_FAILED]":
                merged_parts.append("[...]")  # Placeholder for failed chunks
                continue
            
            if not text:  # Skip empty transcriptions
                continue
            
            # Basic overlap handling: remove duplicate words at boundaries
            if i > 0 and merged_parts and merged_parts[-1] != "[...]":
                # Get last few words from previous chunk
                prev_text = merged_parts[-1]
                prev_words = prev_text.split()[-5:]  # Last 5 words
                current_words = text.split()
                
                # Look for overlap in first few words of current chunk
                overlap_found = 0
                for j in range(min(5, len(current_words), len(prev_words))):
                    if current_words[j].lower() == prev_words[-(j+1)].lower():
                        overlap_found = j + 1
                    else:
                        break
                
                # Remove overlapping words from current text
                if overlap_found > 0:
                    text = ' '.join(current_words[overlap_found:])
                    print(f"🔗 Removed {overlap_found} overlapping word(s) from chunk {i + 1}")
            
            merged_parts.append(text)
        
        # Join all parts with spaces
        result = ' '.join(part for part in merged_parts if part.strip())
        
        # Clean up multiple spaces
        import re
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def _process_transcription(self):
        """Send audio to OpenAI Whisper API and handle the response."""
        transcription_successful = False
        
        try:
            print("🤖 Transcribing audio...")
            
            # Call OpenAI API with retry logic
            response = self._call_openai_with_retry()
            
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
                
                # Mark as successful - we got good text and inserted it
                transcription_successful = True
                
            else:
                print("⚠️ No speech detected in recording.")
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
        
        finally:
            # Clean up based on success or failure
            if transcription_successful:
                self._delete_temp_file()  # Delete file after success
            else:
                self._cleanup_temp_file()  # Move to failed folder for retry
    
    def _insert_text(self, text: str):
        """Insert text at current cursor position."""
        try:
            # Copy text to clipboard
            pyperclip.copy(text)
            print("📋 Text copied to clipboard")
            
            # Small delay to ensure clipboard is ready
            time.sleep(0.2)
            
            # Use pynput to simulate Cmd+V more reliably on Mac
            keyboard_controller = keyboard.Controller()
            
            # Press Cmd+V to paste
            with keyboard_controller.pressed(Key.cmd):
                keyboard_controller.press('v')
                keyboard_controller.release('v')
            
            print("✅ Text pasted successfully!")
            print(f"📝 Transcribed: '{text}'")
            
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
        """Move temporary audio file to failed folder for potential retry."""
        try:
            if self.temp_audio_file and os.path.exists(self.temp_audio_file):
                self._move_to_failed_folder()
                self.temp_audio_file = None
        except Exception as e:
            print(f"⚠️ Warning: Could not move temp file to failed folder: {e}")
    
    def _delete_temp_file(self):
        """Actually delete temporary audio file after successful transcription."""
        try:
            if self.temp_audio_file and os.path.exists(self.temp_audio_file):
                os.remove(self.temp_audio_file)
                self.temp_audio_file = None
                print("🗑️ Recording file deleted after successful transcription")
        except Exception as e:
            print(f"⚠️ Warning: Could not delete temp file: {e}")
    
    def _move_to_failed_folder(self):
        """Move recording to failed folder for potential retry."""
        if not self.temp_audio_file or not os.path.exists(self.temp_audio_file):
            return
        
        # Create failed directory if it doesn't exist
        failed_dir = os.path.join(config.TEMP_DIR, 'failed')
        os.makedirs(failed_dir, exist_ok=True)
        
        # Generate new filename with timestamp for the failed folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = os.path.basename(self.temp_audio_file)
        failed_filename = f"FAILED_{timestamp}_{original_filename}"
        failed_path = os.path.join(failed_dir, failed_filename)
        
        # Move the file
        import shutil
        shutil.move(self.temp_audio_file, failed_path)
        print(f"💾 Recording moved to failed folder: {failed_filename}")
        print(f"🔄 You can retry this recording later using backup.py")
    
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
            
        print("🔒 LOCKED RECORDING STARTED - Press Cmd+Option+Shift again to stop")
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
        """Start listening for global hotkeys using pynput."""
        try:
            print("👂 Listening for hotkeys...")
            print("🛑 To exit: Press Ctrl+C in this terminal window")
            
            # Track pressed keys
            self.pressed_keys = set()
            
            def on_press(key):
                """Handle key press events."""
                try:
                    # Only track our specific modifier keys for privacy
                    key_name = None
                    if str(key) == 'Key.cmd':
                        key_name = 'cmd'
                    elif str(key) == 'Key.alt' or str(key) == 'Key.alt_l' or str(key) == 'Key.alt_r':
                        key_name = 'alt'
                    elif str(key) == 'Key.shift' or str(key) == 'Key.shift_l' or str(key) == 'Key.shift_r':
                        key_name = 'shift'
                    
                    # Only add modifier keys to our tracking (no regular keys for privacy)
                    if key_name:
                        self.pressed_keys.add(key_name)
                    
                    # Check for our hotkey combinations
                    if self._check_hotkey_combination():
                        return  # Combination detected, handled in _check_hotkey_combination
                        
                except Exception as e:
                    print(f"⚠️ Key press error: {e}")
            
            def on_release(key):
                """Handle key release events."""
                try:
                    # Only track our specific modifier keys for privacy
                    key_name = None
                    if str(key) == 'Key.cmd':
                        key_name = 'cmd'
                    elif str(key) == 'Key.alt' or str(key) == 'Key.alt_l' or str(key) == 'Key.alt_r':
                        key_name = 'alt'
                    elif str(key) == 'Key.shift' or str(key) == 'Key.shift_l' or str(key) == 'Key.shift_r':
                        key_name = 'shift'
                    
                    # Only remove modifier keys from our tracking (no regular keys for privacy)
                    if key_name:
                        self.pressed_keys.discard(key_name)
                    
                    # Check if we should stop normal recording
                    if self.is_recording and not self.is_locked_recording:
                        if not self._is_normal_hotkey_pressed():
                            self._on_hotkey_release()
                    
                    # Handle exit condition (Ctrl+C handled separately)
                    pass
                        
                except Exception as e:
                    print(f"⚠️ Key release error: {e}")
            
            # Start the listener
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                print("✅ Hotkey listener started successfully!")
                print("💡 Note: You may need to grant accessibility permissions to Terminal/Python")
                listener.join()
                
        except KeyboardInterrupt:
            print("\n👋 Exiting Voice Dictation Tool...")
        except Exception as e:
            print(f"❌ Hotkey listener error: {e}")
            print("💡 Try granting accessibility permissions to Terminal in System Settings > Privacy & Security > Accessibility")
        finally:
            # Cleanup
            self._cleanup_temp_file()
    
    def _check_hotkey_combination(self):
        """Check if any of our hotkey combinations are pressed."""
        # Map pynput key names to our config - using actual macOS key names
        key_mapping = {
            'cmd': 'cmd',      # Command key
            'option': 'alt',   # Option/Alt key  
            'shift': 'shift'   # Shift key
        }
        
        # Check for locked recording toggle (Cmd+Option+Shift)
        locked_combo = all(key_mapping.get(k, k) in self.pressed_keys for k in config.HOTKEY_LOCKED)
        if locked_combo and not getattr(self, '_locked_combo_processed', False):
            self._locked_combo_processed = True
            # If we're already recording normally, transition to locked
            if self.is_recording and not self.is_locked_recording:
                self._transition_to_locked_recording()
            else:
                self._toggle_locked_recording()
            return True
        elif not locked_combo:
            self._locked_combo_processed = False
        
        # Check for normal recording (Cmd+Option, but NOT Shift)
        normal_combo = (all(key_mapping.get(k, k) in self.pressed_keys for k in config.HOTKEY_NORMAL) 
                       and 'shift' not in self.pressed_keys)
        if normal_combo and not self.is_recording:
            self._on_hotkey_press()
            return True
            
        return False
    
    def _is_normal_hotkey_pressed(self):
        """Check if the normal hotkey combination is still pressed."""
        key_mapping = {
            'cmd': 'cmd',
            'option': 'alt'
        }
        return all(key_mapping.get(k, k) in self.pressed_keys for k in config.HOTKEY_NORMAL)

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