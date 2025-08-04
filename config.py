"""
Configuration settings for the Voice Dictation Tool
"""

# Recording settings
RECORDING_SETTINGS = {
    'sample_rate': 44100,
    'channels': 1,
    'dtype': 'float64'
}

# OpenAI API settings
API_SETTINGS = {
    'model': 'gpt-4o-mini-transcribe',  # Fixed model name for OpenAI Whisper API
    'response_format': 'text',
    'prompt': "The following is a transcript of a person talking, you can remove and duplicated words and any fillers words. If its a longer transcript put into paragraphs for better readability."
}

# Custom vocabulary for better recognition
CUSTOM_VOCABULARY = {
    # Technical tools and platforms
    'n8n': ['n8n', 'n 8 n', 'n eight n', 'nateon', 'AN10', 'N810', 'N8N', 'A10', 'NA10'],
    'Retell': ['Retell', 'retell', 're-tell', 'retail', 'retale', 're tell'],
    'Vapi': ['Vapi', 'vapi', 'VAPI', 'vapey', 'happy'],
    'OpenAI': ['OpenAI', 'open AI', 'open ai'],
    'Supabase': ['Supabase', 'super base', 'supa base'],
    'Cursor': ['Cursor', 'cursor'],
    'GitHub': ['GitHub', 'git hub', 'github'],
    'JSON': ['JSON', 'jason', 'jay son'], 
    'npm': ['npm', 'N pee m.', 'n p m'],
    'pip': ['pip', 'pyp.'],
    'Henryk': ['Henryk', 'henryk', 'henry k', 'henrik', 'henrick', 'Hendryk'],
    'Jannis': ['Jannis', 'jannis', 'jannis', 'jannis', 'Yanis', 'Janice'],
    'LiveKit':['LiveKit', 'live kit', 'LifeKit', 'livekit', 'life kit', 'life Git', 'live Git'],
  
    
    # Add your own custom terms here
    # 'YourTerm': ['YourTerm', 'alternative1', 'alternative2']
}

# Generate vocabulary hint for Whisper API
VOCABULARY_HINT = ', '.join(CUSTOM_VOCABULARY.keys())

# Hotkey configuration - Mac-friendly combinations
HOTKEY_NORMAL = {'cmd', 'option'}  # Cmd+Option for normal recording
HOTKEY_LOCKED = {'cmd', 'option', 'shift'}  # Cmd+Option+Shift for locked recording

# File settings
TEMP_FILE_PREFIX = 'voice_recording_'
RECORDINGS_FILE = 'recordings.json'
TEMP_DIR = 'temp'

# Audio settings
AUDIO_FORMAT = 'wav'
MIN_RECORDING_SECONDS = 1.0  # Minimum recording length to process

# Streaming transcription settings
STREAMING_SETTINGS = {
    'enabled': True,  # Enable pseudo-streaming transcription
    'chunk_interval_seconds': 35,  # Send chunk every 90 seconds
    'max_concurrent_transcriptions': 3,  # Max parallel API calls
    'min_chunk_seconds': 0.5,  # Minimum chunk size to transcribe
    'overlap_seconds': 2,  # Overlap between streaming chunks
    'progress_feedback': True,  # Show real-time progress updates
}

# Large file detection (for fallback to post-processing chunking)
LARGE_FILE_SETTINGS = {
    'duration_threshold_seconds': 120,  # Consider large if over 2 minutes
    'size_threshold_mb': 10,  # Consider large if over 10MB
    'chunk_duration_seconds': 90,  # Chunk size for large files
    'chunk_overlap_seconds': 5,  # Overlap for large file chunks
} 