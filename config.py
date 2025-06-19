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
    'model': 'gpt-4o-mini-transcribe',  # you can try wisper-1
    'response_format': 'text',
    'prompt': "The following is a transcript of a person talking, you can remove and duplicated words and any fillers words. If its a longer transcript put into paragraphs for better readability."
}

# Custom vocabulary for better recognition
CUSTOM_VOCABULARY = {
    # Technical tools and platforms
    'n8n': ['n8n', 'n 8 n', 'n eight n', 'nateon', 'AN10', 'N810', 'N8N', 'A10'],
    'Retell': ['Retell', 'retell', 're-tell', 'retail', 'retale', 're tell'],
    'Vapi': ['Vapi', 'vapi', 'VAPI', 'vapey', 'happy'],
    'OpenAI': ['OpenAI', 'open AI', 'open ai'],
    'Supabase': ['Supabase', 'super base', 'supa base'],
    'Cursor': ['Cursor', 'cursor'],
    'GitHub': ['GitHub', 'git hub', 'github'],
    'JSON': ['JSON', 'jason', 'jay son'], 
    'npm': ['npm', 'N pee m.', 'n p m'],
    'pip': ['pip', 'pyp.'],
    'Jannis':['Yanis', 'Janice'],
    'LiveKit':['LiveKit', 'live kit', 'LifeKit', 'livekit', 'life kit', 'life Git', 'live Git'],
  
    
    # Add your own custom terms here
    # 'YourTerm': ['YourTerm', 'alternative1', 'alternative2']
}

# Generate vocabulary hint for Whisper API
VOCABULARY_HINT = ', '.join(CUSTOM_VOCABULARY.keys())

# Hotkey configuration
HOTKEY = 'ctrl+alt' or 'ctrl+alt+shift'

# File settings
TEMP_FILE_PREFIX = 'voice_recording_'
RECORDINGS_FILE = 'recordings.json'
TEMP_DIR = 'temp'

# Audio settings
AUDIO_FORMAT = 'wav'
MIN_RECORDING_SECONDS = 1.0  # Minimum recording length to process 