# 🎙️ Local Voice Dictation Tools

# Tatoscription 

A lightweight Python-based voice dictation tool that uses OpenAI's transcription API to transcribe speech and automatically paste text anywhere on your system.

## ✨ Features

- **Global Hotkey Support**: Press and hold `Ctrl+Alt` from any application, you can also press `Ctrl+Alt+Shit` to start recording without needing to hold the buttons. Press `Ctrl+Alt+Shit` to deactive again. 
- **Real-time Voice Recording**: Captures high-quality audio from your microphone
- **OpenAI Whisper Integration**: Accurate speech-to-text transcription with custom vocabulary
- **Universal Text Insertion**: Works in any application (browsers, editors, documents)
- **Custom Vocabulary**: Pre-configured for tech terms (n8n, Retell, Vapi, etc.) with auto-correction
- **Configurable Recording Length**: Set your own max recording time in the .env file (1-5+ minutes)
- **Recording History**: Saves all transcriptions with timestamps to JSON file
- **Privacy-Focused**: All processing happens locally except for the API call

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (tested with Python 3.12.3)
- Windows 11 (primary support)
- OpenAI API key
- Working microphone

### Installation

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create environment file:**
   Create a `.env` file in the project root with your settings:
   ```env
   OPENAI_API_KEY=your-openai-api-key-here
   MAX_RECORDING_MINUTES=2
   ```

4. **Run the tool:**
   ```bash
   python start.py
   ```

### Settings (config.py)

You can modify these settings in `config.py`:

- **Recording quality**: Sample rate, channels, audio format
- **Hotkey combination**: Default is `ctrl+alt`
- **API model**: Whisper model to use
- **Custom vocabulary**: Add your own technical terms and corrections
- **File locations**: Temp directory, recordings file

### Custom Vocabulary

The tool comes pre-loaded with technical terms and can auto-correct common mistakes:

**Pre-configured terms**: `n8n`, `Retell`, `Vapi`, `OpenAI`, `Supabase`, `Cursor`, `GitHub`, `JavaScript`, `TypeScript`, `React`, etc.

**Adding your own terms**:
1. Run the helper: `python add_vocabulary.py`
2. Or manually edit `CUSTOM_VOCABULARY` in `config.py`

**Example**: If you say "n8n" but Whisper hears "engine", it automatically corrects to "n8n"!

## 🔧 Troubleshooting

### Common Issues

**1. "OPENAI_API_KEY not found"**
- Create a `.env` file with your API key
- Ensure the file is in the same directory as `voice_transcribe.py`

**2. "No microphone detected"**
- Check Windows microphone permissions
- Ensure your microphone is set as the default recording device
- Test microphone in other applications

**3. "Module not found" errors**
- Run: `pip install -r requirements.txt`
- Ensure you're using the correct Python environment

**4. Recording doesn't start**
- Run as administrator (required for global hotkeys)
- Check if another application is using the microphone
- Verify hotkey combination isn't conflicting with other software

**5. Text doesn't paste automatically**
- Text is always copied to clipboard as backup
- Manually paste with `Ctrl+V` if auto-paste fails
- Check target application allows programmatic pasting

## 🛡️ Privacy & Security

- **Audio data**: Temporarily stored locally, automatically deleted after processing
- **Transcriptions**: Saved locally in `recordings.json`
- **API calls**: Only audio is sent to OpenAI for transcription, but this is safe from any data being used for training purposes too
- **No cloud storage**: All data remains on your local machine

## 📋 Requirements

### Python Packages

- `keyboard`: Global hotkey detection
- `sounddevice`: Audio recording
- `numpy`: Lightweight audio processing (replaces heavy SciPy)
- `openai`: Whisper API integration
- `pyperclip`: Clipboard operations
- `pyautogui`: Automated key presses
- `python-dotenv`: Environment variable loading


## 📜 License

This project is for personal and educational use. 

