#!/usr/bin/env python3
"""
Backup transcription script for failed recordings.
Run this independently to manually retry failed transcriptions.

Usage: python backup.py
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import config

# Load environment variables
load_dotenv()

def get_openai_client():
    """Initialize OpenAI client."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your-api-key-here")
        sys.exit(1)
    return OpenAI(api_key=api_key)

def get_failed_recordings():
    """Get list of failed recording files."""
    failed_dir = os.path.join(config.TEMP_DIR, 'failed')
    
    if not os.path.exists(failed_dir):
        print(f"❌ No failed recordings directory found at: {failed_dir}")
        return []
    
    files = []
    for filename in os.listdir(failed_dir):
        if filename.endswith('.wav') and filename.startswith('FAILED_'):
            filepath = os.path.join(failed_dir, filename)
            # Get file creation time
            creation_time = os.path.getctime(filepath)
            files.append({
                'filename': filename,
                'filepath': filepath,
                'created': datetime.fromtimestamp(creation_time),
                'size_mb': os.path.getsize(filepath) / (1024 * 1024)
            })
    
    # Sort by creation time (newest first)
    files.sort(key=lambda x: x['created'], reverse=True)
    return files

def apply_vocabulary_corrections(text):
    """Apply the same vocabulary corrections as the main app."""
    corrected_text = text
    
    # Apply case-insensitive corrections
    for correct_term, alternatives in config.CUSTOM_VOCABULARY.items():
        for alternative in alternatives:
            # Case-insensitive replacement while preserving surrounding text
            import re
            pattern = r'\b' + re.escape(alternative) + r'\b'
            corrected_text = re.sub(pattern, correct_term, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text

def is_large_file(filepath):
    """Check if file should be processed with chunking."""
    import wave
    
    try:
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        with wave.open(filepath, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            total_frames = wav_file.getnframes()
            duration = total_frames / sample_rate
        
        is_long = duration > 120  # 2 minutes
        is_big = file_size_mb > 10  # 10MB
        
        if is_long or is_big:
            print(f"📏 Large file detected: {duration:.1f}s, {file_size_mb:.1f}MB")
            return True, duration, file_size_mb
        
        return False, duration, file_size_mb
        
    except Exception as e:
        print(f"⚠️ Could not analyze file: {e}")
        return False, 0, 0

def transcribe_file(client, filepath):
    """Transcribe a single audio file with chunking support."""
    try:
        print(f"🤖 Transcribing: {os.path.basename(filepath)}")
        
        # Check if file needs chunking
        needs_chunking, duration, file_size = is_large_file(filepath)
        
        if needs_chunking:
            print(f"🔄 Large file will be processed in chunks...")
            return transcribe_large_file(client, filepath)
        else:
            print(f"📄 Processing normally ({duration:.1f}s, {file_size:.1f}MB)")
            return transcribe_single_file(client, filepath)
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None

def transcribe_single_file(client, filepath):
    """Transcribe a single file without chunking."""
    with open(filepath, 'rb') as audio_file:
        response = client.audio.transcriptions.create(
            model=config.API_SETTINGS['model'],
            file=audio_file,
            response_format=config.API_SETTINGS['response_format'],
            prompt=config.API_SETTINGS['prompt']
        )
    
    # Extract and correct transcription
    transcription = response.strip() if isinstance(response, str) else response.text.strip()
    corrected_transcription = apply_vocabulary_corrections(transcription)
    
    return corrected_transcription

def transcribe_large_file(client, filepath):
    """Transcribe a large file using chunking."""
    import wave
    import numpy as np
    
    chunk_files = []
    
    try:
        # Split into chunks
        chunks = split_audio_file(filepath)
        if not chunks:
            print("❌ Failed to split file. Trying single file approach...")
            return transcribe_single_file(client, filepath)
        
        # Process each chunk
        transcriptions = []
        total_chunks = len(chunks)
        
        for i, chunk_info in enumerate(chunks):
            chunk_file = chunk_info['filename']
            chunk_files.append(chunk_file)
            
            print(f"🎵 Processing chunk {i + 1}/{total_chunks} ({chunk_info['duration']:.1f}s)...")
            
            try:
                chunk_text = transcribe_single_file(client, chunk_file)
                transcriptions.append({
                    'text': chunk_text if chunk_text else "[CHUNK_FAILED]",
                    'chunk_index': i
                })
                print(f"✅ Chunk {i + 1} completed")
                
            except Exception as e:
                print(f"❌ Chunk {i + 1} failed: {e}")
                transcriptions.append({
                    'text': "[CHUNK_FAILED]",
                    'chunk_index': i
                })
        
        # Merge chunks
        merged_text = merge_transcriptions(transcriptions)
        
        if "[CHUNK_FAILED]" in merged_text:
            print("⚠️ Some chunks failed - transcription may be incomplete")
        
        return merged_text
        
    finally:
        # Clean up chunk files
        for chunk_file in chunk_files:
            try:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            except Exception as e:
                print(f"⚠️ Could not delete chunk file: {e}")

def split_audio_file(filepath):
    """Split audio file into chunks."""
    import wave
    import numpy as np
    
    chunk_duration_seconds = 90
    overlap_seconds = 5
    
    try:
        with wave.open(filepath, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            total_frames = wav_file.getnframes()
            
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
            base_name = os.path.splitext(filepath)[0]
            chunk_filename = f"{base_name}_backup_chunk_{chunk_index:03d}.wav"
            
            # Save chunk
            with wave.open(chunk_filename, 'wb') as chunk_file:
                chunk_file.setnchannels(channels)
                chunk_file.setsampwidth(sample_width)
                chunk_file.setframerate(sample_rate)
                chunk_file.writeframes(chunk_data.tobytes())
            
            chunk_duration = len(chunk_data) / sample_rate
            chunks.append({
                'filename': chunk_filename,
                'duration': chunk_duration
            })
            
            start = end - overlap_frames if end < len(audio_array) else end
            chunk_index += 1
        
        print(f"✂️ Split into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        print(f"❌ Error splitting audio: {e}")
        return []

def merge_transcriptions(transcriptions):
    """Merge chunk transcriptions with overlap handling."""
    if not transcriptions:
        return ""
    
    # Sort by chunk index
    transcriptions.sort(key=lambda x: x['chunk_index'])
    
    merged_parts = []
    
    for i, trans_info in enumerate(transcriptions):
        text = trans_info['text'].strip()
        
        if text == "[CHUNK_FAILED]":
            merged_parts.append("[...]")
            continue
        
        if not text:
            continue
        
        # Basic overlap removal
        if i > 0 and merged_parts and merged_parts[-1] != "[...]":
            prev_words = merged_parts[-1].split()[-5:]
            current_words = text.split()
            
            overlap_found = 0
            for j in range(min(5, len(current_words), len(prev_words))):
                if current_words[j].lower() == prev_words[-(j+1)].lower():
                    overlap_found = j + 1
                else:
                    break
            
            if overlap_found > 0:
                text = ' '.join(current_words[overlap_found:])
        
        merged_parts.append(text)
    
    result = ' '.join(part for part in merged_parts if part.strip())
    
    # Clean up spaces
    import re
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def main():
    """Main backup script logic."""
    print("🔄 Backup Transcription Tool")
    print("=" * 40)
    
    # Get failed recordings
    failed_files = get_failed_recordings()
    
    if not failed_files:
        print("✅ No failed recordings found! All transcriptions were successful.")
        return
    
    # Display available files
    print(f"📁 Found {len(failed_files)} failed recording(s):")
    print()
    
    for i, file_info in enumerate(failed_files, 1):
        print(f"{i:2}. {file_info['filename']}")
        print(f"    Created: {file_info['created'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    Size: {file_info['size_mb']:.1f} MB")
        print()
    
    # Get user choice
    while True:
        try:
            choice = input(f"Select a file to transcribe (1-{len(failed_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("👋 Goodbye!")
                return
            
            file_index = int(choice) - 1
            if 0 <= file_index < len(failed_files):
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(failed_files)}")
                
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
    
    # Transcribe selected file
    selected_file = failed_files[file_index]
    print(f"\n🎵 Processing: {selected_file['filename']}")
    print("-" * 40)
    
    # Initialize OpenAI client
    try:
        client = get_openai_client()
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        return
    
    # Perform transcription
    transcription = transcribe_file(client, selected_file['filepath'])
    
    if transcription:
        print("\n✅ Transcription successful!")
        print("=" * 40)
        print(transcription)
        print("=" * 40)
        
        # Ask if user wants to delete the failed file
        delete_choice = input("\n🗑️ Delete this failed recording? (y/n): ").strip().lower()
        if delete_choice == 'y':
            try:
                os.remove(selected_file['filepath'])
                print(f"✅ Deleted: {selected_file['filename']}")
            except Exception as e:
                print(f"❌ Failed to delete file: {e}")
        
        # Ask if user wants to copy to clipboard
        try:
            import pyperclip
            copy_choice = input("📋 Copy transcription to clipboard? (y/n): ").strip().lower()
            if copy_choice == 'y':
                pyperclip.copy(transcription)
                print("✅ Copied to clipboard!")
        except ImportError:
            print("ℹ️ pyperclip not available - transcription not copied to clipboard")
        
    else:
        print("\n❌ Transcription failed. File will remain in failed folder for another attempt.")

if __name__ == "__main__":
    main()