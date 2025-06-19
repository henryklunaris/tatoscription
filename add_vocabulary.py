import re

def add_vocabulary_term():
    """Interactive script to add a new vocabulary term."""
    print("🔤 Add Custom Vocabulary Term")
    print("=" * 40)
    
    # Get the correct term
    correct_term = input("Enter the correct term (e.g., 'n8n'): ").strip()
    if not correct_term:
        print("❌ No term entered. Exiting.")
        return
    
    # Get alternatives
    print(f"\nNow enter alternative ways Whisper might transcribe '{correct_term}':")
    print("(Press Enter after each alternative, or empty line to finish)")
    
    alternatives = [correct_term]  # Include the correct term itself
    while True:
        alt = input(f"Alternative {len(alternatives)}: ").strip()
        if not alt:
            break
        alternatives.append(alt)
    
    if len(alternatives) == 1:
        print("⚠️ No alternatives added. Adding some common variations...")
        # Add some automatic alternatives
        alternatives.extend([
            correct_term.lower(),
            correct_term.upper(),
            correct_term.title()
        ])
    
    # Generate the config entry
    config_entry = f"    '{correct_term}': {alternatives},"
    
    print(f"\n✅ Generated config entry:")
    print(config_entry)
    
    print(f"\n📝 To add this to your vocabulary:")
    print("1. Open 'config.py'")
    print("2. Find the CUSTOM_VOCABULARY section")
    print("3. Add this line inside the dictionary:")
    print(f"   {config_entry}")
    print("4. Save the file")
    print("\n🎉 Then restart the voice dictation tool to use the new term!")

if __name__ == "__main__":
    add_vocabulary_term() 