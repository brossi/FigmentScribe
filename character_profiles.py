"""
Character Profile Templates for Fictional Character Handwriting

This file provides examples of how to define consistent handwriting styles
for fictional characters using the multi-line handwriting synthesis system.

Usage:
    python3 sample.py \
        --lines "${CHARACTERS['character_a']['sample_text']}" \
                "${CHARACTERS['character_b']['sample_text']}" \
        --biases ${CHARACTERS['character_a']['bias']} \
                 ${CHARACTERS['character_b']['bias']} \
        --format svg
"""

# Character handwriting profiles
CHARACTERS = {
    'character_a': {
        'name': 'Character A',
        'bias': 1.8,
        'style': 9,  # For future style priming implementation
        'description': 'Neat, precise handwriting - military background',
        'personality': 'Disciplined, formal, meticulous',
        'sample_text': 'The mission parameters are clearly defined.'
    },

    'character_b': {
        'name': 'Character B',
        'bias': 0.6,
        'style': 12,
        'description': 'Messy, emotional handwriting - artist/creative type',
        'personality': 'Impulsive, passionate, expressive',
        'sample_text': 'cant believe what happened today!!!'
    },

    'character_c': {
        'name': 'Character C',
        'bias': 1.2,
        'style': 7,
        'description': 'Balanced, natural handwriting - everyday person',
        'personality': 'Practical, straightforward, reliable',
        'sample_text': 'Just wanted to let you know I got your message.'
    },
}

# Bias guidelines
BIAS_GUIDE = {
    'very_messy': 0.5,      # Highly variable, emotional, rushed
    'messy': 0.7,           # Casual, informal, quick
    'casual': 1.0,          # Natural variation, everyday writing
    'neat': 1.3,            # Controlled, careful, legible
    'very_neat': 1.6,       # Precise, formal, deliberate
    'calligraphic': 2.0,    # Almost mechanical, very controlled
}

# Example usage functions
def get_character_args(character_id):
    """
    Get command-line arguments for a specific character.

    Args:
        character_id: Key from CHARACTERS dict

    Returns:
        dict: Arguments for sample.py
    """
    char = CHARACTERS.get(character_id)
    if not char:
        raise ValueError(f"Unknown character: {character_id}")

    return {
        'bias': char['bias'],
        'style': char['style'],  # For future use
    }


def generate_document_for_characters(character_lines):
    """
    Generate a multi-character document.

    Args:
        character_lines: List of (character_id, text) tuples

    Example:
        document = [
            ('character_a', 'Official report: Mission successful.'),
            ('character_b', 'Wow! That was incredible!!!'),
            ('character_c', 'I confirm the events as described.'),
        ]
        generate_document_for_characters(document)
    """
    lines = []
    biases = []

    for char_id, text in character_lines:
        char = CHARACTERS[char_id]
        lines.append(text)
        biases.append(char['bias'])

    print("# Generate this document with:")
    print(f"python3 sample.py \\")
    print(f"    --lines {' '.join(repr(line) for line in lines)} \\")
    print(f"    --biases {' '.join(str(b) for b in biases)} \\")
    print(f"    --format svg")

    return lines, biases


if __name__ == '__main__':
    # Example: Generate a three-character conversation
    print("=" * 70)
    print("CHARACTER PROFILE EXAMPLE")
    print("=" * 70)

    document = [
        ('character_a', 'The operation will commence at 0600 hours.'),
        ('character_b', 'This is so exciting I can hardly wait!'),
        ('character_c', 'Understood. I will be ready.'),
    ]

    print("\nCharacter profiles:")
    for char_id, text in document:
        char = CHARACTERS[char_id]
        print(f"\n{char['name']} (bias={char['bias']}):")
        print(f"  Personality: {char['personality']}")
        print(f"  Text: \"{text}\"")

    print("\n" + "=" * 70)
    print("COMMAND TO GENERATE:")
    print("=" * 70 + "\n")

    generate_document_for_characters(document)
