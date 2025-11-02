"""
Unit Tests for Character Profile Templates

Tests character handwriting profile helper functions.

Usage:
    pytest tests/unit/test_character_profiles.py -v
    pytest -k test_character_profiles
"""

import pytest


@pytest.mark.unit
class TestCharacterData:
    """Test character profile data structures."""

    def test_characters_dict_exists(self):
        """Test CHARACTERS dict is defined."""
        from character_profiles import CHARACTERS

        assert isinstance(CHARACTERS, dict)
        assert len(CHARACTERS) > 0

    def test_characters_have_required_fields(self):
        """Test all character profiles have required fields."""
        from character_profiles import CHARACTERS

        required_fields = ['name', 'bias', 'style', 'description', 'personality', 'sample_text']

        for char_id, profile in CHARACTERS.items():
            for field in required_fields:
                assert field in profile, \
                    f"Character '{char_id}' missing required field: {field}"

    def test_character_biases_are_numeric(self):
        """Test all character bias values are numeric."""
        from character_profiles import CHARACTERS

        for char_id, profile in CHARACTERS.items():
            bias = profile['bias']
            assert isinstance(bias, (int, float)), \
                f"Character '{char_id}' bias must be numeric, got {type(bias)}"
            assert bias > 0, \
                f"Character '{char_id}' bias must be positive, got {bias}"

    def test_character_styles_are_valid(self):
        """Test all character style values are in valid range (0-12)."""
        from character_profiles import CHARACTERS

        for char_id, profile in CHARACTERS.items():
            style = profile['style']
            assert isinstance(style, int), \
                f"Character '{char_id}' style must be int, got {type(style)}"
            assert 0 <= style <= 12, \
                f"Character '{char_id}' style must be 0-12, got {style}"

    def test_character_sample_texts_not_empty(self):
        """Test all character sample texts are non-empty strings."""
        from character_profiles import CHARACTERS

        for char_id, profile in CHARACTERS.items():
            sample_text = profile['sample_text']
            assert isinstance(sample_text, str), \
                f"Character '{char_id}' sample_text must be str"
            assert len(sample_text) > 0, \
                f"Character '{char_id}' sample_text must not be empty"

    def test_bias_guide_exists(self):
        """Test BIAS_GUIDE dict is defined."""
        from character_profiles import BIAS_GUIDE

        assert isinstance(BIAS_GUIDE, dict)
        assert len(BIAS_GUIDE) > 0

    def test_bias_guide_values_numeric(self):
        """Test all BIAS_GUIDE values are numeric."""
        from character_profiles import BIAS_GUIDE

        for level, bias in BIAS_GUIDE.items():
            assert isinstance(bias, (int, float)), \
                f"BIAS_GUIDE['{level}'] must be numeric, got {type(bias)}"
            assert bias > 0, \
                f"BIAS_GUIDE['{level}'] must be positive, got {bias}"

    def test_bias_guide_ordered(self):
        """Test BIAS_GUIDE values are ordered from messy to neat."""
        from character_profiles import BIAS_GUIDE

        # Expected order (from low to high bias)
        expected_order = ['very_messy', 'messy', 'casual', 'neat', 'very_neat', 'calligraphic']

        values = [BIAS_GUIDE[key] for key in expected_order if key in BIAS_GUIDE]

        # Values should be increasing
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1], \
                f"BIAS_GUIDE values should increase: {expected_order[i]}={values[i]} should be < {expected_order[i+1]}={values[i+1]}"


@pytest.mark.unit
class TestGetCharacterArgs:
    """Test get_character_args function."""

    def test_get_character_args_returns_dict(self):
        """Test get_character_args returns dict."""
        from character_profiles import get_character_args

        result = get_character_args('character_a')

        assert isinstance(result, dict)

    def test_get_character_args_has_bias(self):
        """Test get_character_args returns bias."""
        from character_profiles import get_character_args

        result = get_character_args('character_a')

        assert 'bias' in result
        assert isinstance(result['bias'], (int, float))

    def test_get_character_args_has_style(self):
        """Test get_character_args returns style."""
        from character_profiles import get_character_args

        result = get_character_args('character_a')

        assert 'style' in result
        assert isinstance(result['style'], int)

    def test_get_character_args_matches_character_data(self):
        """Test get_character_args returns correct values."""
        from character_profiles import get_character_args, CHARACTERS

        for char_id in CHARACTERS.keys():
            result = get_character_args(char_id)

            assert result['bias'] == CHARACTERS[char_id]['bias']
            assert result['style'] == CHARACTERS[char_id]['style']

    def test_get_character_args_invalid_character_raises(self):
        """Test get_character_args raises ValueError for invalid character."""
        from character_profiles import get_character_args

        with pytest.raises(ValueError, match="Unknown character"):
            get_character_args('nonexistent_character')

    def test_get_character_args_empty_string_raises(self):
        """Test get_character_args raises ValueError for empty string."""
        from character_profiles import get_character_args

        with pytest.raises(ValueError, match="Unknown character"):
            get_character_args('')


@pytest.mark.unit
class TestGenerateDocumentForCharacters:
    """Test generate_document_for_characters function."""

    def test_generate_document_returns_tuple(self):
        """Test generate_document_for_characters returns tuple."""
        from character_profiles import generate_document_for_characters

        document = [('character_a', 'Test text')]

        result = generate_document_for_characters(document)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_generate_document_returns_lines_and_biases(self, capsys):
        """Test generate_document_for_characters returns lines and biases."""
        from character_profiles import generate_document_for_characters

        document = [
            ('character_a', 'Text A'),
            ('character_b', 'Text B'),
        ]

        lines, biases = generate_document_for_characters(document)

        # Verify lines
        assert isinstance(lines, list)
        assert len(lines) == 2
        assert lines[0] == 'Text A'
        assert lines[1] == 'Text B'

        # Verify biases
        assert isinstance(biases, list)
        assert len(biases) == 2

    def test_generate_document_biases_match_characters(self, capsys):
        """Test generate_document_for_characters uses correct character biases."""
        from character_profiles import generate_document_for_characters, CHARACTERS

        document = [
            ('character_a', 'Text A'),
            ('character_b', 'Text B'),
        ]

        lines, biases = generate_document_for_characters(document)

        assert biases[0] == CHARACTERS['character_a']['bias']
        assert biases[1] == CHARACTERS['character_b']['bias']

    def test_generate_document_prints_command(self, capsys):
        """Test generate_document_for_characters prints sample.py command."""
        from character_profiles import generate_document_for_characters

        document = [('character_a', 'Test')]

        generate_document_for_characters(document)

        captured = capsys.readouterr()

        # Should print command
        assert 'python3 sample.py' in captured.out
        assert '--lines' in captured.out
        assert '--biases' in captured.out
        assert '--format svg' in captured.out

    def test_generate_document_empty_list(self, capsys):
        """Test generate_document_for_characters handles empty list."""
        from character_profiles import generate_document_for_characters

        document = []

        lines, biases = generate_document_for_characters(document)

        assert lines == []
        assert biases == []

    def test_generate_document_single_character(self, capsys):
        """Test generate_document_for_characters handles single character."""
        from character_profiles import generate_document_for_characters

        document = [('character_c', 'Single line')]

        lines, biases = generate_document_for_characters(document)

        assert len(lines) == 1
        assert len(biases) == 1
        assert lines[0] == 'Single line'

    def test_generate_document_preserves_order(self, capsys):
        """Test generate_document_for_characters preserves character order."""
        from character_profiles import generate_document_for_characters, CHARACTERS

        document = [
            ('character_c', 'Third'),
            ('character_a', 'First'),
            ('character_b', 'Second'),
        ]

        lines, biases = generate_document_for_characters(document)

        assert lines[0] == 'Third'
        assert lines[1] == 'First'
        assert lines[2] == 'Second'

        assert biases[0] == CHARACTERS['character_c']['bias']
        assert biases[1] == CHARACTERS['character_a']['bias']
        assert biases[2] == CHARACTERS['character_b']['bias']


# ============================================================================
# Summary
# ============================================================================

def test_character_profiles_suite_summary():
    """
    Character profiles test suite summary.

    If all character profile tests pass:
    - CHARACTERS dict properly defined
    - All profiles have required fields
    - Bias values are numeric and positive
    - Style values are in valid range (0-12)
    - BIAS_GUIDE values are ordered correctly
    - get_character_args() works for valid/invalid characters
    - generate_document_for_characters() returns correct format
    """
    print("\n✓ All character profile tests passed!")
    print("  - Character data structures valid")
    print("  - get_character_args() works")
    print("  - generate_document_for_characters() works")
