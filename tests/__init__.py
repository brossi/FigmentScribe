"""
Scribe Handwriting Synthesis Test Suite

This package contains comprehensive tests for the Scribe handwriting synthesis
neural network implementation.

Test Categories:
- unit: Fast, isolated component tests
- integration: Multi-component workflow tests
- property: Property-based tests using Hypothesis
- regression: Golden output and performance regression tests

Usage:
    # Run all tests
    pytest

    # Run only unit tests
    pytest tests/unit

    # Run with coverage
    pytest --cov=. --cov-report=html

    # Run specific marker
    pytest -m unit
    pytest -m smoke

    # Run in parallel
    pytest -n auto
"""
