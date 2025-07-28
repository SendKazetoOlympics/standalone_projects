"""
Tests for sam_handler.py

Tests the SAMHandler class and its methods.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from sportsam.sam_handler import SAMHandler, SAMModels

def test_handler_initialization_invalid():
    """Test SAMHandler initialization with an invalid model input."""
    with patch('sportsam.sam_handler.build_sam2_video_predictor', return_value=MagicMock()):
        with pytest.raises(ValueError):
            SAMHandler(frames_path='test_frames', model='INVALID_MODEL')

