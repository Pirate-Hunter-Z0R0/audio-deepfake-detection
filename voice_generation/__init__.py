"""
voice_generation package
------------------------
Integrates Applio RVC voice conversion and Edge-TTS into the Audio Deepfake
Detection project to generate synthetic speech test benchmarks.
"""

from .applio_wrapper import ApplioWrapper

_default_wrapper = None

def _get_wrapper() -> ApplioWrapper:
    global _default_wrapper
    if _default_wrapper is None:
        _default_wrapper = ApplioWrapper()
    return _default_wrapper

def text_to_rvc(*args, **kwargs):
    return _get_wrapper().text_to_rvc(*args, **kwargs)

def convert_audio(*args, **kwargs):
    return _get_wrapper().convert_audio(*args, **kwargs)

def validate_model(*args, **kwargs):
    return _get_wrapper().validate_model(*args, **kwargs)

__all__ = ["ApplioWrapper", "text_to_rvc", "convert_audio", "validate_model"]
