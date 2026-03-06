"""FlowW2N models package.

Conditioning modules (ContentEncoder, SpeakerEncoder) use lazy imports
to avoid speechbrain/torchaudio compatibility issues at import time.
"""

__all__ = [
    "AudioAutoencoder",
    "ContentEncoder",
    "DiffusionTransformer",
    "FlowW2NModel",
    "SpeakerEncoder",
]


def __getattr__(name):
    if name == "AudioAutoencoder":
        from .vae import AudioAutoencoder

        return AudioAutoencoder
    if name == "ContentEncoder":
        from .content_encoder import ContentEncoder

        return ContentEncoder
    if name == "DiffusionTransformer":
        from .dit import DiffusionTransformer

        return DiffusionTransformer
    if name == "FlowW2NModel":
        from .floww2n import FlowW2NModel

        return FlowW2NModel
    if name == "SpeakerEncoder":
        from .speaker_encoder import SpeakerEncoder

        return SpeakerEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
