"""
PyTorch compatibility layer for DeOldify.

This module provides compatibility between different PyTorch versions (1.x through 2.11+)
by monkey-patching torch.load() to handle the weights_only parameter change introduced
in PyTorch 2.6, where the default changed from False to True.

This ensures that checkpoints saved with older PyTorch versions can still be loaded
without requiring users to explicitly pass weights_only=False.
"""

import inspect
import functools
import logging
import torch

logger = logging.getLogger(__name__)

# Store the original torch.load before any modifications
_original_torch_load = torch.load


def _register_safe_globals():
    """
    Register safe globals for PyTorch unpickling.

    PyTorch 2.3+ introduced add_safe_globals to control which types can be
    unpickled when weights_only=True. Since we need weights_only=False for
    backward compatibility with older checkpoints, we register common types
    to ensure smooth loading across versions.
    """
    if not hasattr(torch.serialization, 'add_safe_globals'):
        return

    safe_types = [
        functools.partial,
    ]

    for safe_type in safe_types:
        try:
            torch.serialization.add_safe_globals([safe_type])
        except Exception as e:
            logger.debug("Could not register safe global %s: %s", safe_type, e)


def compatible_torch_load(*args, **kwargs):
    """
    Compatibility wrapper for torch.load().

    Handles the weights_only parameter change introduced in PyTorch 2.6
    where the default changed from False to True.

    Args:
        *args: Positional arguments passed to torch.load()
        **kwargs: Keyword arguments passed to torch.load()

    Returns:
        The loaded object(s) from the checkpoint file.
    """
    if "weights_only" not in kwargs:
        try:
            sig = inspect.signature(_original_torch_load)
            if "weights_only" in sig.parameters:
                kwargs["weights_only"] = False
        except (ValueError, TypeError):
            pass

    return _original_torch_load(*args, **kwargs)


def safe_load_checkpoint(path, map_location=None, **kwargs):
    """
    Safely load a model checkpoint with human-readable error messages.

    Args:
        path: Path to the checkpoint file
        map_location: Optional device mapping for loading
        **kwargs: Additional arguments passed to torch.load()

    Returns:
        The loaded checkpoint state dict

    Raises:
        RuntimeError: With a human-readable message explaining what failed
    """
    load_kwargs = {}
    if map_location is not None:
        load_kwargs['map_location'] = map_location

    # Ensure weights_only=False for backward compatibility
    if "weights_only" not in load_kwargs:
        try:
            sig = inspect.signature(_original_torch_load)
            if "weights_only" in sig.parameters:
                load_kwargs["weights_only"] = False
        except (ValueError, TypeError):
            pass

    load_kwargs.update(kwargs)

    try:
        return _original_torch_load(path, **load_kwargs)
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)

        if 'weights_only' in error_msg or 'Weights only load failed' in error_msg or 'UnpicklingError' in error_type:
            raise RuntimeError(
                f"Failed to load checkpoint from '{path}'. "
                f"This may be caused by PyTorch 2.6+ defaulting to weights_only=True. "
                f"Ensure the checkpoint file exists and is not corrupted. "
                f"If using a custom checkpoint, it may need to be re-saved with torch.save(). "
                f"Original error ({error_type}): {error_msg}"
            ) from e

        raise RuntimeError(
            f"Failed to load checkpoint from '{path}'. "
            f"Original error ({error_type}): {error_msg}"
        ) from e


def _apply_torch_compat_patch():
    """
    Apply the torch.load compatibility monkey-patch.

    This should be called early in the DeOldify import chain to ensure
    all torch.load() calls use the compatibility wrapper.
    """
    _register_safe_globals()
    torch.load = compatible_torch_load
    logger.debug(
        "Applied torch.load compatibility patch for PyTorch version: %s",
        torch.__version__
    )


# Apply the patch immediately when this module is imported
_apply_torch_compat_patch()
