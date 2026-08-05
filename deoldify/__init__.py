import sys
import logging
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.INFO)

# Initialize PyTorch compatibility layer before any other imports.
# This monkey-patches torch.load to handle the weights_only default change
# introduced in PyTorch 2.6+, ensuring checkpoints load correctly across
# PyTorch 1.x through 2.11+.
from deoldify import _compat  # noqa: F401

from deoldify._device import _Device

device = _Device()