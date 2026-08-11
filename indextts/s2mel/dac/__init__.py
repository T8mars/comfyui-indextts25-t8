__version__ = "1.0.0"

# Preserved for model compatibility.  The ComfyUI inference path only imports
# ``dac.nn.quantize`` through the length regulator; importing the full training
# package here would unnecessarily require descript-audiotools and its legacy
# dependency pins.
__model_version__ = "latest"
