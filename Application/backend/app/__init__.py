"""PlantCare AI backend package."""
import os

# On Windows, PyTorch must be loaded before scikit-learn, otherwise loading it fails with
# "WinError 1114 ... c10.dll" (clashing OpenMP runtimes). PyTorch is optional (hybrid CNN decision).
if os.environ.get('PLANTCARE_DISABLE_CNN') != '1':
    try:
        import torch  # noqa: F401
    except Exception:  # not installed / not loadable: the app runs WL only
        pass
