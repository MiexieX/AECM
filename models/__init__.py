"""
model list
"""
from .baseline import Baseline
from .vlci import VLCI
from .vlci_original import VLCI as VLCIOriginal
from .vlci_ccli_clean import VLCICCLIClean
from .vlp import VLP

model_fn = {"baseline": Baseline, "vlci": VLCI, "vlci_original": VLCIOriginal, "vlci_ccli_clean": VLCICCLIClean, "vlp": VLP}
