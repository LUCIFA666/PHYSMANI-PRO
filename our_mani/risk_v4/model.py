"""r08 nested calibration over an unchanged r07 task critic.

The inherited predictor uses the same raw ensemble, observable feature schema,
TRAIN support guard and ranking guard. Only its positive affine calibration is
versioned here; calibrated member spread remains a diagnostic, not a penalty.
"""
import hashlib
import json
from pathlib import Path

from ..risk_v3.model import CandidateProbability as R07CandidateProbability

CALIBRATION_VERSION = 'candidate_calibration_r08_nested_source_cv_v1'
LAMBDAS = (.01, .1, 1.)
TIE_TOLERANCE = 1e-10


def calibration_program_hash(files):
    protocol = dict(version=CALIBRATION_VERSION, files=files, lambdas=LAMBDAS,
        fit_loss='attempt-uniform NLL + lambda*((slope-1)^2+intercept^2)',
        selection='two-source-fold pooled attempt Brier; identity first on ties',
        tie_tolerance=TIE_TOLERANCE, minimum_positive_and_negative_sources=2,
        outer_folds='exact r07 fixed raw-source assignment')
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


class CandidateProbability(R07CandidateProbability):
    def __init__(self, checkpoint, *, controller_sha256=None):
        super().__init__(checkpoint, controller_sha256=controller_sha256)
        if self.metadata.get('calibration_version') != CALIBRATION_VERSION:
            raise ValueError('candidate critic requires the versioned r08 nested calibration')
        root = Path(__file__).resolve().parents[2]
        files = self.metadata['calibration_files']
        actual = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
        if actual != files or calibration_program_hash(actual) != self.metadata['calibration_program_sha256']:
            raise ValueError('r08 calibration implementation differs from the saved artifact')
        if not self.metadata.get('raw_head_checkpoint_sha256') or not self.metadata.get('raw_selection_sha256'):
            raise ValueError('r08 critic requires immutable r07 raw-head lineage')
