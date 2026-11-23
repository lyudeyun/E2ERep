"""Faster version of uniad_b2d_agent.py (UniadAgent) for closed-loop evaluation: same camera
resolution, each camera image processed on arrival, image pre-processing on the GPU
(see gpu_preproc.py). uniad_b2d_agent.py is unchanged.

The UniAD pipeline has no resize, so the model input is bit-identical to UniadAgent.
"""
from Bench2DriveZoo.team_code.gpu_preproc import GpuPreprocMixin
from Bench2DriveZoo.team_code.uniad_b2d_agent import UniadAgent


def get_entry_point():
    return 'UniadAgentGPU'


class UniadAgentGPU(GpuPreprocMixin, UniadAgent):
    pass
