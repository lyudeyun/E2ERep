"""Faster version of vad_b2d_agent.py (VadAgent) for closed-loop evaluation: same camera
resolution, each camera image processed on arrival, image pre-processing on the GPU
(see gpu_preproc.py). vad_b2d_agent.py is unchanged.

Planned waypoints differ from VadAgent by up to ~1.5 cm (the GPU resize is not
bit-identical to cv2.resize). On a Town12 route one tick takes about 210 ms instead of 442 ms.
"""
from Bench2DriveZoo.team_code.gpu_preproc import GpuPreprocMixin
from Bench2DriveZoo.team_code.vad_b2d_agent import VadAgent


def get_entry_point():
    return 'VadAgentGPU'


class VadAgentGPU(GpuPreprocMixin, VadAgent):
    pass
