"""Observation-noise configuration — a single dataclass for the Forge obs randomization.

``NoiseCfg`` subclasses Isaac Lab's :class:`ForgeObsRandCfg` (which subclasses the Factory
:class:`ObsRandCfg`), so every observation-noise field the Forge/Factory env already defines is
inherited here rather than re-declared:

* ``fingertip_pos``      — per-step Gaussian noise on the EE position obs (metres).
* ``fingertip_rot_deg``  — per-step axis-angle perturbation of the EE quaternion obs (degrees).
* ``ft_force``           — per-step Gaussian noise on the force-sensor obs (newtons).
* ``fixed_asset_pos``    — per-reset Gaussian offset of the fixed-asset observation frame, held
                           constant for the whole episode (metres, length-3 ``[x, y, z]``).

The runner copies these inherited fields straight onto ``env_cfg.obs_rand`` (see
``learning/runner.py``), so the noise levels can be tuned from the experiment YAML in ONE place
and flow through to the Forge env. Omitting the ``noise_cfg`` YAML section default-constructs
this class, which equals the Isaac Lab defaults — i.e. existing experiments are unchanged.

Every term disables fully at zero (the env multiplies fresh ``randn`` by the level each step /
reset), so there are no on/off booleans: set a level to ``0.0`` (or ``[0, 0, 0]`` for
``fixed_asset_pos``) to switch that noise off.

The reset-level CONTROL randomization (gains / threshold / dead-zone) lives on
:class:`ForgeCtrlCfg` and is tuned from the ``controller_cfg`` YAML section instead (see
``configs/manager/controller_cfg.py``); it is intentionally not duplicated here.

NOTE: importing this module imports Isaac Lab (ForgeObsRandCfg), so it must be imported only
after the Omniverse ``AppLauncher`` has booted (the runner does this).
"""

from isaaclab.utils import configclass
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeObsRandCfg


@configclass
class NoiseCfg(ForgeObsRandCfg):
    """Unified observation-noise config (registered YAML header ``noise_cfg``).

    Inherits all Forge/Factory ``obs_rand`` fields; declares no new ones. Set any level to 0
    (``fixed_asset_pos`` to ``[0, 0, 0]``) to disable that noise term.
    """

    def __post_init__(self):
        if len(self.fixed_asset_pos) != 3:
            raise ValueError(
                f"NoiseCfg.fixed_asset_pos must be length 3 [x, y, z], got {self.fixed_asset_pos!r}"
            )
        for name in ("fingertip_pos", "fingertip_rot_deg", "ft_force"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"NoiseCfg.{name} must be >= 0, got {getattr(self, name)!r}")
        if any(v < 0.0 for v in self.fixed_asset_pos):
            raise ValueError(
                f"NoiseCfg.fixed_asset_pos components must be >= 0, got {self.fixed_asset_pos!r}"
            )
