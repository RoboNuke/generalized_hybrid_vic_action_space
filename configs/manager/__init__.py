"""Public surface of the config-manager package.

Re-exports the registry-driven loader and the registered dataclass types so callers
can keep writing ``from configs.manager import ConfigManager, SAC_CFG, ModelCfg``.
"""

from configs.manager.controller_cfg import ControlCfg
from configs.manager.loss_cfg import LossCfg
from configs.manager.manager import ConfigManager
from configs.manager.model_cfg import ActorCfg, CriticCfg, ModelCfg
from configs.manager.ppo_cfg import PPO_CFG
from configs.manager.preprocessor_registry import (
    available_preprocessors,
    resolve_preprocessor,
)
from configs.manager.runner_cfg import RunnerCfg
from configs.manager.sac_cfg import SAC_CFG
from configs.manager.sensor_cfg import ContactCfg, SensorCfg

__all__ = [
    "ConfigManager",
    "SAC_CFG",
    "PPO_CFG",
    "ModelCfg",
    "ActorCfg",
    "CriticCfg",
    "RunnerCfg",
    "ControlCfg",
    "SensorCfg",
    "ContactCfg",
    "LossCfg",
    "resolve_preprocessor",
    "available_preprocessors",
]
