#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import functools
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from math import prod
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import torch
import yaml
from tensordict import TensorDictBase
from torchrl.data import Composite
from torchrl.data.tensor_specs import UnboundedContinuous
from torchrl.envs import EnvBase
from torchrl.envs.transforms import ExcludeTransform, Transform

BENCHMARL_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_EXPERIMENT_CONFIG_DIR = (
    Path(__file__).resolve().parent / "conf" / "experiment" / "benchmark"
)
TREE_BASE_CONFIG_PATH = (
    Path(__file__).resolve().parent / "conf" / "experiment" / "base_tree.yaml"
)
sys.path.insert(0, str(BENCHMARL_ROOT))

from benchmarl.algorithms import (
    IppoConfig,
    IqlConfig,
    MappoConfig,
    MasacConfig,
    VdnConfig,
)
from benchmarl.benchmark import Benchmark
from benchmarl.environments import FlatlandTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models import (
    FlatlandTreeGNNCriticConfig,
    FlatlandTreeGNNPolicyConfig,
    FlatlandTreeLSTMCriticConfig,
    FlatlandTreeLSTMFeatureConfig,
    FlatlandTreeLSTMPolicyConfig,
    FlatlandTreeTransformerFeatureConfig,
    FlatlandTreeTransformerCriticConfig,
    FlatlandTreeTransformerPolicyConfig,
    GnnConfig,
    GruConfig,
    LstmConfig,
    MlpConfig,
    SequenceModelConfig,
)

AGENT_GROUP = "agents"
OBS_KEYS: tuple[str, ...] = (
    "agents_attr",
    "adjacency",
    "node_attr",
    "node_order",
    "edge_order",
)
EXCLUDED_OBS_KEYS: tuple[str, ...] = (
    "adjacency",
    "node_attr",
    "node_order",
    "edge_order",
    "valid_actions",
    "adjacency_offset",
    "positional_encoding",
)
ALGORITHM_ORDER: tuple[str, ...] = (
    "ippo",
    "mappo",
    "masac",
    "vdn",
    "iql",
)


class FlatlandTreeActionMaskTransform(Transform):
    def __init__(self, agent_group: str = AGENT_GROUP) -> None:
        super().__init__()
        self.agent_group = agent_group

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        valid_actions = tensordict.get(
            (self.agent_group, "observation", "valid_actions")
        ).to(torch.bool)
        tensordict.set((self.agent_group, "action_mask"), valid_actions)
        return tensordict

    def transform_observation_spec(self, observation_spec):
        if not isinstance(observation_spec, Composite):
            return observation_spec
        if self.agent_group not in observation_spec.keys():
            return observation_spec

        group_spec = observation_spec[self.agent_group]
        if "observation" not in group_spec.keys():
            return observation_spec

        obs_spec = group_spec["observation"]
        if "valid_actions" not in obs_spec.keys():
            return observation_spec

        group_spec["action_mask"] = obs_spec["valid_actions"].clone()
        return observation_spec


class FlatlandMlpBenchmarkTransform(Transform):
    def __init__(
        self,
        agent_group: str = AGENT_GROUP,
        obs_keys: Sequence[str] = OBS_KEYS,
    ) -> None:
        super().__init__()
        self.agent_group = agent_group
        self.obs_keys = tuple(obs_keys)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs_prefix = (self.agent_group, "observation")
        flat_observation = torch.cat(
            [
                tensordict.get((*obs_prefix, key)).to(torch.float32)
                for key in self.obs_keys
            ],
            dim=-1,
        )
        tensordict.set((*obs_prefix, "flat_observation"), flat_observation)
        tensordict.set(
            (self.agent_group, "action_mask"),
            tensordict.get((*obs_prefix, "valid_actions")).to(torch.bool),
        )
        return tensordict

    def transform_observation_spec(self, observation_spec):
        if not isinstance(observation_spec, Composite):
            return observation_spec
        if self.agent_group not in observation_spec.keys():
            return observation_spec

        group_spec = observation_spec[self.agent_group]
        if "observation" not in group_spec.keys():
            return observation_spec

        obs_spec = group_spec["observation"]
        if any(key not in obs_spec.keys() for key in self.obs_keys):
            return observation_spec
        if "valid_actions" not in obs_spec.keys():
            return observation_spec

        agent_shape_len = len(obs_spec.shape)
        flat_dim = sum(
            int(prod(obs_spec[key].shape[agent_shape_len:])) for key in self.obs_keys
        )
        spec_device = obs_spec[self.obs_keys[0]].device
        obs_spec["flat_observation"] = UnboundedContinuous(
            shape=(*obs_spec.shape, flat_dim),
            dtype=torch.float32,
            device=spec_device,
        )
        group_spec["action_mask"] = obs_spec["valid_actions"].clone()
        return observation_spec


class FlatlandTreeHybridObservationTransform(Transform):
    def __init__(
        self,
        agent_group: str = AGENT_GROUP,
        obs_keys: Sequence[str] = OBS_KEYS,
    ) -> None:
        super().__init__()
        self.agent_group = agent_group
        self.obs_keys = tuple(obs_keys)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs_prefix = (self.agent_group, "observation")
        flat_observation = torch.cat(
            [
                tensordict.get((*obs_prefix, key)).to(torch.float32)
                for key in self.obs_keys
            ],
            dim=-1,
        )
        tensordict.set((*obs_prefix, "flat_observation"), flat_observation)
        return tensordict

    def transform_observation_spec(self, observation_spec):
        if not isinstance(observation_spec, Composite):
            return observation_spec
        if self.agent_group not in observation_spec.keys():
            return observation_spec

        group_spec = observation_spec[self.agent_group]
        if "observation" not in group_spec.keys():
            return observation_spec

        obs_spec = group_spec["observation"]
        if any(key not in obs_spec.keys() for key in self.obs_keys):
            return observation_spec

        agent_shape_len = len(obs_spec.shape)
        flat_dim = sum(
            int(prod(obs_spec[key].shape[agent_shape_len:])) for key in self.obs_keys
        )
        spec_device = obs_spec[self.obs_keys[0]].device
        obs_spec["flat_observation"] = UnboundedContinuous(
            shape=(*obs_spec.shape, flat_dim),
            dtype=torch.float32,
            device=spec_device,
        )
        return observation_spec


def _resolve_algorithm_names(algo: str) -> list[str]:
    if algo == "all":
        return list(ALGORITHM_ORDER)
    return [algo]


def _build_algorithm_config(algo_name: str):
    builders = {
        "ippo": IppoConfig,
        "mappo": MappoConfig,
        "masac": MasacConfig,
        "vdn": VdnConfig,
        "iql": IqlConfig,
    }
    if algo_name not in builders:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    return builders[algo_name].get_from_yaml()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Flatland benchmark for a selected model family."
    )
    parser.add_argument(
        "--model",
        default="mlp",
        choices=[
            "mlp",
            "lstm",
            "gru",
            "gnn",
            "treelstm",
            "treelstm_gru",
            "treelstm_mlp",
            "treemlp",
            "treegru",
            "treetransformer",
            "treegnn",
        ],
        help="Model family to benchmark. Defaults to mlp.",
    )
    parser.add_argument(
        "--algo",
        default="ippo",
        choices=[*ALGORITHM_ORDER, "all"],
        help="Algorithm to run (ippo/mappo/masac/vdn/iql) or 'all'.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional label for this benchmark batch. A timestamped run folder will be created.",
    )
    return parser.parse_args()


def _resolve_train_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_run_folder(model_name: str, run_name: str | None) -> Path:
    root = Path(__file__).resolve().parent / "benchmark_runs"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = run_name.strip().replace(" ", "_") if run_name else model_name
    run_dir = root / f"{timestamp}__{slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_manifest(
    run_dir: Path,
    *,
    model_name: str,
    run_name: str | None,
    task: Any,
    experiment_configs_by_algorithm: dict[str, ExperimentConfig],
    algorithm_configs: Sequence[Any],
    model_config: Any,
    critic_model_config: Any,
    seed: int,
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "model_name": model_name,
        "save_folder": str(run_dir),
        "algorithms": [
            _algorithm_name_from_config(config) for config in algorithm_configs
        ],
        "seed": seed,
        "task": _serialize_config(task.config),
        "experiment_config_dir": str(BENCHMARK_EXPERIMENT_CONFIG_DIR),
        "experiment_by_algorithm": {
            name: _serialize_config(config)
            for name, config in experiment_configs_by_algorithm.items()
        },
        "model": _serialize_config(model_config),
        "critic_model": _serialize_config(critic_model_config),
    }
    with open(run_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


TREE_MODEL_NAMES = {
    "treelstm",
    "treelstm_gru",
    "treelstm_mlp",
    "treemlp",
    "treegru",
    "treetransformer",
    "treegnn",
}


def _is_tree_model(model_name: str) -> bool:
    return model_name in TREE_MODEL_NAMES


@functools.lru_cache(maxsize=1)
def _load_tree_base_overrides() -> dict[str, Any]:
    if not TREE_BASE_CONFIG_PATH.exists():
        return {}
    return _load_yaml_dict(TREE_BASE_CONFIG_PATH)


def _apply_overrides_if_present(config_obj: Any, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if hasattr(config_obj, key):
            setattr(config_obj, key, value)


def _apply_tree_experiment_overrides(config: ExperimentConfig, model_name: str) -> None:
    if not _is_tree_model(model_name):
        return
    tree_base = _load_tree_base_overrides()
    _apply_overrides_if_present(config, tree_base.get("experiment", {}))


def _apply_tree_algorithm_overrides(algorithm_config: Any, model_name: str) -> None:
    if not _is_tree_model(model_name):
        return
    tree_base = _load_tree_base_overrides()
    _apply_overrides_if_present(algorithm_config, tree_base.get("algorithm", {}))


def _apply_tree_model_overrides(
    model_config: Any,
    critic_model_config: Any,
    model_name: str,
) -> None:
    if not _is_tree_model(model_name):
        return

    tree_base = _load_tree_base_overrides()
    actor_tree_overrides = tree_base.get("model", {})
    critic_tree_overrides = tree_base.get("critic_model", actor_tree_overrides)
    recurrent_overrides = tree_base.get("sequence_recurrent_model", {})
    mlp_overrides = tree_base.get("sequence_mlp_model", {})

    def apply(config_obj: Any, direct_tree_overrides: dict[str, Any]) -> None:
        if isinstance(config_obj, SequenceModelConfig):
            for layer in config_obj.model_configs:
                layer_name = type(layer).__name__.lower()
                if "flatlandtree" in layer_name:
                    _apply_overrides_if_present(layer, direct_tree_overrides)
                elif layer_name.startswith("gru") or layer_name.startswith("lstm"):
                    _apply_overrides_if_present(layer, recurrent_overrides)
                elif layer_name.startswith("mlp"):
                    _apply_overrides_if_present(layer, mlp_overrides)
            return

        _apply_overrides_if_present(config_obj, direct_tree_overrides)

    apply(model_config, actor_tree_overrides)
    apply(critic_model_config, critic_tree_overrides)


def _resolve_device_token(value: Any, train_device: str) -> Any:
    if isinstance(value, str) and value in {"auto", "train_device"}:
        return train_device
    return value


def _algorithm_name_from_config(algorithm_config: Any) -> str:
    return type(algorithm_config).__name__.replace("Config", "").lower()


def _build_experiment_config(
    model_name: str,
    run_dir: Path,
    algorithm_name: str,
) -> ExperimentConfig:
    config = ExperimentConfig.get_from_yaml()
    train_device = _resolve_train_device()

    base_overrides = _load_yaml_dict(BENCHMARK_EXPERIMENT_CONFIG_DIR / "base.yaml")
    algorithm_overrides = _load_yaml_dict(
        BENCHMARK_EXPERIMENT_CONFIG_DIR / f"{algorithm_name}.yaml"
    )

    merged_overrides = {**base_overrides, **algorithm_overrides}
    for key, value in merged_overrides.items():
        setattr(config, key, _resolve_device_token(value, train_device))

    _apply_tree_experiment_overrides(config, model_name)

    if model_name not in {"mlp", "lstm", "gru", "gnn"}:
        config.disable_value_estimator_vmap = True

    algo_save_folder = run_dir / algorithm_name
    algo_save_folder.mkdir(parents=True, exist_ok=True)
    config.save_folder = str(algo_save_folder)
    return config


def _serialize_config(value: Any) -> Any:
    if is_dataclass(value):
        return _serialize_config(asdict(value))
    if isinstance(value, dict):
        return {key: _serialize_config(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_config(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def _print_hydra_config(
    experiment_config: ExperimentConfig,
    algorithm_config: Any,
    task: Any,
    model_config: Any,
    critic_model_config: Any,
    seed: int,
) -> None:
    payload = {
        "experiment": _serialize_config(experiment_config),
        "algorithm_name": _algorithm_name_from_config(algorithm_config),
        "algorithm": _serialize_config(algorithm_config),
        "task": _serialize_config(task.config),
        "model_name": type(model_config).__name__.replace("Config", "").lower(),
        "model": _serialize_config(model_config),
        "critic_model_name": type(critic_model_config)
        .__name__.replace("Config", "")
        .lower(),
        "critic_model": _serialize_config(critic_model_config),
        "seed": seed,
    }
    print("\nHydra config:\n")
    print(yaml.safe_dump(payload, sort_keys=False))


def _build_tree_task(include_hybrid_observation: bool = True):
    task = FlatlandTask.FLATLAND.get_from_yaml()
    reward_coefs = dict(task.config.get("reward_coefs") or {})
    reward_coefs["delay_reward"] = 1
    #   reward_coefs["departure_reward"] = 1
    task.config.update(
        {
            "num_agents": 2,
            "reward_coefs": reward_coefs,
        }
    )

    base_get_env_transforms = task.get_env_transforms

    def get_env_transforms(self, env: EnvBase):
        transforms = list(base_get_env_transforms(env))
        transforms.append(FlatlandTreeActionMaskTransform(agent_group=AGENT_GROUP))
        if include_hybrid_observation:
            transforms.append(
                FlatlandTreeHybridObservationTransform(agent_group=AGENT_GROUP)
            )
        return transforms

    def action_mask_spec(self, env: EnvBase) -> Composite | None:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "action_mask":
                    del group_obs_spec[key]
            if group_obs_spec.is_empty():
                del observation_spec[group]
        if observation_spec.is_empty():
            return None
        return observation_spec

    task.get_env_transforms = MethodType(get_env_transforms, task)
    task.action_mask_spec = MethodType(action_mask_spec, task)
    return task


def _build_mlp_task():
    task = _build_tree_task()

    base_get_env_transforms = task.get_env_transforms

    def get_env_transforms(self, env: EnvBase):
        transforms = list(base_get_env_transforms(env))
        transforms.append(FlatlandMlpBenchmarkTransform(agent_group=AGENT_GROUP))
        transforms.append(
            ExcludeTransform(
                *[(AGENT_GROUP, "observation", key) for key in EXCLUDED_OBS_KEYS]
            )
        )
        return transforms

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.observation_spec.clone()
        if (
            AGENT_GROUP in observation_spec.keys()
            and "action_mask" in observation_spec[AGENT_GROUP].keys()
        ):
            del observation_spec[(AGENT_GROUP, "action_mask")]
        if (
            AGENT_GROUP in observation_spec.keys()
            and "observation" in observation_spec[AGENT_GROUP].keys()
        ):
            group_observation_spec = observation_spec[AGENT_GROUP]["observation"]
            for key in list(group_observation_spec.keys()):
                if key != "flat_observation":
                    del group_observation_spec[key]
        return observation_spec

    def state_spec(self, env: EnvBase) -> Composite | None:
        return None

    def action_mask_spec(self, env: EnvBase) -> Composite | None:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "action_mask":
                    del group_obs_spec[key]
            if group_obs_spec.is_empty():
                del observation_spec[group]
        if observation_spec.is_empty():
            return None
        return observation_spec

    task.get_env_transforms = MethodType(get_env_transforms, task)
    task.observation_spec = MethodType(observation_spec, task)
    task.state_spec = MethodType(state_spec, task)
    task.action_mask_spec = MethodType(action_mask_spec, task)
    return task


def _build_task(model_name: str):
    if model_name in {"mlp", "lstm", "gru", "gnn"}:
        return _build_mlp_task()
    if model_name in {"treemlp", "treegru"}:
        return _build_tree_task(include_hybrid_observation=False)
    return _build_tree_task(include_hybrid_observation=True)


def _build_model_configs(model_name: str):
    model_root = BENCHMARL_ROOT / "benchmarl" / "conf" / "model" / "layers"
    if model_name == "mlp":
        return MlpConfig.get_from_yaml(), MlpConfig.get_from_yaml()
    if model_name == "lstm":
        return (
            LstmConfig.get_from_yaml(str(model_root / "lstm.yaml")),
            LstmConfig.get_from_yaml(str(model_root / "lstm.yaml")),
        )
    if model_name == "gru":
        return (
            GruConfig.get_from_yaml(str(model_root / "gru.yaml")),
            GruConfig.get_from_yaml(str(model_root / "gru.yaml")),
        )
    if model_name == "gnn":
        try:
            import torch_geometric
        except ImportError as err:
            raise ImportError("gnn requires torch_geometric to be installed.") from err

        gnn_actor = GnnConfig.get_from_yaml(str(model_root / "gnn.yaml"))
        gnn_actor.topology = "full"
        gnn_actor.self_loops = True
        gnn_actor.gnn_class = torch_geometric.nn.conv.GATv2Conv
        gnn_actor.gnn_kwargs = {
            "aggr": "mean",
            "add_self_loops": False,
        }

        gnn_critic = GnnConfig.get_from_yaml(str(model_root / "gnn.yaml"))
        gnn_critic.topology = "full"
        gnn_critic.self_loops = True
        gnn_critic.gnn_class = torch_geometric.nn.conv.GATv2Conv
        gnn_critic.gnn_kwargs = {
            "aggr": "mean",
            "add_self_loops": False,
        }

        model_config = SequenceModelConfig(
            model_configs=[
                gnn_actor,
                MlpConfig.get_from_yaml(str(model_root / "mlp.yaml")),
            ],
            intermediate_sizes=[128],
        )
        critic_model_config = SequenceModelConfig(
            model_configs=[
                gnn_critic,
                MlpConfig.get_from_yaml(str(model_root / "mlp.yaml")),
            ],
            intermediate_sizes=[128],
        )
        return model_config, critic_model_config
    if model_name == "treelstm":
        model_config = FlatlandTreeLSTMPolicyConfig.get_from_yaml(
            str(model_root / "flatland_treelstm.yaml")
        )
        critic_model_config = FlatlandTreeLSTMCriticConfig.get_from_yaml(
            str(model_root / "flatland_treelstm_critic.yaml")
        )
        _apply_tree_model_overrides(model_config, critic_model_config, model_name)
        return model_config, critic_model_config
    if model_name == "treelstm_gru":
        tree_feature_path = model_root / "flatland_treelstm_feature.yaml"
        gru_path = model_root / "gru.yaml"
        intermediate_size = 128

        model_config = SequenceModelConfig(
            model_configs=[
                FlatlandTreeLSTMFeatureConfig.get_from_yaml(str(tree_feature_path)),
                GruConfig.get_from_yaml(str(gru_path)),
            ],
            intermediate_sizes=[intermediate_size],
        )
        critic_model_config = SequenceModelConfig(
            model_configs=[
                FlatlandTreeLSTMFeatureConfig.get_from_yaml(str(tree_feature_path)),
                GruConfig.get_from_yaml(str(gru_path)),
            ],
            intermediate_sizes=[intermediate_size],
        )
        _apply_tree_model_overrides(model_config, critic_model_config, model_name)
        return model_config, critic_model_config
    if model_name == "treelstm_mlp":
        tree_feature_path = model_root / "flatland_treelstm_feature.yaml"
        mlp_path = model_root / "mlp.yaml"
        intermediate_size = 128

        model_config = SequenceModelConfig(
            model_configs=[
                FlatlandTreeLSTMFeatureConfig.get_from_yaml(str(tree_feature_path)),
                MlpConfig.get_from_yaml(str(mlp_path)),
            ],
            intermediate_sizes=[intermediate_size],
        )
        critic_model_config = SequenceModelConfig(
            model_configs=[
                FlatlandTreeLSTMFeatureConfig.get_from_yaml(str(tree_feature_path)),
                MlpConfig.get_from_yaml(str(mlp_path)),
            ],
            intermediate_sizes=[intermediate_size],
        )
        _apply_tree_model_overrides(model_config, critic_model_config, model_name)
        return model_config, critic_model_config
    if model_name == "treemlp":
        tree_feature_path = model_root / "flatland_tree_transformer_feature.yaml"
        mlp_path = model_root / "mlp.yaml"
        intermediate_size = 128

        model_config = SequenceModelConfig(
            model_configs=[
                FlatlandTreeTransformerFeatureConfig.get_from_yaml(
                    str(tree_feature_path)
                ),
                MlpConfig.get_from_yaml(str(mlp_path)),
            ],
            intermediate_sizes=[intermediate_size],
        )
        critic_model_config = SequenceModelConfig(
            model_configs=[
                FlatlandTreeTransformerFeatureConfig.get_from_yaml(
                    str(tree_feature_path)
                ),
                MlpConfig.get_from_yaml(str(mlp_path)),
            ],
            intermediate_sizes=[intermediate_size],
        )
        _apply_tree_model_overrides(model_config, critic_model_config, model_name)
        return model_config, critic_model_config
    if model_name == "treegru":
        tree_feature_path = model_root / "flatland_tree_transformer_feature.yaml"
        gru_path = model_root / "gru.yaml"
        intermediate_size = 128

        model_config = SequenceModelConfig(
            model_configs=[
                FlatlandTreeTransformerFeatureConfig.get_from_yaml(
                    str(tree_feature_path)
                ),
                GruConfig.get_from_yaml(str(gru_path)),
            ],
            intermediate_sizes=[intermediate_size],
        )
        critic_model_config = SequenceModelConfig(
            model_configs=[
                FlatlandTreeTransformerFeatureConfig.get_from_yaml(
                    str(tree_feature_path)
                ),
                GruConfig.get_from_yaml(str(gru_path)),
            ],
            intermediate_sizes=[intermediate_size],
        )
        _apply_tree_model_overrides(model_config, critic_model_config, model_name)
        return model_config, critic_model_config
    if model_name == "treetransformer":
        model_config = FlatlandTreeTransformerPolicyConfig.get_from_yaml(
            str(model_root / "flatland_tree_transformer.yaml")
        )
        critic_model_config = FlatlandTreeTransformerCriticConfig.get_from_yaml(
            str(model_root / "flatland_tree_transformer_critic.yaml")
        )
        _apply_tree_model_overrides(model_config, critic_model_config, model_name)
        return model_config, critic_model_config
    if model_name == "treegnn":
        model_config = FlatlandTreeGNNPolicyConfig.get_from_yaml(
            str(model_root / "flatland_treegnn.yaml")
        )
        critic_model_config = FlatlandTreeGNNCriticConfig.get_from_yaml(
            str(model_root / "flatland_treegnn_critic.yaml")
        )
        _apply_tree_model_overrides(model_config, critic_model_config, model_name)
        return model_config, critic_model_config
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    args = _parse_args()
    model_name = args.model
    run_dir = _make_run_folder(model_name=model_name, run_name=args.run_name)

    task = _build_task(model_name)
    model_config, critic_model_config = _build_model_configs(model_name)

    algorithm_names = _resolve_algorithm_names(args.algo)
    algorithm_configs = [
        _build_algorithm_config(algo_name) for algo_name in algorithm_names
    ]

    for algorithm_config in algorithm_configs:
        if hasattr(algorithm_config, "minibatch_advantage"):
            algorithm_config.minibatch_advantage = True
        if hasattr(algorithm_config, "entropy_coef"):
            algorithm_config.entropy_coef = 0.0005
        _apply_tree_algorithm_overrides(algorithm_config, model_name)

    experiment_configs_by_algorithm: dict[str, ExperimentConfig] = {}
    for algorithm_config in algorithm_configs:
        algorithm_name = _algorithm_name_from_config(algorithm_config)
        experiment_configs_by_algorithm[algorithm_name] = _build_experiment_config(
            model_name=model_name,
            run_dir=run_dir,
            algorithm_name=algorithm_name,
        )

    _write_manifest(
        run_dir,
        model_name=model_name,
        run_name=args.run_name,
        task=task,
        experiment_configs_by_algorithm=experiment_configs_by_algorithm,
        algorithm_configs=algorithm_configs,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
    )
    print(f"\nBenchmark run folder: {run_dir}\n")

    for algorithm_config in algorithm_configs:
        algorithm_name = _algorithm_name_from_config(algorithm_config)
        experiment_config = experiment_configs_by_algorithm[algorithm_name]

        _print_hydra_config(
            experiment_config=experiment_config,
            algorithm_config=algorithm_config,
            task=task,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=0,
        )

        benchmark = Benchmark(
            algorithm_configs=[algorithm_config],
            model_config=model_config,
            critic_model_config=critic_model_config,
            tasks=[task],
            seeds={0},
            experiment_config=experiment_config,
        )
        benchmark.run_sequential()


if __name__ == "__main__":
    main()
