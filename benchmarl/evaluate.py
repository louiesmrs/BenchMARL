#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import argparse
from pathlib import Path

from benchmarl.hydra_config import reload_experiment_from_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates the experiment from a checkpoint file."
    )
    parser.add_argument(
        "checkpoint_file", type=str, help="The name of the checkpoint file"
    )
    parser.add_argument(
        "--env-pickle",
        type=str,
        default=None,
        help="Optional path to a Flatland .pkl env file to evaluate on",
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override experiment.render during evaluation",
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=None,
        help="Override number of evaluation episodes",
    )
    args = parser.parse_args()
    checkpoint_file = str(Path(args.checkpoint_file).resolve())

    overrides = []
    overrides.append("experiment.evaluation_only=true")
    if args.env_pickle is not None:
        overrides.append(f"+task.env_pickle={args.env_pickle}")
    if args.render is not None:
        overrides.append(f"experiment.render={str(args.render).lower()}")
    if args.evaluation_episodes is not None:
        overrides.append(
            f"experiment.evaluation_episodes={args.evaluation_episodes}"
        )

    experiment = reload_experiment_from_file(checkpoint_file, overrides or None)

    if args.env_pickle is not None:
        from flatland.envs.persistence import RailEnvPersister

        env, _ = RailEnvPersister.load_new(args.env_pickle)
        n_agents_env = env.get_num_agents()
        n_agents_cfg = experiment.task.config.get("num_agents")
        if n_agents_cfg is not None and n_agents_env != n_agents_cfg:
            raise ValueError(
                f"env_pickle has {n_agents_env} agents but checkpoint was trained with {n_agents_cfg}."
            )

    experiment.evaluate()
    experiment.close()
