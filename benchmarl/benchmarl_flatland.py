from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.environments.flatland.common import FlatlandTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig


def _configure_experiment(experiment_config: ExperimentConfig, algo_name: str):
    experiment_config.lr = 5e-5
    experiment_config.on_policy_n_minibatch_iters = 10
    experiment_config.on_policy_n_envs_per_worker = 10
    experiment_config.on_policy_collected_frames_per_batch = 6000
    experiment_config.on_policy_minibatch_size = 400
    experiment_config.max_n_frames = None
    experiment_config.max_n_iters = 1
    return experiment_config


def _configure_algorithm(algo_name: str):
    if algo_name == "mappo":
        algo_config = MappoConfig.get_from_yaml()
    elif algo_name == "ippo":
        algo_config = IppoConfig.get_from_yaml()
    else:
        raise ValueError("algo must be one of: mappo, ippo")

    algo_config.lmbda = 1.0
    algo_config.clip_epsilon = 0.1
    algo_config.critic_coef = 1.0
    algo_config.entropy_coef = 0.0
    return algo_config


def run(algo_name: str, task_name: str, seed: int):
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config = _configure_experiment(experiment_config, algo_name)

    task = FlatlandTask.PHASE_1_3_7_TO_10_AGENTS.get_from_yaml()

    algorithm_config = _configure_algorithm(algo_name)
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed,
        config=experiment_config,
    )
    experiment.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["mappo", "ippo"], required=True)
    parser.add_argument("--task", default="phase_1_3_7_to_10_agents")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    run(args.algo, args.task, args.seed)
