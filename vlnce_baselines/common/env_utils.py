import os
import random
import sys
from typing import List, Optional, Type, Union

import habitat
from habitat import logger
try:
    from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
except Exception:
    from vlnce_baselines.config.shim_config import Config
    try:
        from habitat import Env, RLEnv, VectorEnv, make_dataset
    except Exception:
        Env = RLEnv = VectorEnv = make_dataset = None

random.seed(0)

SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)


# 新加: 单环境调试时优先使用线程版 VectorEnv，这样环境异常会直接在主进程抛出，
# 不会只表现成子进程 EOFError；也允许用环境变量强制开启该模式。
def _should_use_threaded_vector_env(num_envs: int) -> bool:
    force_threaded = os.environ.get("ETPNAV_FORCE_THREADED_VECTOR_ENV", "")
    if force_threaded.lower() in {"1", "true", "yes", "on"}:
        return True
    return bool(sys.gettrace()) or num_envs == 1


# 新加: 兼容新版 habitat-baselines 不再提供 habitat_baselines.utils.env_utils.make_env_fn。
def make_env_fn(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    dataset=None,
) -> Union[Env, RLEnv]:
    if dataset is None:
        dataset = make_dataset(
            config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
        )
    print(
        "[make_env_fn]",
        "seed=", config.TASK_CONFIG.SEED,
        "UPPER_GPU=", getattr(config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0, "GPU_DEVICE_ID", "NA"),
        flush=True,
    )
    env = env_class(config=config, dataset=dataset)
    env.seed(config.TASK_CONFIG.SEED)
    return env


def is_slurm_job() -> bool:
    return SLURM_JOBID is not None


def is_slurm_batch_job() -> bool:
    r"""Heuristic to determine if a slurm job is a batch job or not. Batch jobs
    will have a job name that is not a shell unless the user specifically set the job
    name to that of a shell. Interactive jobs have a shell name as their job name.
    """
    return is_slurm_job() and os.environ.get("SLURM_JOB_NAME", None) not in (
        None,
        "bash",
        "zsh",
        "fish",
        "tcsh",
        "sh",
    )


def construct_envs(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
    auto_reset_done: bool = True,
    episodes_allowed: Optional[List[str]] = None,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param auto_reset_done: Whether or not to automatically reset the env on done
    :return: VectorEnv object created according to specification.
    """

    num_envs_per_gpu = config.NUM_ENVIRONMENTS
    if isinstance(config.SIMULATOR_GPU_IDS, list):
        gpus = config.SIMULATOR_GPU_IDS
    else:
        gpus = [config.SIMULATOR_GPU_IDS]
    num_gpus = len(gpus)
    num_envs = num_gpus * num_envs_per_gpu

    if episodes_allowed is not None:
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episodes_allowed
        config.freeze()

    configs = []
    env_classes = [env_class for _ in range(num_envs)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    logger.info(f"SPLTI: {config.TASK_CONFIG.DATASET.SPLIT}, NUMBER OF SCENES: {len(scenes)}")

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multi-process logic relies on being able"
                " to split scenes uniquely between processes"
            )

        if len(scenes) < num_envs and len(scenes) != 1:
            raise RuntimeError(
                "reduce the number of GPUs or envs as there"
                " aren't enough number of scenes"
            )

        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)

        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_gpus):
        for j in range(num_envs_per_gpu):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_envs_per_gpu) + j

            task_config = proc_config.TASK_CONFIG
            task_config.SEED += proc_id
            if len(scenes) > 0:
                task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]

            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]
            task_config.SIMULATOR.HABITAT_SIM_V0.gpu_device_id = gpus[i]

            task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS
            task_config.SIMULATOR.AGENT_0.sensors = config.SENSORS

            # 强制同步新版 Habitat 运行时会读取的小写别名
            if hasattr(task_config, "simulator"):
                if hasattr(task_config.simulator, "habitat_sim_v0"):
                    task_config.simulator.habitat_sim_v0.gpu_device_id = gpus[i]
                if hasattr(task_config.simulator, "agent_0"):
                    task_config.simulator.agent_0.sensors = config.SENSORS

            proc_config.freeze()
            configs.append(proc_config) 

    use_threaded_env = _should_use_threaded_vector_env(num_envs)
    env_entry = habitat.ThreadedVectorEnv if use_threaded_env else habitat.VectorEnv
    envs = env_entry(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)), 
        auto_reset_done=auto_reset_done,
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs


def construct_envs_auto_reset_false(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> VectorEnv:
    return construct_envs(config, env_class, auto_reset_done=False)

def construct_envs_for_rl(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
    auto_reset_done: bool = True,
    episodes_allowed: Optional[List[str]] = None,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param auto_reset_done: Whether or not to automatically reset the env on done
    :return: VectorEnv object created according to specification.
    """

    num_envs_per_gpu = config.NUM_ENVIRONMENTS
    if isinstance(config.SIMULATOR_GPU_IDS, list):
        gpus = config.SIMULATOR_GPU_IDS
    else:
        gpus = [config.SIMULATOR_GPU_IDS]
    num_gpus = len(gpus)
    num_envs = num_gpus * num_envs_per_gpu

    if episodes_allowed is not None:
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episodes_allowed
        config.freeze()

    configs = []
    env_classes = [env_class for _ in range(num_envs)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multi-process logic relies on being able"
                " to split scenes uniquely between processes"
            )

        if len(scenes) < num_envs and len(scenes) != 1:
            raise RuntimeError(
                "reduce the number of GPUs or envs as there"
                " aren't enough number of scenes"
            )
        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)

        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_gpus):
        for j in range(num_envs_per_gpu):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_envs_per_gpu) + j

            task_config = proc_config.TASK_CONFIG
            task_config.SEED += proc_id
            if len(scenes) > 0:
                task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]

            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]
            task_config.SIMULATOR.HABITAT_SIM_V0.gpu_device_id = gpus[i]

            task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS
            task_config.SIMULATOR.AGENT_0.sensors = config.SENSORS

            # 强制同步新版 Habitat 运行时会读取的小写别名
            if hasattr(task_config, "simulator"):
                if hasattr(task_config.simulator, "habitat_sim_v0"):
                    task_config.simulator.habitat_sim_v0.gpu_device_id = gpus[i]
                if hasattr(task_config.simulator, "agent_0"):
                    task_config.simulator.agent_0.sensors = config.SENSORS


            proc_config.freeze()
            configs.append(proc_config)

    use_threaded_env = _should_use_threaded_vector_env(num_envs)
    env_entry = habitat.ThreadedVectorEnv if use_threaded_env else habitat.VectorEnv
    envs = env_entry(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)),
        auto_reset_done=auto_reset_done,
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs
