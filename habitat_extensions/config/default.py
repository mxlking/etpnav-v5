from typing import List, Optional, Union

from omegaconf import DictConfig, OmegaConf

from habitat.config.default import CONFIG_FILE_SEPARATOR

try:
    from habitat.config.default import Config as CN
except Exception:
    from navmorph_compat import CompatCN as CN

_C = CN()
_C.defrost()

# Ensure minimal nested nodes exist for older code that assigns into them.
if not hasattr(_C, "TASK"):
    _C.TASK = CN()
if not hasattr(_C.TASK, "ACTIONS"):
    _C.TASK.ACTIONS = CN()
if not hasattr(_C, "ENVIRONMENT"):
    _C.ENVIRONMENT = CN()
if not hasattr(_C, "SIMULATOR"):
    _C.SIMULATOR = CN()
if not hasattr(_C, "DATASET"):
    _C.DATASET = CN()
if not hasattr(_C.TASK, "TOP_DOWN_MAP_VLNCE"):
    _C.TASK.TOP_DOWN_MAP_VLNCE = CN()

# ----------------------------------------------------------------------------
# CUSTOM ACTION: HIGHTOLOWINFERENCE ACTION
# ----------------------------------------------------------------------------
_C.TASK.ACTIONS.HIGHTOLOWINFERENCE = CN()
_C.TASK.ACTIONS.HIGHTOLOWINFERENCE.TYPE = 'MoveHighToLowActionInference'
# ----------------------------------------------------------------------------
# CUSTOM ACTION: HIGHTOLOWEVAL ACTION
# ----------------------------------------------------------------------------
_C.TASK.ACTIONS.HIGHTOLOWEVAL = CN()
_C.TASK.ACTIONS.HIGHTOLOWEVAL.TYPE = 'MoveHighToLowActionEval'
# ----------------------------------------------------------------------------
# CUSTOM ACTION: HIGHTOLOW ACTION
# ----------------------------------------------------------------------------
_C.TASK.ACTIONS.HIGHTOLOW = CN()
_C.TASK.ACTIONS.HIGHTOLOW.TYPE = 'MoveHighToLowAction'
# ----------------------------------------------------------------------------
# GPS SENSOR
# ----------------------------------------------------------------------------
_C.TASK.GLOBAL_GPS_SENSOR = CN()
_C.TASK.GLOBAL_GPS_SENSOR.TYPE = "GlobalGPSSensor"
_C.TASK.GLOBAL_GPS_SENSOR.DIMENSIONALITY = 3
# ----------------------------------------------------------------------------
# OREINTATION SENSOR
# ----------------------------------------------------------------------------
_C.TASK.OREINTATION_SENSOR = CN()
_C.TASK.OREINTATION_SENSOR.TYPE = "OrienSensor"
# ----------------------------------------------------------------------------
# # RXR INSTRUCTION SENSOR
# ----------------------------------------------------------------------------
_C.TASK.RXR_INSTRUCTION_SENSOR = CN()
_C.TASK.RXR_INSTRUCTION_SENSOR.TYPE = "RxRInstructionSensor"
_C.TASK.RXR_INSTRUCTION_SENSOR.features_path = "data/datasets/RxR_VLNCE_v0/text_features/rxr_{split}/{id:06}_{lang}_text_features.npz"
_C.TASK.RXR_INSTRUCTION_SENSOR.max_text_len = 512
_C.TASK.INSTRUCTION_SENSOR_UUID = "rxr_instruction"
# ----------------------------------------------------------------------------
# SHORTEST PATH SENSOR (previously: VLN_ORACLE_ACTION_SENSOR)
# ----------------------------------------------------------------------------
_C.TASK.SHORTEST_PATH_SENSOR = CN()
_C.TASK.SHORTEST_PATH_SENSOR.TYPE = "ShortestPathSensor"
# all goals can be navigated to within 0.5m.
_C.TASK.SHORTEST_PATH_SENSOR.GOAL_RADIUS = 0.5
# compatibility with the dataset generation oracle and paper results.
# if False, use the ShortestPathFollower in Habitat
_C.TASK.SHORTEST_PATH_SENSOR.USE_ORIGINAL_FOLLOWER = False
# ----------------------------------------------------------------------------
# VLN ORACLE PROGRESS SENSOR
# ----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR = CN()
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR.TYPE = "VLNOracleProgressSensor"
# ----------------------------------------------------------------------------
# NDTW MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.NDTW = CN()
_C.TASK.NDTW.TYPE = "NDTW"
_C.TASK.NDTW.SPLIT = "val_seen"
_C.TASK.NDTW.FDTW = True  # False: DTW
_C.TASK.NDTW.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json"
)
_C.TASK.NDTW.SUCCESS_DISTANCE = 3.0
# ----------------------------------------------------------------------------
# SDTW MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.SDTW = CN()
_C.TASK.SDTW.TYPE = "SDTW"
# ----------------------------------------------------------------------------
# PATH_LENGTH MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.PATH_LENGTH = CN()
_C.TASK.PATH_LENGTH.TYPE = "PathLength"
# ----------------------------------------------------------------------------
# ORACLE_NAVIGATION_ERROR MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.ORACLE_NAVIGATION_ERROR = CN()
_C.TASK.ORACLE_NAVIGATION_ERROR.TYPE = "OracleNavigationError"
# ----------------------------------------------------------------------------
# ORACLE_SUCCESS MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.ORACLE_SUCCESS = CN()
_C.TASK.ORACLE_SUCCESS.TYPE = "OracleSuccess"
_C.TASK.ORACLE_SUCCESS.SUCCESS_DISTANCE = 3.0
# ----------------------------------------------------------------------------
# ORACLE_SPL MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.ORACLE_SPL = CN()
_C.TASK.ORACLE_SPL.TYPE = "OracleSPL"
# ----------------------------------------------------------------------------
# STEPS_TAKEN MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.STEPS_TAKEN = CN()
_C.TASK.STEPS_TAKEN.TYPE = "StepsTaken"
# ----------------------------------------------------------------------------
# POSITION MEASUREMENT For faster eval
# ----------------------------------------------------------------------------
_C.TASK.POSITION = CN()
_C.TASK.POSITION.TYPE = 'Position'
# ----------------------------------------------------------------------------
_C.TASK.POSITION_INFER = CN()
_C.TASK.POSITION_INFER.TYPE = 'PositionInfer'
# -----------------------------------------------------------------------------
# TOP_DOWN_MAP_VLNCE MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP_VLNCE = CN()
_C.TASK.TOP_DOWN_MAP_VLNCE.TYPE = "TopDownMapVLNCE"
_C.TASK.TOP_DOWN_MAP_VLNCE.MAX_EPISODE_STEPS = 1000
_C.TASK.TOP_DOWN_MAP_VLNCE.MAP_RESOLUTION = 512
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_SOURCE_AND_TARGET = True
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_BORDER = False
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_SHORTEST_PATH = False
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_REFERENCE_PATH = False
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_FIXED_WAYPOINTS = False
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_MP3D_AGENT_PATH = False
_C.TASK.TOP_DOWN_MAP_VLNCE.GRAPHS_FILE = "data/connectivity_graphs.pkl"
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR = CN()
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR.DRAW = False
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR.FOV = 79
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR.VISIBILITY_DIST = 5.0
# ----------------------------------------------------------------------------
# DATASET EXTENSIONS
# ----------------------------------------------------------------------------
_C.DATASET.ROLES = ["guide"]  # options: "*", "guide", "follower"
# language options by region: "*", "te-IN", "hi-IN", "en-US", "en-IN"
_C.DATASET.LANGUAGES = ["*"]
# a list or set of episode IDs to allow in dataset creation. None allows all.
_C.DATASET.EPISODES_ALLOWED = None


# 新加: 将旧版任务配置转换为新版 Habitat 运行时可接受的 OmegaConf 结构。
def _clone_cfg(value):
    return OmegaConf.create(OmegaConf.to_container(value, resolve=False))


def _set_default(node: DictConfig, key: str, value) -> None:
    if key not in node:
        node[key] = value


def _build_agent_config(agent_cfg: DictConfig, sim_cfg: DictConfig) -> DictConfig:
    agent = _clone_cfg(agent_cfg)
    _set_default(agent, "HEIGHT", 1.5)
    _set_default(agent, "RADIUS", 0.1)
    _set_default(agent, "MAX_CLIMB", 0.2)
    _set_default(agent, "MAX_SLOPE", 45.0)
    _set_default(agent, "START_POSITION", [0.0, 0.0, 0.0])
    _set_default(agent, "START_ROTATION", [0.0, 0.0, 0.0, 1.0])

    sensor_names = list(getattr(agent, "SENSORS", []))
    sim_sensors = {}
    for sensor_name in sensor_names:
        if sensor_name in sim_cfg:
            sensor_cfg = _clone_cfg(sim_cfg[sensor_name])
            _set_default(sensor_cfg, "POSITION", [0.0, 1.25, 0.0])
            _set_default(sensor_cfg, "ORIENTATION", [0.0, 0.0, 0.0])
            if "TYPE" not in sensor_cfg and "type" not in sensor_cfg:
                if "DEPTH" in sensor_name:
                    sensor_cfg.TYPE = "HabitatSimDepthSensor"
                elif "SEMANTIC" in sensor_name:
                    sensor_cfg.TYPE = "HabitatSimSemanticSensor"
                else:
                    sensor_cfg.TYPE = "HabitatSimRGBSensor"
            if "DEPTH" in sensor_name:
                _set_default(sensor_cfg, "MIN_DEPTH", 0.0)
                _set_default(sensor_cfg, "MAX_DEPTH", 10.0)
                _set_default(sensor_cfg, "NORMALIZE_DEPTH", True)
            sim_sensors[sensor_name.lower()] = sync_task_config_keys(sensor_cfg)
    agent.sim_sensors = OmegaConf.create(sim_sensors)
    return sync_task_config_keys(agent)


def _build_simulator_agents(sim_cfg: DictConfig) -> None:
    # 新加: 始终根据 AGENT_0.SENSORS 的当前列表重建 AGENTS，而非仅在
    # AGENTS 不存在时构建。因为 eval / _set_config 会动态向 SENSORS 追加
    # pano 相机（RGB_30, DEPTH_30 …），如果跳过重建，后添加的传感器永远
    # 不会出现在 sim_sensors 里，Habitat 就不会为它们创建 sensor 实例。
    if "AGENT_0" in sim_cfg:
        sim_cfg.AGENTS = OmegaConf.create(
            {"main_agent": _build_agent_config(sim_cfg.AGENT_0, sim_cfg)}
        )

    if "AGENTS_ORDER" not in sim_cfg and "AGENTS" in sim_cfg:
        sim_cfg.AGENTS_ORDER = list(sim_cfg.AGENTS.keys())

    _set_default(sim_cfg, "DEFAULT_AGENT_ID", 0)
    _set_default(sim_cfg, "DEFAULT_AGENT_NAVMESH", True)
    _set_default(sim_cfg, "NAVMESH_INCLUDE_STATIC_OBJECTS", False)
    _set_default(sim_cfg, "SCENE_DATASET", "default")
    _set_default(sim_cfg, "ADDITIONAL_OBJECT_PATHS", [])
    _set_default(sim_cfg, "CREATE_RENDERER", False)
    _set_default(sim_cfg, "REQUIRES_TEXTURES", True)
    _set_default(sim_cfg, "AUTO_SLEEP", False)
    _set_default(sim_cfg, "STEP_PHYSICS", True)
    _set_default(sim_cfg, "CONCUR_RENDER", False)
    _set_default(sim_cfg, "NEEDS_MARKERS", True)
    _set_default(sim_cfg, "UPDATE_ARTICULATED_AGENT", True)
    _set_default(sim_cfg, "DEBUG_RENDER", False)
    _set_default(sim_cfg, "DEBUG_RENDER_ARTICULATED_AGENT", False)
    _set_default(sim_cfg, "KINEMATIC_MODE", False)
    _set_default(sim_cfg, "SHOULD_SETUP_SEMANTIC_IDS", True)
    _set_default(sim_cfg, "DEBUG_RENDER_GOAL", True)
    _set_default(sim_cfg, "ROBOT_JOINT_START_NOISE", 0.0)
    _set_default(sim_cfg, "CTRL_FREQ", 120.0)
    _set_default(sim_cfg, "AC_FREQ_RATIO", 4)
    _set_default(sim_cfg, "LOAD_OBJS", True)
    _set_default(sim_cfg, "HOLD_THRESH", 0.15)
    _set_default(sim_cfg, "GRASP_IMPULSE", 10000.0)
    _set_default(sim_cfg, "OBJECT_IDS_START", 100)
    _set_default(
        sim_cfg,
        "RENDERER",
        OmegaConf.create(
            {
                "enable_batch_renderer": False,
                "composite_files": None,
                "classic_replay_renderer": False,
            }
        ),
    )
    sim_cfg.agents = sim_cfg.AGENTS
    sim_cfg.agents_order = sim_cfg.AGENTS_ORDER
    sim_cfg.default_agent_id = sim_cfg.DEFAULT_AGENT_ID
    sim_cfg.default_agent_navmesh = sim_cfg.DEFAULT_AGENT_NAVMESH
    sim_cfg.navmesh_include_static_objects = sim_cfg.NAVMESH_INCLUDE_STATIC_OBJECTS
    sim_cfg.scene_dataset = sim_cfg.SCENE_DATASET
    sim_cfg.additional_object_paths = sim_cfg.ADDITIONAL_OBJECT_PATHS
    sim_cfg.renderer = sim_cfg.RENDERER


def _build_task_lab_sensors(task_cfg: DictConfig) -> DictConfig:
    sensors = OmegaConf.create({})
    for sensor_name in list(getattr(task_cfg, "SENSORS", [])):
        if sensor_name == "INSTRUCTION_SENSOR":
            sensors["instruction_sensor"] = OmegaConf.create(
                {
                    "type": "InstructionSensor",
                    "instruction_sensor_uuid": getattr(
                        task_cfg, "INSTRUCTION_SENSOR_UUID", "instruction"
                    ),
                }
            )
        elif sensor_name == "SHORTEST_PATH_SENSOR":
            sensors["shortest_path_sensor"] = OmegaConf.create(
                {"type": "OracleNavigationActionSensor"}
            )
    return sensors


def _measurement_type_from_name(name: str) -> str:
    mapping = {
        "DISTANCE_TO_GOAL": "DistanceToGoal",
        "SUCCESS": "Success",
        "SPL": "SPL",
        "SOFT_SPL": "SoftSPL",
        "NDTW": "NDTW",
        "ORACLE_SUCCESS": "OracleSuccess",
        "PATH_LENGTH": "PathLength",
        "ORACLE_SPL": "OracleSPL",
        "PL": "PL",
        "POSITION": "Position",
        "POSITION_INFER": "PositionInfer",
        "COLLISIONS": "Collisions",
        "STEPS_TAKEN": "StepsTaken",
        "TOP_DOWN_MAP_VLNCE": "TopDownMap",
    }
    return mapping.get(name, name)


def _build_task_measurements(task_cfg: DictConfig, env_cfg: DictConfig) -> DictConfig:
    measurements = OmegaConf.create({})
    for measurement_name in list(getattr(task_cfg, "MEASUREMENTS", [])):
        key = measurement_name.lower()
        base_cfg = (
            _clone_cfg(task_cfg[measurement_name])
            if measurement_name in task_cfg and isinstance(task_cfg[measurement_name], DictConfig)
            else OmegaConf.create({})
        )
        base_cfg.type = _measurement_type_from_name(measurement_name)
        if measurement_name == "TOP_DOWN_MAP_VLNCE":
            _set_default(
                base_cfg,
                "max_episode_steps",
                getattr(env_cfg, "MAX_EPISODE_STEPS", 1000),
            )
            measurements["top_down_map_vlnce"] = base_cfg
        else:
            measurements[key] = base_cfg
    return measurements


def _action_type_from_name(name: str) -> str:
    mapping = {
        "STOP": "StopAction",
        "MOVE_FORWARD": "MoveForwardAction",
        "TURN_LEFT": "TurnLeftAction",
        "TURN_RIGHT": "TurnRightAction",
        "LOOK_UP": "LookUpAction",
        "LOOK_DOWN": "LookDownAction",
        # 新加：保持 ETPNav 原始动作语义，只做新版配置结构适配。
        "HIGHTOLOW": "MoveHighToLowAction",
        "HIGHTOLOWEVAL": "MoveHighToLowActionEval",
        "HIGHTOLOWINFERENCE": "MoveHighToLowActionInference",
    }
    return mapping.get(name, name)


def _build_task_actions(task_cfg: DictConfig, sim_cfg: DictConfig) -> DictConfig:
    actions = OmegaConf.create({})
    tilt_angle = getattr(sim_cfg, "TILT_ANGLE", 15)
    for action_name in list(getattr(task_cfg, "POSSIBLE_ACTIONS", [])):
        action_cfg = OmegaConf.create({"type": _action_type_from_name(action_name)})
        if action_cfg.type in {
            "MoveForwardAction",
            "TurnLeftAction",
            "TurnRightAction",
            "LookUpAction",
            "LookDownAction",
        }:
            action_cfg.tilt_angle = tilt_angle
        action_key = (
            "hightolow"
            if action_name in {"HIGHTOLOW", "HIGHTOLOWEVAL", "HIGHTOLOWINFERENCE"}
            else action_name.lower()
        )
        actions[action_key] = action_cfg
    return actions


def _ensure_runtime_compat(cfg: DictConfig) -> DictConfig:
    if "SIMULATOR" in cfg:
        _build_simulator_agents(cfg.SIMULATOR)

    if "TASK" in cfg:
        task_cfg = cfg.TASK
        sim_cfg = cfg.SIMULATOR if "SIMULATOR" in cfg else OmegaConf.create({})
        env_cfg = cfg.ENVIRONMENT if "ENVIRONMENT" in cfg else OmegaConf.create({})

        task_cfg.lab_sensors = _build_task_lab_sensors(task_cfg)
        task_cfg.measurements = _build_task_measurements(task_cfg, env_cfg)
        task_cfg.actions = _build_task_actions(task_cfg, sim_cfg)
        if "PHYSICS_TARGET_SPS" in task_cfg:
            task_cfg.physics_target_sps = task_cfg.PHYSICS_TARGET_SPS
        elif "physics_target_sps" not in task_cfg:
            task_cfg.physics_target_sps = 60.0

    return cfg


def _ensure_task_defaults(cfg: DictConfig) -> DictConfig:
    if "SEED" not in cfg:
        cfg.SEED = 100

    if "ENVIRONMENT" not in cfg:
        cfg.ENVIRONMENT = {}
    if "ITERATOR_OPTIONS" not in cfg.ENVIRONMENT:
        cfg.ENVIRONMENT.ITERATOR_OPTIONS = {}
    if "SHUFFLE" not in cfg.ENVIRONMENT.ITERATOR_OPTIONS:
        cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
    if "MAX_SCENE_REPEAT_STEPS" not in cfg.ENVIRONMENT.ITERATOR_OPTIONS:
        cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    if "MAX_EPISODE_SECONDS" not in cfg.ENVIRONMENT:
        cfg.ENVIRONMENT.MAX_EPISODE_SECONDS = 10000000
    if "MAX_EPISODE_STEPS" not in cfg.ENVIRONMENT:
        cfg.ENVIRONMENT.MAX_EPISODE_STEPS = 1000

    if "DATASET" not in cfg:
        cfg.DATASET = {}
    if "CONTENT_SCENES" not in cfg.DATASET:
        cfg.DATASET.CONTENT_SCENES = ["*"]
    if "ROLES" not in cfg.DATASET:
        cfg.DATASET.ROLES = ["guide"]
    if "LANGUAGES" not in cfg.DATASET:
        cfg.DATASET.LANGUAGES = ["*"]

    if "SIMULATOR" in cfg and getattr(cfg.SIMULATOR, "TYPE", None) == "Sim-v1":
        cfg.SIMULATOR.TYPE = "Sim-v0"
    if "SIMULATOR" in cfg and "HABITAT_SIM_V0" in cfg.SIMULATOR:
        _set_default(cfg.SIMULATOR.HABITAT_SIM_V0, "GPU_GPU", False)
        cfg.SIMULATOR.HABITAT_SIM_V0.gpu_gpu = cfg.SIMULATOR.HABITAT_SIM_V0.GPU_GPU

    return _ensure_runtime_compat(cfg)


# 新加: 只做大小写别名映射，不再调用 _ensure_runtime_compat。
# 旧版会在每次递归返回时触发 _ensure_runtime_compat → _build_simulator_agents，
# 而 OmegaConf 赋值是深拷贝（非引用），导致先创建的小写别名（如 "simulator"）
# 与后续对大写键（如 "SIMULATOR"）的修改脱节。
def sync_task_config_keys(cfg: DictConfig) -> DictConfig:
    for key in list(cfg.keys()):
        value = cfg[key]
        if isinstance(value, DictConfig):
            sync_task_config_keys(value)
        lower_key = str(key).lower()
        if lower_key != key and lower_key not in cfg:
            cfg[lower_key] = value
    return cfg

# def sync_task_config_keys(cfg: DictConfig) -> DictConfig:
#     for key in list(cfg.keys()):
#         value = cfg[key]
#         if isinstance(value, DictConfig):
#             sync_task_config_keys(value)
#         lower_key = str(key).lower()
#         if lower_key != key:
#             cfg[lower_key] = value
#     return cfg

def to_runtime_task_config(cfg) -> DictConfig:
    if isinstance(cfg, DictConfig):
        runtime_cfg = _clone_cfg(cfg)
    elif hasattr(cfg, "to_dict"):
        runtime_cfg = OmegaConf.create(cfg.to_dict())
    else:
        runtime_cfg = OmegaConf.create(cfg)
    # 新加: 先完成所有结构性变换（sensor / action / measurement 构建），
    # 再统一做大小写别名映射——确保别名拿到的是最终状态的快照。
    _ensure_task_defaults(runtime_cfg)
    sync_task_config_keys(runtime_cfg)
    # 新加: 显式重同步 simulator 节点。get_extended_config 首次调用时
    # sync_task_config_keys 创建了 "simulator" 小写副本（只含 2 个传感器），
    # 之后 eval 代码向 SIMULATOR 添加了 22 个全景相机传感器，但 "simulator"
    # 副本不受影响（OmegaConf 赋值是深拷贝）。sync_task_config_keys 遇到
    # 已存在的 "simulator" 键时会跳过。这里强制用 SIMULATOR 的最终状态覆盖，
    # 确保 Habitat Env (通过 config.simulator.agents.sim_sensors) 能看到全部传感器。
        # 只对 SIMULATOR 这条链做定点重同步，别全局覆盖
        # 只对 SIMULATOR 这条链做精确同步，避免污染 TASK.measurements 等结构
    if "SIMULATOR" in runtime_cfg:
        gpu_id = None
        sensors = None

        if "HABITAT_SIM_V0" in runtime_cfg.SIMULATOR and "GPU_DEVICE_ID" in runtime_cfg.SIMULATOR.HABITAT_SIM_V0:
            gpu_id = int(runtime_cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID)
            OmegaConf.update(
                runtime_cfg,
                "SIMULATOR.HABITAT_SIM_V0.gpu_device_id",
                gpu_id,
                force_add=True,
            )

        if "AGENT_0" in runtime_cfg.SIMULATOR and "SENSORS" in runtime_cfg.SIMULATOR.AGENT_0:
            sensors = runtime_cfg.SIMULATOR.AGENT_0.SENSORS
            OmegaConf.update(
                runtime_cfg,
                "SIMULATOR.AGENT_0.sensors",
                sensors,
                force_add=True,
            )

        # 不直接引用，显式 clone 一份，避免旧的小写副本残留
        runtime_cfg.simulator = _clone_cfg(runtime_cfg.SIMULATOR)

        if gpu_id is not None:
            OmegaConf.update(
                runtime_cfg,
                "simulator.habitat_sim_v0.gpu_device_id",
                gpu_id,
                force_add=True,
            )

        if sensors is not None:
            OmegaConf.update(
                runtime_cfg,
                "simulator.agent_0.sensors",
                sensors,
                force_add=True,
            )

    return runtime_cfg


def get_extended_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> DictConfig:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    # 新加: 始终返回适配新版 Habitat 运行时的数据结构。
    return to_runtime_task_config(config)
