from typing import List, Optional, Union

from habitat.config.default import CONFIG_FILE_SEPARATOR
try:
    from habitat.config.default import Config as CN
except Exception:
    # 新加: 在没有旧版 habitat.Config 时回退到本地兼容配置节点。
    from navmorph_compat import CompatCN as CN

from habitat_extensions.config.default import (
    get_extended_config as get_task_config,
)

# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/vlnce_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.TRAINER_NAME = "dagger"
_C.ENV_NAME = "VLNCEDaggerEnv"
_C.local_rank = 0
_C.SIMULATOR_GPU_IDS = [0]
_C.TORCH_GPU_ID = 0
_C.TORCH_GPU_IDS = [0]
_C.GPU_NUMBERS = 1
_C.NUM_ENVIRONMENTS = 1
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR = "videos/debug"
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.RESULTS_DIR = "data/checkpoints/pretrained/evals"
_C.CHECKPOINT_FOLDER = "data/checkpoints/debug"
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints/debug"
_C.LOG_FILE = "train.log"

# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val_seen"
_C.EVAL.EPISODE_COUNT = -1
_C.EVAL.EPISODE_ID_ALLOWLIST = ""
_C.EVAL.LANGUAGES = ["en-US", "en-IN"]
_C.EVAL.SAMPLE = False
_C.EVAL.SAVE_RESULTS = True
_C.EVAL.LOG_EVERY_EPISODE = 1
_C.EVAL.EVAL_NONLEARNING = False
_C.EVAL.NONLEARNING = CN()
_C.EVAL.NONLEARNING.AGENT = "RandomAgent"

# -----------------------------------------------------------------------------
# INFERENCE CONFIG
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.SPLIT = "test"
_C.INFERENCE.LANGUAGES = ["en-US", "en-IN"]
_C.INFERENCE.SAMPLE = False
_C.INFERENCE.USE_CKPT_CONFIG = True
_C.INFERENCE.CKPT_PATH = "data/checkpoints/CMA_PM_DA_Aug.pth"
_C.INFERENCE.PREDICTIONS_FILE = "predictions.json"
_C.INFERENCE.INFERENCE_NONLEARNING = False
_C.INFERENCE.NONLEARNING = CN()
_C.INFERENCE.NONLEARNING.AGENT = "RandomAgent"
_C.INFERENCE.FORMAT = "rxr"  # either 'rxr' or 'r2r'
# -----------------------------------------------------------------------------
# IMITATION LEARNING CONFIG
# -----------------------------------------------------------------------------
_C.IL = CN()
_C.IL.lr = 2.5e-4
_C.IL.batch_size = 5
_C.IL.epochs = 4
_C.IL.use_iw = True
# inflection coefficient for RxR training set GT trajectories (guide): 1.9
# inflection coefficient for R2R training set GT trajectories: 3.2
_C.IL.inflection_weight_coef = 3.2
# load an already trained model for fine tuning
_C.IL.waypoint_aug = False
_C.IL.load_from_ckpt = False
_C.IL.ckpt_to_load = "data/checkpoints/ckpt.0.pth"
# if True, loads the optimizer state, epoch, and step_id from the ckpt dict.
_C.IL.is_requeue = False
# it True, start training from the saved epoch
# -----------------------------------------------------------------------------
# IL: RXR TRAINER CONFIG
# -----------------------------------------------------------------------------
_C.IL.RECOLLECT_TRAINER = CN()
_C.IL.RECOLLECT_TRAINER.preload_trajectories_file = True
_C.IL.RECOLLECT_TRAINER.trajectories_file = (
    "data/trajectories_dirs/debug/trajectories.json.gz"
)
# if set to a positive int, episodes with longer paths are ignored in training
_C.IL.RECOLLECT_TRAINER.max_traj_len = -1
# if set to a positive int, effective_batch_size must be some multiple of
# IL.batch_size. Gradient accumulation enables an arbitrarily high "effective"
# batch size.
_C.IL.RECOLLECT_TRAINER.effective_batch_size = -1
_C.IL.RECOLLECT_TRAINER.preload_size = 30
_C.IL.RECOLLECT_TRAINER.use_iw = True
_C.IL.RECOLLECT_TRAINER.gt_file = (
    "data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz"
)
# -----------------------------------------------------------------------------
# IL: DAGGER CONFIG
# -----------------------------------------------------------------------------
_C.IL.DAGGER = CN()
_C.IL.DAGGER.iterations = 10
_C.IL.DAGGER.update_size = 5000
_C.IL.DAGGER.p = 0.75
_C.IL.DAGGER.expert_policy_sensor = "SHORTEST_PATH_SENSOR"
_C.IL.DAGGER.expert_policy_sensor_uuid = "shortest_path_sensor"
_C.IL.DAGGER.load_space = False
# if True, load saved observation space and action space
_C.IL.DAGGER.lmdb_map_size = 1.0e12
# if True, saves data to disk in fp16 and converts back to fp32 when loading.
_C.IL.DAGGER.lmdb_fp16 = False
# How often to commit the writes to the DB, less commits is
# better, but everything must be in memory until a commit happens/
_C.IL.DAGGER.lmdb_commit_frequency = 500
# If True, load precomputed features directly from lmdb_features_dir.
_C.IL.DAGGER.preload_lmdb_features = False
_C.IL.DAGGER.lmdb_features_dir = (
    "data/trajectories_dirs/debug/trajectories.lmdb"
)
# -----------------------------------------------------------------------------
# RL CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.POLICY = CN()
_C.RL.POLICY.OBS_TRANSFORMS = CN()
_C.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = [
    "CenterCropperPerSensor",
]
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = [
    ("rgb", (224, 224)),
    ("depth", (256, 256)),
]
_C.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR = CN()
_C.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = [
    ("rgb", (224, 298)),
    ("depth", (256, 341)),
]
# -----------------------------------------------------------------------------
# MODELING CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.policy_name = "CMAPolicy"  # or "Seq2SeqPolicy"
_C.MODEL.ablate_depth = False
_C.MODEL.ablate_rgb = False
_C.MODEL.ablate_instruction = False

_C.MODEL.INSTRUCTION_ENCODER = CN()
_C.MODEL.INSTRUCTION_ENCODER.sensor_uuid = "instruction"
_C.MODEL.INSTRUCTION_ENCODER.vocab_size = 2504
_C.MODEL.INSTRUCTION_ENCODER.use_pretrained_embeddings = True
_C.MODEL.INSTRUCTION_ENCODER.embedding_file = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/embeddings.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.dataset_vocab = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/train/train.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.fine_tune_embeddings = False
_C.MODEL.INSTRUCTION_ENCODER.embedding_size = 50
_C.MODEL.INSTRUCTION_ENCODER.hidden_size = 128
_C.MODEL.INSTRUCTION_ENCODER.rnn_type = "LSTM"
_C.MODEL.INSTRUCTION_ENCODER.final_state_only = True
_C.MODEL.INSTRUCTION_ENCODER.bidirectional = False

_C.MODEL.spatial_output = True
_C.MODEL.RGB_ENCODER = CN()
_C.MODEL.RGB_ENCODER.cnn_type = "TorchVisionResNet50"
_C.MODEL.RGB_ENCODER.output_size = 256

_C.MODEL.DEPTH_ENCODER = CN()
_C.MODEL.DEPTH_ENCODER.cnn_type = "VlnResnetDepthEncoder"
_C.MODEL.DEPTH_ENCODER.output_size = 128
# type of resnet to use
_C.MODEL.DEPTH_ENCODER.backbone = "resnet50"
# path to DDPPO resnet weights
_C.MODEL.DEPTH_ENCODER.ddppo_checkpoint = (
    "data/ddppo-models/gibson-2plus-resnet50.pth"
)

_C.MODEL.STATE_ENCODER = CN()
_C.MODEL.STATE_ENCODER.hidden_size = 512
_C.MODEL.STATE_ENCODER.rnn_type = "GRU"

_C.MODEL.SEQ2SEQ = CN()
_C.MODEL.SEQ2SEQ.use_prev_action = False

_C.MODEL.PROGRESS_MONITOR = CN()
_C.MODEL.PROGRESS_MONITOR.use = False
_C.MODEL.PROGRESS_MONITOR.alpha = 1.0  # loss multiplier

# -----------------------------------------------------------------------------
# STATENAV CONFIG
# -----------------------------------------------------------------------------
_C.STATENAV = CN()
_C.STATENAV.hidden_dim = 512
_C.STATENAV.action_dim = 512
_C.STATENAV.latent_groups = 16
_C.STATENAV.latent_classes = 16
_C.STATENAV.state_bins = 10
_C.STATENAV.chunk_len = 16
_C.STATENAV.lambda_kl = 0.1
_C.STATENAV.lambda_state = 0.5
_C.STATENAV.gamma_rank = 0.3
_C.STATENAV.rank_margin = 0.05
_C.STATENAV.adapter_delta_scale = 0.1
_C.STATENAV.adapter_init_logit_scale = -3.0
_C.STATENAV.adapter_uncert_gamma = 2.0
_C.STATENAV.adapter_cautious_bias_scale = 0.02
_C.STATENAV.adapter_stop_idx = 0
_C.STATENAV.adapter_stop_exempt = True
_C.STATENAV.adapter_warmup_steps = 2000
_C.STATENAV.uncert_alpha_ent = 1.0  # unused_for_v3_main_path
_C.STATENAV.uncert_alpha_kl = 1.0  # unused_for_v3_main_path
_C.STATENAV.use_reality_calibration = False
_C.STATENAV.lambda_u = 1.0
_C.STATENAV.lambda_s = 0.0
_C.STATENAV.lambda_curve = 1.0
_C.STATENAV.lambda_imagination = 0.1
_C.STATENAV.lambda_health = 0.1
_C.STATENAV.lambda_imagination_health = 0.5
_C.STATENAV.lambda_contrast = 0.1
_C.STATENAV.lambda_attn = 0.05
_C.STATENAV.lambda_relation_aux = 0.1
_C.STATENAV.lambda_health_entropy = 0.01
_C.STATENAV.lambda_curve_health = 1.0
_C.STATENAV.lambda_attn_health = 0.25
_C.STATENAV.decision_mode = "self_state_adapter"
_C.STATENAV.eval_action_source = "statenav"
_C.STATENAV.decision_alpha_u = 0.0
_C.STATENAV.decision_beta_d = 1.0
_C.STATENAV.progress_curve_horizon = 5
_C.STATENAV.progress_curve_hidden_dim = 256
_C.STATENAV.progress_curve_smooth_window = 3
_C.STATENAV.relation_dim = 512
_C.STATENAV.use_imagination_rollout = True
_C.STATENAV.imagination_horizon = 3
_C.STATENAV.imagination_topk = 5
_C.STATENAV.dt_low_percentile = 0.75
_C.STATENAV.dt_high_percentile = 0.95
_C.STATENAV.dt_threshold_ema = 0.95
_C.STATENAV.attn_warmup_steps = 500
_C.STATENAV.high_level_backtrack_bias = 5.0
_C.STATENAV.enable_level2_frontier = True
# Keep Level-3 opt-in by default until EB-01 is closed by diagnostics.
_C.STATENAV.enable_level3_macro = False
_C.STATENAV.level3_policy = "energy_recovery"
_C.STATENAV.level3_min_step = 3
_C.STATENAV.level3_min_dnorm = 1.0
_C.STATENAV.level3_switch_margin = 0.0
_C.STATENAV.level3_allow_stop = False
_C.STATENAV.level3_stop_margin = 0.0
_C.STATENAV.backtrack_history_size = 8
_C.STATENAV.backtrack_recency_bias = 0.05
_C.STATENAV.use_relation_state_extractor = True
_C.STATENAV.use_attention_transition = True
_C.STATENAV.use_open_eye_rollout = True
_C.STATENAV.alpha_latent = 1.0
_C.STATENAV.beta_node = 1.0
_C.STATENAV.lambda_micro_kl = 0.1
_C.STATENAV.lambda_node_surprise = 0.5
_C.STATENAV.use_topo_bank = True
_C.STATENAV.use_topo_gate = True
_C.STATENAV.use_node_surprise = True
_C.STATENAV.node_surprise_hidden_dim = 256
_C.STATENAV.node_update_max_k = 5
_C.STATENAV.node_novelty_tau = 0.35
_C.STATENAV.node_decision_candidate_min = 2
_C.STATENAV.contrast_margin = 0.1
_C.STATENAV.gumbel_temp = 1.0
_C.STATENAV.hard_gumbel = True
_C.STATENAV.free_bits = 0.0  # used_in_training_kl_loss_only for V3 main path
_C.STATENAV.kl_warmup_steps = 2000
_C.STATENAV.grad_clip_norm = 5.0
_C.STATENAV.max_keep_checkpoints = 3
_C.STATENAV.stage2_etp_lr_scale = 0.1

_C.STATENAV.LOGGING = CN()
_C.STATENAV.LOGGING.use_tqdm = True
_C.STATENAV.LOGGING.step_log_every = 1
_C.STATENAV.LOGGING.write_step_metrics = True
_C.STATENAV.LOGGING.step_metrics_filename = "step_metrics.tsv"
_C.STATENAV.LOGGING.log_full_model_report_to_runtime = False
_C.STATENAV.LOGGING.save_best_by_train_loss = False
_C.STATENAV.LOGGING.log_prior_post_metrics = False
_C.STATENAV.LOGGING.log_prior_post_eval = False
_C.STATENAV.LOGGING.save_v4_episode_traces = False
_C.STATENAV.LOGGING.v4_trace_dirname = "v4_traces"
_C.STATENAV.LOGGING.v4_trace_max_episodes = 20
_C.STATENAV.LOGGING.v4_trace_topk = 5
_C.STATENAV.LOGGING.save_v5_episode_traces = False
_C.STATENAV.LOGGING.v5_trace_dirname = "v5_traces"
_C.STATENAV.LOGGING.v5_trace_max_episodes = 20
_C.STATENAV.LOGGING.v5_trace_topk = 5

_C.STATENAV.stage2_unfreeze_keywords = [
    "global_sap_head",
    "global_encoder.encoder.x_layers",
]

_C.STATENAV.LABEL_BUILDER = CN()
_C.STATENAV.LABEL_BUILDER.state_bins = 10
_C.STATENAV.LABEL_BUILDER.prefix_radius = 3.0
_C.STATENAV.LABEL_BUILDER.local_window = 1
_C.STATENAV.LABEL_BUILDER.deviation_threshold = 3.0
_C.STATENAV.LABEL_BUILDER.off_path_threshold = 4.5
_C.STATENAV.LABEL_BUILDER.heading_threshold_rad = 1.0471975512
_C.STATENAV.LABEL_BUILDER.repeated_window = 6
_C.STATENAV.LABEL_BUILDER.repeated_ratio_threshold = 0.5
_C.STATENAV.LABEL_BUILDER.candidate_instability_window = 4
_C.STATENAV.LABEL_BUILDER.candidate_instability_threshold = 0.75
_C.STATENAV.LABEL_BUILDER.recovery_tolerance_steps = 1
_C.STATENAV.LABEL_BUILDER.max_prefix_backtrack = 1
_C.STATENAV.LABEL_BUILDER.min_valid_conditions = 3

# -----------------------------------------------------------------------------
# EFES CONFIG
# -----------------------------------------------------------------------------
_C.EFES = CN()
_C.EFES.d_model = 768
_C.EFES.d_z = 256
_C.EFES.d_h = 512
_C.EFES.d_action = 128
_C.EFES.max_nodes = 50
_C.EFES.sigma_min = 0.01
_C.EFES.eps_path = 0.3
_C.EFES.history_window = 5
_C.EFES.timeout_steps = 10
_C.EFES.node_decision_candidate_min = 2
_C.EFES.tau_low = 1.0
_C.EFES.tau_high = 2.5
_C.EFES.pi_threshold = 0.5
_C.EFES.stop_idx = 0
_C.EFES.use_macro = True
_C.EFES.use_grounding = True
_C.EFES.use_confidence = True
_C.EFES.use_recovery = True
_C.EFES.phase1_iters = 10000
_C.EFES.phase2_warmup_iters = 100
_C.EFES.efes_lr_phase1 = 1.0e-4
_C.EFES.efes_lr_phase2 = 5.0e-5
_C.EFES.etp_lr_phase2 = 1.0e-5
_C.EFES.lambda_kl = 1.0
_C.EFES.lambda_node = 0.5
_C.EFES.lambda_cal = 0.1
_C.EFES.max_keep_checkpoints = 3
_C.EFES.grad_clip_norm = 5.0
_C.EFES.etp_unfreeze_keywords = [
    "global_sap_head",
    "global_encoder.encoder.x_layers",
]
_C.EFES.LOGGING = CN()
_C.EFES.LOGGING.use_tqdm = True
_C.EFES.LOGGING.step_log_every = 1
_C.EFES.LOGGING.write_step_metrics = True
_C.EFES.LOGGING.step_metrics_filename = "step_metrics.tsv"
_C.EFES.LOGGING.log_full_model_report_to_runtime = False
_C.EFES.LOGGING.save_best_by_train_loss = False
_C.EFES.LOGGING.log_diag_metrics = True

# -----------------------------------------------------------------------------
# EFES-V2 CONFIG
# -----------------------------------------------------------------------------
_C.EFES_V2 = CN()
_C.EFES_V2.d_model = 768
_C.EFES_V2.d_z = 256
_C.EFES_V2.d_h = 512
_C.EFES_V2.d_action = 128
_C.EFES_V2.max_nodes = 50
_C.EFES_V2.sigma_min = 0.01
_C.EFES_V2.diag_sigma_min = 0.05
_C.EFES_V2.diag_sigma_max = 5.0
_C.EFES_V2.eps_path = 0.3
_C.EFES_V2.history_window = 5
_C.EFES_V2.future_window = 3
_C.EFES_V2.timeout_steps = 10
_C.EFES_V2.cooldown_steps = 3
_C.EFES_V2.node_decision_candidate_min = 2
_C.EFES_V2.novelty_threshold = 0.35
_C.EFES_V2.router_temperature = 1.0
_C.EFES_V2.tau_low = 1.0
_C.EFES_V2.tau_high = 2.5
_C.EFES_V2.pi_threshold = 0.5
_C.EFES_V2.stop_idx = 0
_C.EFES_V2.belief_relax_penalty = 1.0
_C.EFES_V2.phase1_iters = 10000
_C.EFES_V2.phase2_warmup_iters = 100
_C.EFES_V2.efes_lr_phase1 = 1.0e-4
_C.EFES_V2.efes_lr_phase2 = 5.0e-5
_C.EFES_V2.etp_lr_phase2 = 1.0e-5
_C.EFES_V2.lambda_kl = 1.0
_C.EFES_V2.lambda_node = 0.5
_C.EFES_V2.lambda_pi = 0.1
_C.EFES_V2.lambda_boot_start = 0.5
_C.EFES_V2.lambda_boot_end = 0.0
_C.EFES_V2.max_keep_checkpoints = 3
_C.EFES_V2.grad_clip_norm = 5.0
_C.EFES_V2.etp_unfreeze_keywords = [
    "global_sap_head",
    "global_encoder.encoder.x_layers",
]
_C.EFES_V2.LOGGING = CN()
_C.EFES_V2.LOGGING.use_tqdm = True
_C.EFES_V2.LOGGING.step_log_every = 1
_C.EFES_V2.LOGGING.write_step_metrics = True
_C.EFES_V2.LOGGING.step_metrics_filename = "step_metrics.tsv"
_C.EFES_V2.LOGGING.log_full_model_report_to_runtime = False
_C.EFES_V2.LOGGING.save_best_by_train_loss = False
_C.EFES_V2.LOGGING.log_diag_metrics = True

# -----------------------------------------------------------------------------
# EFES-SELF CONFIG
# -----------------------------------------------------------------------------
_C.EFES_SELF = CN()
_C.EFES_SELF.d_model = 768
_C.EFES_SELF.d_z = 256
_C.EFES_SELF.d_h = 512
_C.EFES_SELF.d_action = 128
_C.EFES_SELF.max_nodes = 50
_C.EFES_SELF.sigma_min = 0.01
_C.EFES_SELF.diag_sigma_min = 0.05
_C.EFES_SELF.diag_sigma_max = 5.0
_C.EFES_SELF.eps_path = 0.3
_C.EFES_SELF.history_window = 5
_C.EFES_SELF.future_window = 3
_C.EFES_SELF.timeout_steps = 16
_C.EFES_SELF.cooldown_steps = 5
_C.EFES_SELF.node_decision_candidate_min = 3
_C.EFES_SELF.novelty_threshold = 0.45
_C.EFES_SELF.tau_low = 1.0
_C.EFES_SELF.tau_high = 2.5
_C.EFES_SELF.stop_idx = 0
_C.EFES_SELF.action_source = "etp"
_C.EFES_SELF.conditioner_max_delta = 1.0
_C.EFES_SELF.conditioner_gate_bias = -2.0
_C.EFES_SELF.conditioner_mode = "prob_mixture"
_C.EFES_SELF.conditioner_beta_max = 0.1
_C.EFES_SELF.conditioner_prob_eps = 1.0e-8
_C.EFES_SELF.conditioner_rupture_threshold = 0.75
_C.EFES_SELF.conditioner_eligibility_temperature = 0.1
_C.EFES_SELF.conditioner_stop_delta_scale = 1.0
_C.EFES_SELF.use_self_revision = True
_C.EFES_SELF.use_macro_rupture = True
_C.EFES_SELF.use_grounding_rupture = True
_C.EFES_SELF.use_typed_rupture = True
_C.EFES_SELF.phase1_iters = 10000
_C.EFES_SELF.enable_phase2 = False
_C.EFES_SELF.phase2_warmup_iters = 100
_C.EFES_SELF.efes_lr_phase1 = 1.0e-4
_C.EFES_SELF.efes_lr_phase2 = 5.0e-5
_C.EFES_SELF.etp_lr_phase2 = 1.0e-5
_C.EFES_SELF.lambda_plan = 0.05
_C.EFES_SELF.lambda_self = 0.2
_C.EFES_SELF.lambda_cons = 0.5
_C.EFES_SELF.lambda_aux = 0.0
_C.EFES_SELF.lambda_bind = 1.0
_C.EFES_SELF.lambda_kl = 1.0
_C.EFES_SELF.lambda_node = 0.25
_C.EFES_SELF.lambda_pi = 0.1
_C.EFES_SELF.lambda_condition_kl = 0.0
_C.EFES_SELF.lambda_condition_budget = 0.05
_C.EFES_SELF.conditioner_gate_budget = 0.08
_C.EFES_SELF.lambda_local = 1.0
_C.EFES_SELF.lambda_topo = 1.0
_C.EFES_SELF.lambda_prog = 1.0
_C.EFES_SELF.lambda_clarity = 1.0
_C.EFES_SELF.micro_free_nats = 3.0
_C.EFES_SELF.micro_kl_cap = 20.0
_C.EFES_SELF.max_keep_checkpoints = 3
_C.EFES_SELF.grad_clip_norm = 5.0
_C.EFES_SELF.etp_unfreeze_keywords = [
    "global_sap_head",
    "global_encoder.encoder.x_layers",
]
_C.EFES_SELF.LOGGING = CN()
_C.EFES_SELF.LOGGING.use_tqdm = True
_C.EFES_SELF.LOGGING.step_log_every = 1
_C.EFES_SELF.LOGGING.write_step_metrics = True
_C.EFES_SELF.LOGGING.step_metrics_filename = "step_metrics.tsv"
_C.EFES_SELF.LOGGING.log_full_model_report_to_runtime = False
_C.EFES_SELF.LOGGING.save_best_by_train_loss = False
_C.EFES_SELF.LOGGING.log_diag_metrics = True

# -----------------------------------------------------------------------------
# EFES-V3 CONFIG
# -----------------------------------------------------------------------------
_C.EFES_V3 = CN()
_C.EFES_V3.d_model = 768
_C.EFES_V3.d_z = 256
_C.EFES_V3.d_h = 512
_C.EFES_V3.d_action = 128
_C.EFES_V3.max_nodes = 50
_C.EFES_V3.sigma_min = 0.01
_C.EFES_V3.history_window = 5
_C.EFES_V3.lr = 1.0e-4
_C.EFES_V3.alpha = 0.5
_C.EFES_V3.beta = 0.1
_C.EFES_V3.lambda_macro = 1.0
_C.EFES_V3.grad_clip_norm = 5.0
_C.EFES_V3.max_keep_checkpoints = 3
_C.EFES_V3.gate_bias = -2.0
_C.EFES_V3.max_delta = 1.0
_C.EFES_V3.action_source = "corrected"
_C.EFES_V3.use_somatic_loop = True
_C.EFES_V3.use_predictor = True
_C.EFES_V3.use_corrector = True
_C.EFES_V3.LOGGING = CN()
_C.EFES_V3.LOGGING.use_tqdm = True
_C.EFES_V3.LOGGING.step_log_every = 1
_C.EFES_V3.LOGGING.write_step_metrics = True
_C.EFES_V3.LOGGING.step_metrics_filename = "step_metrics.tsv"
_C.EFES_V3.LOGGING.log_full_model_report_to_runtime = False
_C.EFES_V3.LOGGING.save_best_by_train_loss = False
_C.EFES_V3.LOGGING.log_diag_metrics = True


def purge_keys(config: CN, keys: List[str]) -> None:
    for k in keys:
        # 新加: 兼容新版配置结构中某些旧键已不存在的情况。
        if k in config:
            del config[k]
        config.register_deprecated_key(k)


# 新加: 将旧版平铺配置镜像到新版 habitat_baselines.* 命名空间。
def apply_habitat_baselines_namespace(config: CN) -> None:
    if not hasattr(config, "habitat_baselines"):
        # 新加: 新版 habitat-baselines 统一从 habitat_baselines 根命名空间取配置。
        config.habitat_baselines = CN()
    if not hasattr(config.habitat_baselines, "il"):
        # 新加: 兼容新版 IL 输出目录与评估结果开关的读取路径。
        config.habitat_baselines.il = CN()
    if not hasattr(config.habitat_baselines, "rl"):
        # 新加: 兼容新版 RL/obs_transforms 配置命名空间。
        config.habitat_baselines.rl = CN()
    if not hasattr(config.habitat_baselines.rl, "policy"):
        # 新加: 上游会从 habitat_baselines.rl.policy.* 读取策略配置。
        config.habitat_baselines.rl.policy = CN()

    # 新加: 将旧版顶层输出路径与设备配置镜像到新版默认字段。
    config.habitat_baselines.checkpoint_folder = config.CHECKPOINT_FOLDER
    config.habitat_baselines.tensorboard_dir = config.TENSORBOARD_DIR
    config.habitat_baselines.video_dir = config.VIDEO_DIR
    config.habitat_baselines.log_file = config.LOG_FILE
    config.habitat_baselines.num_environments = config.NUM_ENVIRONMENTS
    config.habitat_baselines.torch_gpu_id = config.TORCH_GPU_ID
    config.habitat_baselines.il.results_dir = config.RESULTS_DIR + "/{split}"
    config.habitat_baselines.il.eval_save_results = config.EVAL.SAVE_RESULTS
    config.habitat_baselines.il.output_log_dir = "data/logs"

    # 新加：把旧版 RL.POLICY.OBS_TRANSFORMS 镜像到新版 habitat_baselines.rl.policy.*。
    agent_name = "main_agent"
    if not hasattr(config.habitat_baselines.rl.policy, agent_name):
        # 新加: 新版单智能体默认按 main_agent 组织策略配置。
        setattr(config.habitat_baselines.rl.policy, agent_name, CN())

    policy_cfg = getattr(config.habitat_baselines.rl.policy, agent_name)
    # 新加: 继续沿用 ETPNav 自己的 policy_name，不引入新的策略定义。
    policy_cfg.name = getattr(config.MODEL, "policy_name", "Policy")
    policy_cfg.obs_transforms = CN()

    # 新加: 将旧版 ENABLED_TRANSFORMS 列表展开成新版逐 transform 子节点结构。
    enabled = list(getattr(config.RL.POLICY.OBS_TRANSFORMS, "ENABLED_TRANSFORMS", []))
    for transform_name in enabled:
        transform_key = transform_name.lower()
        transform_cfg = CN()
        # 新加: 上游 registry 通过 type 字段定位具体 ObservationTransformer。
        transform_cfg.type = transform_name
        transform_cfg.RL = CN()
        transform_cfg.RL.POLICY = CN()
        # 新加: 复用 ETPNav 原始 OBS_TRANSFORMS 配置，保持参数语义不变。
        transform_cfg.RL.POLICY.OBS_TRANSFORMS = config.RL.POLICY.OBS_TRANSFORMS.clone()
        setattr(policy_cfg.obs_transforms, transform_key, transform_cfg)


# 新加: 兼容旧配置系统 merge_from_list 后数值字段被保留为字符串的情况。
def coerce_legacy_scalar_types(config: CN) -> None:
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    config.local_rank = int(config.local_rank)
    config.TORCH_GPU_ID = int(config.TORCH_GPU_ID)
    config.GPU_NUMBERS = int(config.GPU_NUMBERS)
    config.NUM_ENVIRONMENTS = int(config.NUM_ENVIRONMENTS)
    config.IL.lr = float(config.IL.lr)
    config.IL.batch_size = int(config.IL.batch_size)
    config.IL.epochs = int(config.IL.epochs)
    config.STATENAV.hidden_dim = int(config.STATENAV.hidden_dim)
    config.STATENAV.action_dim = int(config.STATENAV.action_dim)
    config.STATENAV.latent_groups = int(config.STATENAV.latent_groups)
    config.STATENAV.latent_classes = int(config.STATENAV.latent_classes)
    config.STATENAV.state_bins = int(config.STATENAV.state_bins)
    config.STATENAV.chunk_len = int(config.STATENAV.chunk_len)
    config.STATENAV.lambda_kl = float(config.STATENAV.lambda_kl)
    config.STATENAV.lambda_state = float(config.STATENAV.lambda_state)
    config.STATENAV.gamma_rank = float(config.STATENAV.gamma_rank)
    config.STATENAV.rank_margin = float(config.STATENAV.rank_margin)
    config.STATENAV.adapter_delta_scale = float(config.STATENAV.adapter_delta_scale)
    config.STATENAV.adapter_init_logit_scale = float(config.STATENAV.adapter_init_logit_scale)
    config.STATENAV.adapter_uncert_gamma = float(config.STATENAV.adapter_uncert_gamma)
    config.STATENAV.adapter_cautious_bias_scale = float(config.STATENAV.adapter_cautious_bias_scale)
    config.STATENAV.adapter_stop_idx = int(config.STATENAV.adapter_stop_idx)
    config.STATENAV.adapter_stop_exempt = _to_bool(config.STATENAV.adapter_stop_exempt)
    config.STATENAV.adapter_warmup_steps = int(config.STATENAV.adapter_warmup_steps)
    config.STATENAV.uncert_alpha_ent = float(config.STATENAV.uncert_alpha_ent)
    config.STATENAV.uncert_alpha_kl = float(config.STATENAV.uncert_alpha_kl)
    config.STATENAV.use_reality_calibration = _to_bool(config.STATENAV.use_reality_calibration)
    config.STATENAV.lambda_u = float(config.STATENAV.lambda_u)
    config.STATENAV.lambda_s = float(config.STATENAV.lambda_s)
    config.STATENAV.lambda_curve = float(config.STATENAV.lambda_curve)
    config.STATENAV.lambda_imagination = float(config.STATENAV.lambda_imagination)
    config.STATENAV.lambda_health = float(config.STATENAV.lambda_health)
    config.STATENAV.lambda_imagination_health = float(config.STATENAV.lambda_imagination_health)
    config.STATENAV.lambda_contrast = float(config.STATENAV.lambda_contrast)
    config.STATENAV.lambda_attn = float(config.STATENAV.lambda_attn)
    config.STATENAV.lambda_relation_aux = float(config.STATENAV.lambda_relation_aux)
    config.STATENAV.lambda_health_entropy = float(config.STATENAV.lambda_health_entropy)
    config.STATENAV.lambda_curve_health = float(config.STATENAV.lambda_curve_health)
    config.STATENAV.lambda_attn_health = float(config.STATENAV.lambda_attn_health)
    config.STATENAV.decision_mode = str(config.STATENAV.decision_mode)
    config.STATENAV.eval_action_source = str(config.STATENAV.eval_action_source)
    config.STATENAV.decision_alpha_u = float(config.STATENAV.decision_alpha_u)
    config.STATENAV.decision_beta_d = float(config.STATENAV.decision_beta_d)
    config.STATENAV.progress_curve_horizon = int(config.STATENAV.progress_curve_horizon)
    config.STATENAV.progress_curve_hidden_dim = int(config.STATENAV.progress_curve_hidden_dim)
    config.STATENAV.progress_curve_smooth_window = int(config.STATENAV.progress_curve_smooth_window)
    config.STATENAV.relation_dim = int(config.STATENAV.relation_dim)
    config.STATENAV.use_imagination_rollout = _to_bool(config.STATENAV.use_imagination_rollout)
    config.STATENAV.imagination_horizon = int(config.STATENAV.imagination_horizon)
    config.STATENAV.imagination_topk = int(config.STATENAV.imagination_topk)
    config.STATENAV.dt_low_percentile = float(config.STATENAV.dt_low_percentile)
    config.STATENAV.dt_high_percentile = float(config.STATENAV.dt_high_percentile)
    config.STATENAV.dt_threshold_ema = float(config.STATENAV.dt_threshold_ema)
    config.STATENAV.attn_warmup_steps = int(config.STATENAV.attn_warmup_steps)
    config.STATENAV.high_level_backtrack_bias = float(config.STATENAV.high_level_backtrack_bias)
    config.STATENAV.enable_level2_frontier = _to_bool(config.STATENAV.enable_level2_frontier)
    config.STATENAV.enable_level3_macro = _to_bool(config.STATENAV.enable_level3_macro)
    config.STATENAV.level3_policy = str(config.STATENAV.level3_policy)
    config.STATENAV.level3_min_step = int(config.STATENAV.level3_min_step)
    config.STATENAV.level3_min_dnorm = float(config.STATENAV.level3_min_dnorm)
    config.STATENAV.level3_switch_margin = float(config.STATENAV.level3_switch_margin)
    config.STATENAV.level3_allow_stop = _to_bool(config.STATENAV.level3_allow_stop)
    config.STATENAV.level3_stop_margin = float(config.STATENAV.level3_stop_margin)
    config.STATENAV.backtrack_history_size = int(config.STATENAV.backtrack_history_size)
    config.STATENAV.backtrack_recency_bias = float(config.STATENAV.backtrack_recency_bias)
    config.STATENAV.use_relation_state_extractor = _to_bool(
        config.STATENAV.use_relation_state_extractor
    )
    config.STATENAV.use_attention_transition = _to_bool(
        config.STATENAV.use_attention_transition
    )
    config.STATENAV.use_open_eye_rollout = _to_bool(
        config.STATENAV.use_open_eye_rollout
    )
    config.STATENAV.alpha_latent = float(config.STATENAV.alpha_latent)
    config.STATENAV.beta_node = float(config.STATENAV.beta_node)
    config.STATENAV.lambda_micro_kl = float(config.STATENAV.lambda_micro_kl)
    config.STATENAV.lambda_node_surprise = float(config.STATENAV.lambda_node_surprise)
    config.STATENAV.use_topo_bank = _to_bool(config.STATENAV.use_topo_bank)
    config.STATENAV.use_topo_gate = _to_bool(config.STATENAV.use_topo_gate)
    config.STATENAV.use_node_surprise = _to_bool(config.STATENAV.use_node_surprise)
    config.STATENAV.node_surprise_hidden_dim = int(config.STATENAV.node_surprise_hidden_dim)
    config.STATENAV.node_update_max_k = int(config.STATENAV.node_update_max_k)
    config.STATENAV.node_novelty_tau = float(config.STATENAV.node_novelty_tau)
    config.STATENAV.node_decision_candidate_min = int(config.STATENAV.node_decision_candidate_min)
    config.STATENAV.contrast_margin = float(config.STATENAV.contrast_margin)
    config.STATENAV.gumbel_temp = float(config.STATENAV.gumbel_temp)
    config.STATENAV.hard_gumbel = _to_bool(config.STATENAV.hard_gumbel)
    config.STATENAV.free_bits = float(config.STATENAV.free_bits)
    config.STATENAV.kl_warmup_steps = int(config.STATENAV.kl_warmup_steps)
    config.STATENAV.grad_clip_norm = float(config.STATENAV.grad_clip_norm)
    config.STATENAV.max_keep_checkpoints = int(config.STATENAV.max_keep_checkpoints)
    config.STATENAV.stage2_etp_lr_scale = float(config.STATENAV.stage2_etp_lr_scale)
    config.STATENAV.LOGGING.log_prior_post_metrics = _to_bool(
        config.STATENAV.LOGGING.log_prior_post_metrics
    )
    config.STATENAV.LOGGING.log_prior_post_eval = _to_bool(
        config.STATENAV.LOGGING.log_prior_post_eval
    )
    config.STATENAV.LOGGING.save_v4_episode_traces = _to_bool(
        config.STATENAV.LOGGING.save_v4_episode_traces
    )
    config.STATENAV.LOGGING.v4_trace_dirname = str(config.STATENAV.LOGGING.v4_trace_dirname)
    config.STATENAV.LOGGING.v4_trace_max_episodes = int(
        config.STATENAV.LOGGING.v4_trace_max_episodes
    )
    config.STATENAV.LOGGING.v4_trace_topk = int(config.STATENAV.LOGGING.v4_trace_topk)
    config.STATENAV.LOGGING.save_v5_episode_traces = _to_bool(
        getattr(
            config.STATENAV.LOGGING,
            "save_v5_episode_traces",
            config.STATENAV.LOGGING.save_v4_episode_traces,
        )
    )
    config.STATENAV.LOGGING.v5_trace_dirname = str(
        getattr(config.STATENAV.LOGGING, "v5_trace_dirname", "v5_traces")
    )
    config.STATENAV.LOGGING.v5_trace_max_episodes = int(
        getattr(
            config.STATENAV.LOGGING,
            "v5_trace_max_episodes",
            config.STATENAV.LOGGING.v4_trace_max_episodes,
        )
    )
    config.STATENAV.LOGGING.v5_trace_topk = int(
        getattr(
            config.STATENAV.LOGGING,
            "v5_trace_topk",
            config.STATENAV.LOGGING.v4_trace_topk,
        )
    )
    config.STATENAV.LABEL_BUILDER.state_bins = int(config.STATENAV.LABEL_BUILDER.state_bins)
    config.STATENAV.LABEL_BUILDER.prefix_radius = float(config.STATENAV.LABEL_BUILDER.prefix_radius)
    config.STATENAV.LABEL_BUILDER.local_window = int(config.STATENAV.LABEL_BUILDER.local_window)
    config.STATENAV.LABEL_BUILDER.deviation_threshold = float(config.STATENAV.LABEL_BUILDER.deviation_threshold)
    config.STATENAV.LABEL_BUILDER.off_path_threshold = float(config.STATENAV.LABEL_BUILDER.off_path_threshold)
    config.STATENAV.LABEL_BUILDER.heading_threshold_rad = float(
        config.STATENAV.LABEL_BUILDER.heading_threshold_rad
    )
    config.STATENAV.LABEL_BUILDER.repeated_window = int(config.STATENAV.LABEL_BUILDER.repeated_window)
    config.STATENAV.LABEL_BUILDER.repeated_ratio_threshold = float(
        config.STATENAV.LABEL_BUILDER.repeated_ratio_threshold
    )
    config.STATENAV.LABEL_BUILDER.candidate_instability_window = int(
        config.STATENAV.LABEL_BUILDER.candidate_instability_window
    )
    config.STATENAV.LABEL_BUILDER.candidate_instability_threshold = float(
        config.STATENAV.LABEL_BUILDER.candidate_instability_threshold
    )
    config.STATENAV.LABEL_BUILDER.recovery_tolerance_steps = int(
        config.STATENAV.LABEL_BUILDER.recovery_tolerance_steps
    )
    config.STATENAV.LABEL_BUILDER.max_prefix_backtrack = int(
        config.STATENAV.LABEL_BUILDER.max_prefix_backtrack
    )
    config.STATENAV.LABEL_BUILDER.min_valid_conditions = int(
        config.STATENAV.LABEL_BUILDER.min_valid_conditions
    )
    config.EFES.d_model = int(config.EFES.d_model)
    config.EFES.d_z = int(config.EFES.d_z)
    config.EFES.d_h = int(config.EFES.d_h)
    config.EFES.d_action = int(config.EFES.d_action)
    config.EFES.max_nodes = int(config.EFES.max_nodes)
    config.EFES.sigma_min = float(config.EFES.sigma_min)
    config.EFES.eps_path = float(config.EFES.eps_path)
    config.EFES.history_window = int(config.EFES.history_window)
    config.EFES.timeout_steps = int(config.EFES.timeout_steps)
    config.EFES.node_decision_candidate_min = int(config.EFES.node_decision_candidate_min)
    config.EFES.tau_low = float(config.EFES.tau_low)
    config.EFES.tau_high = float(config.EFES.tau_high)
    config.EFES.pi_threshold = float(config.EFES.pi_threshold)
    config.EFES.stop_idx = int(config.EFES.stop_idx)
    config.EFES.use_macro = _to_bool(config.EFES.use_macro)
    config.EFES.use_grounding = _to_bool(config.EFES.use_grounding)
    config.EFES.use_confidence = _to_bool(config.EFES.use_confidence)
    config.EFES.use_recovery = _to_bool(config.EFES.use_recovery)
    config.EFES.phase1_iters = int(config.EFES.phase1_iters)
    config.EFES.phase2_warmup_iters = int(config.EFES.phase2_warmup_iters)
    config.EFES.efes_lr_phase1 = float(config.EFES.efes_lr_phase1)
    config.EFES.efes_lr_phase2 = float(config.EFES.efes_lr_phase2)
    config.EFES.etp_lr_phase2 = float(config.EFES.etp_lr_phase2)
    config.EFES.lambda_kl = float(config.EFES.lambda_kl)
    config.EFES.lambda_node = float(config.EFES.lambda_node)
    config.EFES.lambda_cal = float(config.EFES.lambda_cal)
    config.EFES.max_keep_checkpoints = int(config.EFES.max_keep_checkpoints)
    config.EFES.grad_clip_norm = float(config.EFES.grad_clip_norm)
    config.EFES.LOGGING.use_tqdm = _to_bool(config.EFES.LOGGING.use_tqdm)
    config.EFES.LOGGING.step_log_every = int(config.EFES.LOGGING.step_log_every)
    config.EFES.LOGGING.write_step_metrics = _to_bool(config.EFES.LOGGING.write_step_metrics)
    config.EFES.LOGGING.step_metrics_filename = str(config.EFES.LOGGING.step_metrics_filename)
    config.EFES.LOGGING.log_full_model_report_to_runtime = _to_bool(
        config.EFES.LOGGING.log_full_model_report_to_runtime
    )
    config.EFES.LOGGING.save_best_by_train_loss = _to_bool(
        config.EFES.LOGGING.save_best_by_train_loss
    )
    config.EFES.LOGGING.log_diag_metrics = _to_bool(config.EFES.LOGGING.log_diag_metrics)
    config.EFES_V2.d_model = int(config.EFES_V2.d_model)
    config.EFES_V2.d_z = int(config.EFES_V2.d_z)
    config.EFES_V2.d_h = int(config.EFES_V2.d_h)
    config.EFES_V2.d_action = int(config.EFES_V2.d_action)
    config.EFES_V2.max_nodes = int(config.EFES_V2.max_nodes)
    config.EFES_V2.sigma_min = float(config.EFES_V2.sigma_min)
    config.EFES_V2.diag_sigma_min = float(config.EFES_V2.diag_sigma_min)
    config.EFES_V2.diag_sigma_max = float(config.EFES_V2.diag_sigma_max)
    config.EFES_V2.eps_path = float(config.EFES_V2.eps_path)
    config.EFES_V2.history_window = int(config.EFES_V2.history_window)
    config.EFES_V2.future_window = int(config.EFES_V2.future_window)
    config.EFES_V2.timeout_steps = int(config.EFES_V2.timeout_steps)
    config.EFES_V2.cooldown_steps = int(config.EFES_V2.cooldown_steps)
    config.EFES_V2.node_decision_candidate_min = int(config.EFES_V2.node_decision_candidate_min)
    config.EFES_V2.novelty_threshold = float(config.EFES_V2.novelty_threshold)
    config.EFES_V2.router_temperature = float(config.EFES_V2.router_temperature)
    config.EFES_V2.tau_low = float(config.EFES_V2.tau_low)
    config.EFES_V2.tau_high = float(config.EFES_V2.tau_high)
    config.EFES_V2.pi_threshold = float(config.EFES_V2.pi_threshold)
    config.EFES_V2.stop_idx = int(config.EFES_V2.stop_idx)
    config.EFES_V2.belief_relax_penalty = float(config.EFES_V2.belief_relax_penalty)
    config.EFES_V2.phase1_iters = int(config.EFES_V2.phase1_iters)
    config.EFES_V2.phase2_warmup_iters = int(config.EFES_V2.phase2_warmup_iters)
    config.EFES_V2.efes_lr_phase1 = float(config.EFES_V2.efes_lr_phase1)
    config.EFES_V2.efes_lr_phase2 = float(config.EFES_V2.efes_lr_phase2)
    config.EFES_V2.etp_lr_phase2 = float(config.EFES_V2.etp_lr_phase2)
    config.EFES_V2.lambda_kl = float(config.EFES_V2.lambda_kl)
    config.EFES_V2.lambda_node = float(config.EFES_V2.lambda_node)
    config.EFES_V2.lambda_pi = float(config.EFES_V2.lambda_pi)
    config.EFES_V2.lambda_boot_start = float(config.EFES_V2.lambda_boot_start)
    config.EFES_V2.lambda_boot_end = float(config.EFES_V2.lambda_boot_end)
    config.EFES_V2.max_keep_checkpoints = int(config.EFES_V2.max_keep_checkpoints)
    config.EFES_V2.grad_clip_norm = float(config.EFES_V2.grad_clip_norm)
    config.EFES_V2.LOGGING.use_tqdm = _to_bool(config.EFES_V2.LOGGING.use_tqdm)
    config.EFES_V2.LOGGING.step_log_every = int(config.EFES_V2.LOGGING.step_log_every)
    config.EFES_V2.LOGGING.write_step_metrics = _to_bool(config.EFES_V2.LOGGING.write_step_metrics)
    config.EFES_V2.LOGGING.step_metrics_filename = str(config.EFES_V2.LOGGING.step_metrics_filename)
    config.EFES_V2.LOGGING.log_full_model_report_to_runtime = _to_bool(
        config.EFES_V2.LOGGING.log_full_model_report_to_runtime
    )
    config.EFES_V2.LOGGING.save_best_by_train_loss = _to_bool(
        config.EFES_V2.LOGGING.save_best_by_train_loss
    )
    config.EFES_V2.LOGGING.log_diag_metrics = _to_bool(config.EFES_V2.LOGGING.log_diag_metrics)
    config.EFES_SELF.d_model = int(config.EFES_SELF.d_model)
    config.EFES_SELF.d_z = int(config.EFES_SELF.d_z)
    config.EFES_SELF.d_h = int(config.EFES_SELF.d_h)
    config.EFES_SELF.d_action = int(config.EFES_SELF.d_action)
    config.EFES_SELF.max_nodes = int(config.EFES_SELF.max_nodes)
    config.EFES_SELF.sigma_min = float(config.EFES_SELF.sigma_min)
    config.EFES_SELF.diag_sigma_min = float(config.EFES_SELF.diag_sigma_min)
    config.EFES_SELF.diag_sigma_max = float(config.EFES_SELF.diag_sigma_max)
    config.EFES_SELF.eps_path = float(config.EFES_SELF.eps_path)
    config.EFES_SELF.history_window = int(config.EFES_SELF.history_window)
    config.EFES_SELF.future_window = int(config.EFES_SELF.future_window)
    config.EFES_SELF.timeout_steps = int(config.EFES_SELF.timeout_steps)
    config.EFES_SELF.cooldown_steps = int(config.EFES_SELF.cooldown_steps)
    config.EFES_SELF.node_decision_candidate_min = int(config.EFES_SELF.node_decision_candidate_min)
    config.EFES_SELF.novelty_threshold = float(config.EFES_SELF.novelty_threshold)
    config.EFES_SELF.tau_low = float(config.EFES_SELF.tau_low)
    config.EFES_SELF.tau_high = float(config.EFES_SELF.tau_high)
    config.EFES_SELF.stop_idx = int(config.EFES_SELF.stop_idx)
    config.EFES_SELF.action_source = str(getattr(config.EFES_SELF, "action_source", "etp")).strip().lower()
    if config.EFES_SELF.action_source not in {"etp", "efes_safe", "efes_hard"}:
        raise ValueError(
            "Unknown EFES_SELF.action_source={}. Expected one of etp, efes_safe, efes_hard.".format(
                config.EFES_SELF.action_source
            )
        )
    config.EFES_SELF.conditioner_max_delta = float(getattr(config.EFES_SELF, "conditioner_max_delta", 1.0))
    config.EFES_SELF.conditioner_gate_bias = float(getattr(config.EFES_SELF, "conditioner_gate_bias", -2.0))
    config.EFES_SELF.conditioner_mode = str(
        getattr(config.EFES_SELF, "conditioner_mode", "prob_mixture")
    ).strip().lower()
    if config.EFES_SELF.conditioner_mode not in {"prob_mixture", "logit_residual"}:
        raise ValueError(
            "Unknown EFES_SELF.conditioner_mode={}. Expected prob_mixture or logit_residual.".format(
                config.EFES_SELF.conditioner_mode
            )
        )
    config.EFES_SELF.conditioner_beta_max = float(getattr(config.EFES_SELF, "conditioner_beta_max", 0.1))
    config.EFES_SELF.conditioner_prob_eps = float(getattr(config.EFES_SELF, "conditioner_prob_eps", 1.0e-8))
    config.EFES_SELF.conditioner_rupture_threshold = float(
        getattr(config.EFES_SELF, "conditioner_rupture_threshold", 0.75)
    )
    config.EFES_SELF.conditioner_eligibility_temperature = float(
        getattr(config.EFES_SELF, "conditioner_eligibility_temperature", 0.1)
    )
    config.EFES_SELF.conditioner_stop_delta_scale = float(
        getattr(config.EFES_SELF, "conditioner_stop_delta_scale", 1.0)
    )
    config.EFES_SELF.use_self_revision = _to_bool(getattr(config.EFES_SELF, "use_self_revision", True))
    config.EFES_SELF.use_macro_rupture = _to_bool(getattr(config.EFES_SELF, "use_macro_rupture", True))
    config.EFES_SELF.use_grounding_rupture = _to_bool(getattr(config.EFES_SELF, "use_grounding_rupture", True))
    config.EFES_SELF.use_typed_rupture = _to_bool(getattr(config.EFES_SELF, "use_typed_rupture", True))
    config.EFES_SELF.phase1_iters = int(config.EFES_SELF.phase1_iters)
    config.EFES_SELF.enable_phase2 = _to_bool(config.EFES_SELF.enable_phase2)
    config.EFES_SELF.phase2_warmup_iters = int(config.EFES_SELF.phase2_warmup_iters)
    config.EFES_SELF.efes_lr_phase1 = float(config.EFES_SELF.efes_lr_phase1)
    config.EFES_SELF.efes_lr_phase2 = float(config.EFES_SELF.efes_lr_phase2)
    config.EFES_SELF.etp_lr_phase2 = float(config.EFES_SELF.etp_lr_phase2)
    config.EFES_SELF.lambda_plan = float(config.EFES_SELF.lambda_plan)
    config.EFES_SELF.lambda_self = float(getattr(config.EFES_SELF, "lambda_self", 0.2))
    config.EFES_SELF.lambda_cons = float(getattr(config.EFES_SELF, "lambda_cons", 0.5))
    config.EFES_SELF.lambda_aux = float(getattr(config.EFES_SELF, "lambda_aux", 0.0))
    config.EFES_SELF.lambda_bind = float(config.EFES_SELF.lambda_bind)
    config.EFES_SELF.lambda_kl = float(config.EFES_SELF.lambda_kl)
    config.EFES_SELF.lambda_node = float(config.EFES_SELF.lambda_node)
    config.EFES_SELF.lambda_pi = float(config.EFES_SELF.lambda_pi)
    config.EFES_SELF.lambda_condition_kl = float(getattr(config.EFES_SELF, "lambda_condition_kl", 0.0))
    config.EFES_SELF.lambda_condition_budget = float(
        getattr(config.EFES_SELF, "lambda_condition_budget", 0.05)
    )
    config.EFES_SELF.conditioner_gate_budget = float(getattr(config.EFES_SELF, "conditioner_gate_budget", 0.08))
    config.EFES_SELF.lambda_local = float(config.EFES_SELF.lambda_local)
    config.EFES_SELF.lambda_topo = float(config.EFES_SELF.lambda_topo)
    config.EFES_SELF.lambda_prog = float(config.EFES_SELF.lambda_prog)
    config.EFES_SELF.lambda_clarity = float(config.EFES_SELF.lambda_clarity)
    config.EFES_SELF.micro_free_nats = float(config.EFES_SELF.micro_free_nats)
    config.EFES_SELF.micro_kl_cap = float(config.EFES_SELF.micro_kl_cap)
    config.EFES_SELF.max_keep_checkpoints = int(config.EFES_SELF.max_keep_checkpoints)
    config.EFES_SELF.grad_clip_norm = float(config.EFES_SELF.grad_clip_norm)
    config.EFES_SELF.LOGGING.use_tqdm = _to_bool(config.EFES_SELF.LOGGING.use_tqdm)
    config.EFES_SELF.LOGGING.step_log_every = int(config.EFES_SELF.LOGGING.step_log_every)
    config.EFES_SELF.LOGGING.write_step_metrics = _to_bool(config.EFES_SELF.LOGGING.write_step_metrics)
    config.EFES_SELF.LOGGING.step_metrics_filename = str(config.EFES_SELF.LOGGING.step_metrics_filename)
    config.EFES_SELF.LOGGING.log_full_model_report_to_runtime = _to_bool(
        config.EFES_SELF.LOGGING.log_full_model_report_to_runtime
    )
    config.EFES_SELF.LOGGING.save_best_by_train_loss = _to_bool(
        config.EFES_SELF.LOGGING.save_best_by_train_loss
    )
    config.EFES_SELF.LOGGING.log_diag_metrics = _to_bool(config.EFES_SELF.LOGGING.log_diag_metrics)

    config.EFES_V3.d_model = int(config.EFES_V3.d_model)
    config.EFES_V3.d_z = int(config.EFES_V3.d_z)
    config.EFES_V3.d_h = int(config.EFES_V3.d_h)
    config.EFES_V3.d_action = int(config.EFES_V3.d_action)
    config.EFES_V3.max_nodes = int(config.EFES_V3.max_nodes)
    config.EFES_V3.sigma_min = float(config.EFES_V3.sigma_min)
    config.EFES_V3.history_window = int(config.EFES_V3.history_window)
    config.EFES_V3.lr = float(config.EFES_V3.lr)
    config.EFES_V3.alpha = float(config.EFES_V3.alpha)
    config.EFES_V3.beta = float(config.EFES_V3.beta)
    config.EFES_V3.lambda_macro = float(config.EFES_V3.lambda_macro)
    config.EFES_V3.grad_clip_norm = float(config.EFES_V3.grad_clip_norm)
    config.EFES_V3.max_keep_checkpoints = int(config.EFES_V3.max_keep_checkpoints)
    config.EFES_V3.gate_bias = float(config.EFES_V3.gate_bias)
    config.EFES_V3.max_delta = float(config.EFES_V3.max_delta)
    config.EFES_V3.action_source = str(getattr(config.EFES_V3, "action_source", "corrected")).strip().lower()
    if config.EFES_V3.action_source not in {"corrected", "etp"}:
        raise ValueError(
            "Unknown EFES_V3.action_source={}. Expected corrected or etp.".format(
                config.EFES_V3.action_source
            )
        )
    config.EFES_V3.use_somatic_loop = _to_bool(getattr(config.EFES_V3, "use_somatic_loop", True))
    config.EFES_V3.use_predictor = _to_bool(getattr(config.EFES_V3, "use_predictor", True))
    config.EFES_V3.use_corrector = _to_bool(getattr(config.EFES_V3, "use_corrector", True))
    config.EFES_V3.LOGGING.use_tqdm = _to_bool(config.EFES_V3.LOGGING.use_tqdm)
    config.EFES_V3.LOGGING.step_log_every = int(config.EFES_V3.LOGGING.step_log_every)
    config.EFES_V3.LOGGING.write_step_metrics = _to_bool(config.EFES_V3.LOGGING.write_step_metrics)
    config.EFES_V3.LOGGING.step_metrics_filename = str(config.EFES_V3.LOGGING.step_metrics_filename)
    config.EFES_V3.LOGGING.log_full_model_report_to_runtime = _to_bool(
        config.EFES_V3.LOGGING.log_full_model_report_to_runtime
    )
    config.EFES_V3.LOGGING.save_best_by_train_loss = _to_bool(
        config.EFES_V3.LOGGING.save_best_by_train_loss
    )
    config.EFES_V3.LOGGING.log_diag_metrics = _to_bool(config.EFES_V3.LOGGING.log_diag_metrics)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values. Initialized from the
    habitat_baselines default config. Overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    # 新加: 不再依赖新版 habitat-baselines 内部私有默认配置结构。
    config = _C.clone()
    purge_keys(config, ["SIMULATOR_GPU_ID", "TEST_EPISODE_COUNT"])

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        prev_task_config = ""
        for config_path in config_paths:
            config.merge_from_file(config_path)
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                config.TASK_CONFIG = get_task_config(
                    config.BASE_TASK_CONFIG_PATH
                )
                prev_task_config = config.BASE_TASK_CONFIG_PATH

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    # 新加: 在完成文件与命令行合并后，统一纠正关键标量字段的运行时类型。
    coerce_legacy_scalar_types(config)
    apply_habitat_baselines_namespace(config)

    config.freeze()
    return config
