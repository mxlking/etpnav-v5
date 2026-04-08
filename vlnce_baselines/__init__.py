from navmorph_compat import patch_habitat_compat

patch_habitat_compat()

from vlnce_baselines.trainers.train_efes import EFESTrainer  # noqa: F401
from vlnce_baselines.trainers.train_statenav_v5_stage1 import StateNavV5Stage1Trainer  # noqa: F401
from vlnce_baselines.trainers.train_statenav_v5_stage2 import StateNavV5Stage2Trainer  # noqa: F401
from vlnce_baselines.trainers.train_statenav_v6_stage1 import StateNavV6Trainer  # noqa: F401
