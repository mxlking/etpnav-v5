from habitat_baselines.common.baseline_registry import baseline_registry
from torch.optim import AdamW

from vlnce_baselines.trainers.train_statenav_v5_stage1 import StateNavV5Stage1Trainer, logger


@baseline_registry.register_trainer(name="StateNavV5Stage2")
class StateNavV5Stage2Trainer(StateNavV5Stage1Trainer):
    stage_name = "Stage2"

    def _configure_trainable_modules(self) -> None:
        super()._configure_trainable_modules()

        keywords = list(self.config.STATENAV.stage2_unfreeze_keywords)
        matched_params = 0
        matched_param_names = []
        etp_params = []
        for name, param in self.policy.named_parameters():
            if any(keyword in name for keyword in keywords):
                param.requires_grad_(True)
                matched_params += param.numel()
                matched_param_names.append(name)
                etp_params.append(param)

        statenav_params = [p for p in self.statenav_agent.parameters() if p.requires_grad]
        etp_lr = float(self.config.IL.lr) * float(self.config.STATENAV.stage2_etp_lr_scale)
        param_groups = [{"params": statenav_params, "lr": float(self.config.IL.lr)}]
        if etp_params:
            param_groups.append({"params": etp_params, "lr": etp_lr})
        self.optimizer = AdamW(param_groups, lr=float(self.config.IL.lr))

        if matched_params == 0:
            logger.warning("Stage2 did not match any ETP parameters to unfreeze. Check stage2_unfreeze_keywords.")
        else:
            logger.info(
                "Stage2 unfroze %d ETP parameter tensors across keywords=%s",
                len(matched_param_names),
                keywords,
            )
            logger.info(
                "Stage2 optimizer lr groups: StateNav lr=%.2e | ETP-top lr=%.2e",
                float(self.config.IL.lr),
                etp_lr,
            )
            for name in matched_param_names:
                logger.info("  [Stage2 unfrozen] %s", name)
