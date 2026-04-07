from copy import deepcopy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space

# -----------------------------------------------------------------------------
# 配置类型兼容
# -----------------------------------------------------------------------------
# 旧版代码常直接 from habitat import Config
# 但在你当前重构过程中，可能已经引入了 shim_config 来兼容新版 habitat-baselines。
# 这里保持原逻辑：优先用 habitat.Config，失败后回退到本地 shim。
try:
    from habitat import Config
except Exception:
    from vlnce_baselines.config.shim_config import Config

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.models.etp.vlnbert_init import get_vlnbert_models
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    CLIPEncoder,
)
from vlnce_baselines.models.policy import ILPolicy

from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.models.utils import (
    angle_feature_with_ele,
    dir_angle_feature_with_ele,
    angle_feature_torch,
    length2mask,
)
import math


# -----------------------------------------------------------------------------
# Policy 注册入口
# -----------------------------------------------------------------------------
# 这是 habitat-baselines 的策略注册机制。
# 训练 / eval 时会通过 registry 名字拿到这个 policy。
@baseline_registry.register_policy
class PolicyViewSelectionETP(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        # ---------------------------------------------------------------------
        # ILPolicy 需要一个 backbone/net 和动作数
        # 这里真正的主体网络是 ETP(...)
        # ---------------------------------------------------------------------
        super().__init__(
            ETP(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        # ---------------------------------------------------------------------
        # 这里把顶层 config.TORCH_GPU_ID 同步到 MODEL.TORCH_GPU_ID，
        # 这样 ETP 内部就能直接通过 model_config.TORCH_GPU_ID 选择设备。
        # ---------------------------------------------------------------------
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


# -----------------------------------------------------------------------------
# Critic 网络
# -----------------------------------------------------------------------------
# 当前文件里虽然定义了 Critic，但在你贴出的主体 forward 逻辑中没有直接使用。
# 一般用于 PPO / RL 式 value 估计，保留这个模块以备后续扩展。
class Critic(nn.Module):
    def __init__(self, drop_ratio):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        # 输出标量 value
        return self.state2value(state).squeeze()


# -----------------------------------------------------------------------------
# ETP 主体网络
# -----------------------------------------------------------------------------
# 这是整个 policy 的主干：
# 1) language 模式：编码指令
# 2) waypoint 模式：从全景 RGB-D 预测候选 waypoint
# 3) panorama 模式：融合 12 个视角的全景特征
# 4) navigation 模式：在图结构上做导航决策
class ETP(Net):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions,
    ):
        super().__init__()

        # ---------------------------------------------------------------------
        # 新版 habitat-baselines 的 Net 抽象类要求有 hidden_size / output_size
        # 这里统一按 768 作为主干隐藏维度。
        # ---------------------------------------------------------------------
        self._hidden_size = 768

        # ---------------------------------------------------------------------
        # 设备选择
        # ---------------------------------------------------------------------
        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print('\nInitalizing the ETP model ...')

        # ---------------------------------------------------------------------
        # VLN-BERT 主体
        # ---------------------------------------------------------------------
        # 用于：
        # - 指令编码（language）
        # - 全景特征融合（panorama）
        # - 图导航决策（navigation）
        self.vln_bert = get_vlnbert_models(config=model_config)

        # ---------------------------------------------------------------------
        # 环境 dropout
        # ---------------------------------------------------------------------
        # 一般用于视觉特征 dropout，减小过拟合。
        self.drop_env = nn.Dropout(p=0.4)

        # ---------------------------------------------------------------------
        # 深度编码器
        # ---------------------------------------------------------------------
        # 当前只允许使用 VlnResnetDepthEncoder。
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"

        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        )

        # ---------------------------------------------------------------------
        # 深度空间池化
        # ---------------------------------------------------------------------
        # 把 depth encoder 的 2D feature map 压成 [B, 12, C] 这种 token 级表示。
        self.space_pool_depth = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=2)
        )

        # ---------------------------------------------------------------------
        # RGB 编码器
        # ---------------------------------------------------------------------
        # 你现在最终使用的是 CLIPEncoder，而不是 ResNet50。
        self.rgb_encoder = CLIPEncoder(self.device)

        # RGB 特征同样做空间池化
        self.space_pool_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=2)
        )

        # ---------------------------------------------------------------------
        # 全景固定索引与角度特征
        # ---------------------------------------------------------------------
        # pano_img_idxes: [0, 1, ..., 11]
        # 这里约定是“逆时针”顺序。
        self.pano_img_idxes = np.arange(0, 12, dtype=np.int64)

        # 把 12 个方向转换成逆时针角度弧度
        pano_angle_rad_c = (1 - self.pano_img_idxes / 12) * 2 * math.pi

        # 预先计算固定的 12 方向 angle feature
        self.pano_angle_fts = angle_feature_torch(torch.from_numpy(pano_angle_rad_c))

    @property
    def output_size(self):
        # ---------------------------------------------------------------------
        # 新版 Net 接口要求：返回主干输出隐藏维度
        # ---------------------------------------------------------------------
        return self._hidden_size

    @property
    def is_blind(self):
        # ---------------------------------------------------------------------
        # 若 RGB 或 Depth 编码器是 blind，则视为整体 blind
        # ---------------------------------------------------------------------
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        # ---------------------------------------------------------------------
        # 当前 policy 这里没有显式 RNN 堆叠，但为了兼容接口，返回 1
        # ---------------------------------------------------------------------
        return 1

    @property
    def recurrent_hidden_size(self):
        # 兼容新版 habitat-baselines 抽象接口
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        # 兼容新版 habitat-baselines 抽象接口
        return self._hidden_size

    def forward(
        self,
        mode=None,
        txt_ids=None,
        txt_masks=None,
        txt_embeds=None,
        waypoint_predictor=None,
        observations=None,
        in_train=True,
        rgb_fts=None,
        dep_fts=None,
        loc_fts=None,
        nav_types=None,
        view_lens=None,
        gmap_vp_ids=None,
        gmap_step_ids=None,
        gmap_img_fts=None,
        gmap_pos_fts=None,
        gmap_masks=None,
        gmap_visited_masks=None,
        gmap_pair_dists=None,
    ):
        # =====================================================================
        # 1) language 模式
        # =====================================================================
        # 输入 token ids / mask，输出语言编码
        if mode == 'language':
            encoded_sentence = self.vln_bert.forward_txt(
                txt_ids,
                txt_masks,
            )
            return encoded_sentence

        # =====================================================================
        # 2) waypoint 模式
        # =====================================================================
        # 给定当前 12 视角 RGB-D observation，预测候选 waypoint。
        elif mode == 'waypoint':
            # 当前 batch size 由 rgb observation 的 batch 维决定
            batch_size = observations['rgb'].shape[0]

            # -----------------------------------------------------------------
            # 全景编码：显式按角度顺序取 12 个视角
            # -----------------------------------------------------------------
            NUM_ANGLES = 120   # 每 3 度一个离散角度，共 120 个角度 bin
            NUM_IMGS = 12      # 12 张全景图
            NUM_CLASSES = 12   # 每个角度方向上 12 个距离 bin

            # 显式角度顺序，避免依赖 dict 插入顺序
            _ANGLE_ORDER = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

            # depth / rgb 对应 12 个视角的 key
            depth_keys = ['depth'] + [f'depth_{a}' for a in _ANGLE_ORDER[1:]]
            rgb_keys = ['rgb'] + [f'rgb_{a}' for a in _ANGLE_ORDER[1:]]

            # 检查 observation 是否缺失某个全景视角
            missing = [k for k in (depth_keys + rgb_keys) if k not in observations]
            if missing:
                raise KeyError(f"Missing pano observation keys: {missing}")

            # -----------------------------------------------------------------
            # [B, 12, ...] 堆叠全景视图
            # -----------------------------------------------------------------
            # 这里你已经做了向量化重构，替代了原来 Python 双层循环逐张赋值的慢写法。
            depth_views = torch.stack(
                [observations[k] for k in depth_keys], dim=1
            ).contiguous()
            rgb_views = torch.stack(
                [observations[k] for k in rgb_keys], dim=1
            ).contiguous()

            # -----------------------------------------------------------------
            # 重排顺序：把原始顺序改成 waypoint predictor 期望的输入顺序
            # -----------------------------------------------------------------
            # 原始逻辑里：
            #   ra_count = (NUM_IMGS - a_count) % NUM_IMGS
            # 对应最终顺序 [0, 11, 10, ..., 1]
            reorder_idx = torch.as_tensor(
                [0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                device=depth_views.device,
                dtype=torch.long,
            )

            # [B, 12, ...] -> [B*12, ...]
            depth_batch = depth_views.index_select(1, reorder_idx).flatten(0, 1).contiguous()
            rgb_batch = rgb_views.index_select(1, reorder_idx).flatten(0, 1).contiguous()

            # waypoint predictor / encoder 只需要统一的 depth / rgb key
            obs_view12 = {
                'depth': depth_batch,
                'rgb': rgb_batch,
            }

            # -----------------------------------------------------------------
            # 编码 12 个视角的 RGB / Depth
            # -----------------------------------------------------------------
            depth_embedding = self.depth_encoder(obs_view12)  # [B*12, 128, 4, 4]
            rgb_embedding = self.rgb_encoder(obs_view12)      # [B*12, 2048/512?, 7, 7] 取决于 CLIPEncoder 实现

            # -----------------------------------------------------------------
            # waypoint 预测
            # -----------------------------------------------------------------
            waypoint_heatmap_logits = waypoint_predictor(
                rgb_embedding,
                depth_embedding
            )

            # -----------------------------------------------------------------
            # 再把视角顺序转回“逆时针/全景”顺序，供后续 pano/candidate 使用
            # -----------------------------------------------------------------
            rgb_embed_reshape = rgb_embedding.reshape(
                batch_size, NUM_IMGS, 512, 1, 1
            )
            depth_embed_reshape = depth_embedding.reshape(
                batch_size, NUM_IMGS, 128, 4, 4
            )

            rgb_feats = torch.cat(
                (
                    rgb_embed_reshape[:, 0:1, :],
                    torch.flip(rgb_embed_reshape[:, 1:, :], [1]),
                ),
                dim=1,
            )
            depth_feats = torch.cat(
                (
                    depth_embed_reshape[:, 0:1, :],
                    torch.flip(depth_embed_reshape[:, 1:, :], [1]),
                ),
                dim=1,
            )

            # -----------------------------------------------------------------
            # 从热图 logits -> 角度/距离概率图
            # -----------------------------------------------------------------
            batch_x_norm = torch.softmax(
                waypoint_heatmap_logits.reshape(batch_size, NUM_ANGLES * NUM_CLASSES),
                dim=1,
            )
            batch_x_norm = batch_x_norm.reshape(batch_size, NUM_ANGLES, NUM_CLASSES)

            # 在角度维两端各补一份，用于 NMS 边界处理
            batch_x_norm_wrap = torch.cat(
                (
                    batch_x_norm[:, -1:, :],
                    batch_x_norm,
                    batch_x_norm[:, :1, :],
                ),
                dim=1,
            )

            # -----------------------------------------------------------------
            # NMS：最多保留 5 个候选 waypoint
            # -----------------------------------------------------------------
            batch_output_map = nms(
                batch_x_norm_wrap.unsqueeze(1),
                max_predictions=5,
                sigma=(7.0, 5.0),
            )

            # 去掉刚才 wrap 时补的首尾
            batch_output_map = batch_output_map.squeeze(1)[:, 1:-1, :]

            # -----------------------------------------------------------------
            # 训练阶段：对候选区域做采样增强
            # -----------------------------------------------------------------
            if in_train:
                HEATMAP_OFFSET = 5

                # 角度 offset 对齐
                batch_way_heats_regional = torch.cat(
                    (
                        waypoint_heatmap_logits[:, -HEATMAP_OFFSET:, :],
                        waypoint_heatmap_logits[:, :-HEATMAP_OFFSET, :],
                    ),
                    dim=1,
                )

                # [B, 120, 12] -> [B, 12, 10, 12]
                batch_way_heats_regional = batch_way_heats_regional.reshape(
                    batch_size, 12, 10, 12
                )

                batch_sample_angle_idxes = []
                batch_sample_distance_idxes = []

                for j in range(batch_size):
                    # 候选角度 index
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]

                    # 对应到 12 张图中的图像 index（顺时针）
                    img_idxes = ((angle_idxes.cpu().numpy() + 5) // 10)
                    img_idxes[img_idxes == 12] = 0

                    # 取候选图对应的局部 heatmap 区域
                    way_heats_regional = batch_way_heats_regional[j][img_idxes].view(
                        img_idxes.size, -1
                    )

                    # 归一化为类别分布
                    way_heats_probs = F.softmax(way_heats_regional, 1)
                    probs_c = torch.distributions.Categorical(way_heats_probs)

                    # 从候选区域采样一个 angle-distance 组合
                    way_heats_act = probs_c.sample().detach()

                    sample_angle_idxes = []
                    sample_distance_idxes = []

                    for k, way_act in enumerate(way_heats_act):
                        if img_idxes[k] != 0:
                            angle_pointer = (img_idxes[k] - 1) * 10 + 5
                        else:
                            angle_pointer = 0

                        sample_angle_idxes.append(way_act // 12 + angle_pointer)
                        sample_distance_idxes.append(way_act % 12)

                    batch_sample_angle_idxes.append(sample_angle_idxes)
                    batch_sample_distance_idxes.append(sample_distance_idxes)

            else:
                # eval 时直接使用 NMS 输出，不做采样增强
                None

            # -----------------------------------------------------------------
            # 对 12 个 pano 视角做空间池化
            # -----------------------------------------------------------------
            rgb_feats = self.space_pool_rgb(rgb_feats)
            depth_feats = self.space_pool_depth(depth_feats)

            # -----------------------------------------------------------------
            # 构建 candidate 级特征
            # -----------------------------------------------------------------
            cand_rgb = []
            cand_depth = []
            cand_angle_fts = []
            cand_img_idxes = []
            cand_angles = []
            cand_distances = []

            for j in range(batch_size):
                if in_train:
                    angle_idxes = torch.tensor(batch_sample_angle_idxes[j])
                    distance_idxes = torch.tensor(batch_sample_distance_idxes[j])
                else:
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    distance_idxes = batch_output_map[j].nonzero()[:, 1]

                # -------------------------------------------------------------
                # 角度：
                # angle_rad_c  : 顺时针
                # angle_rad_cc : 逆时针
                # -------------------------------------------------------------
                angle_rad_c = angle_idxes.cpu().float() / 120 * 2 * math.pi
                angle_rad_cc = 2 * math.pi - angle_idxes.float() / 120 * 2 * math.pi

                cand_angle_fts.append(angle_feature_torch(angle_rad_c))
                cand_angles.append(angle_rad_cc.tolist())

                # 距离 bin -> 实际距离（每格 0.25m）
                cand_distances.append(((distance_idxes + 1) * 0.25).tolist())

                # -------------------------------------------------------------
                # 角度 bin -> 对应 pano 图像 index（逆时针）
                # -------------------------------------------------------------
                img_idxes = 12 - (angle_idxes.cpu().numpy() + 5) // 10
                img_idxes[img_idxes == 12] = 0
                cand_img_idxes.append(img_idxes)

                # 取候选点对应的 rgb/depth token
                cand_rgb.append(rgb_feats[j, img_idxes, ...])
                cand_depth.append(depth_feats[j, img_idxes, ...])

            # -----------------------------------------------------------------
            # 构建 pano 级特征
            # -----------------------------------------------------------------
            pano_rgb = rgb_feats
            pano_depth = depth_feats
            pano_angle_fts = deepcopy(self.pano_angle_fts)
            pano_img_idxes = deepcopy(self.pano_img_idxes)

            # -----------------------------------------------------------------
            # 输出说明：
            # cand_angle_fts : 顺时针 angle feature
            # cand_angles    : 逆时针角度列表
            # -----------------------------------------------------------------
            outputs = {
                'cand_rgb': cand_rgb,               # List[[K, C_rgb]]
                'cand_depth': cand_depth,           # List[[K, C_depth]]
                'cand_angle_fts': cand_angle_fts,   # List[[K, 4]]
                'cand_img_idxes': cand_img_idxes,   # List[[K]]
                'cand_angles': cand_angles,         # List[[K]]
                'cand_distances': cand_distances,   # List[[K]]

                'pano_rgb': pano_rgb,               # [B, 12, C_rgb]
                'pano_depth': pano_depth,           # [B, 12, C_depth]
                'pano_angle_fts': pano_angle_fts,   # [12, 4]
                'pano_img_idxes': pano_img_idxes,   # [12]
            }

            return outputs

        # =====================================================================
        # 3) panorama 模式
        # =====================================================================
        # 输入 12 个视角的 pano token，交给 VLN-BERT 做全景级融合。
        elif mode == 'panorama':
            rgb_fts = self.drop_env(rgb_fts)
            outs = self.vln_bert.forward_panorama(
                rgb_fts,
                dep_fts,
                loc_fts,
                nav_types,
                view_lens,
            )
            return outs

        # =====================================================================
        # 4) navigation 模式
        # =====================================================================
        # 输入 graph map / language / 位置等信息，输出导航决策相关表示。
        elif mode == 'navigation':
            outs = self.vln_bert.forward_navigation(
                txt_embeds,
                txt_masks,
                gmap_vp_ids,
                gmap_step_ids,
                gmap_img_fts,
                gmap_pos_fts,
                gmap_masks,
                gmap_visited_masks,
                gmap_pair_dists,
            )
            return outs


# -----------------------------------------------------------------------------
# BertLayerNorm
# -----------------------------------------------------------------------------
# 这是一个 TF 风格 LayerNorm 实现。
# 某些旧 VLN-BERT / 自定义模块可能会用到它。
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style
        (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # 均值
        u = x.mean(-1, keepdim=True)

        # 方差
        s = (x - u).pow(2).mean(-1, keepdim=True)

        # 归一化
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)

        # 仿射变换
        return self.weight * x + self.bias