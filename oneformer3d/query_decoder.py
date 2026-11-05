import torch  # c1
import torch.nn as nn

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


class CrossAttentionLayer(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # todo: why BaseModule doesn't call it without us?
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).

        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources)):
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).

        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z, _ = self.attn(y, y, y)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).

        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out

@MODELS.register_module()
class QueryDecoder(BaseModule):
    """Query decoder.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    # def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
    #              num_classes, in_channels, d_model, num_heads, hidden_dim,
    #              dropout, activation_fn, iter_pred, attn_mask, fix_attention,
    #              objectness_flag, **kwargs):
    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 objectness_flag,
                 multi_scale_in_channels=None,
                 **kwargs):  # 创新
        super().__init__()
        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.num_queries = num_instance_queries + num_semantic_queries
        if num_instance_queries + num_semantic_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))

            # ============================================================
            # Auxiliary Cross Attention
            # D1、D2共享辅助Attention
            # 但与D0主Attention完全独立
            # ============================================================

        self.aux_cross_attn_layers = nn.ModuleList([])

        for i in range(num_layers):
            self.aux_cross_attn_layers.append(
                    CrossAttentionLayer(
                        d_model,
                        num_heads,
                        dropout,
                        fix_attention
                    )
                )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes + 1))
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        # ============================================================
        # Residual Adaptive Multi-scale Query Fusion
        # D0为主分支，D1/D2为辅助尺度
        # ============================================================

        self.multi_scale_in_channels = multi_scale_in_channels

        self.use_multi_scale = (
                multi_scale_in_channels is not None
                and len(multi_scale_in_channels) > 1
        )

        if self.use_multi_scale:

            assert multi_scale_in_channels[0] == in_channels

            # D1、D2 -> d_model
            # D0仍使用原来的 self.input_proj
            self.extra_scale_proj = nn.ModuleList()

            for c in multi_scale_in_channels[1:]:
                self.extra_scale_proj.append(
                    nn.Sequential(
                        nn.Linear(c, d_model),
                        nn.LayerNorm(d_model),
                        nn.ReLU()
                    )
                )

            # Query自适应尺度权重
            gate_hidden = d_model // 2

            self.scale_gate = nn.Sequential(
                nn.Linear(d_model * 2, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1)
            )

            # 多尺度残差强度
            # sigmoid(-0.6931472) ≈ 0.3333
            # 实际残差强度：
            # 0.15 * sigmoid(-0.6931472) ≈ 0.05
            self.fusion_logit = nn.Parameter(
                torch.tensor(-0.6931472)
            )

            # 调试变量
            self.last_scale_weights = None
            self.last_fusion_strength = None


    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.

        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)

        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i]))
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _project_multi_scale(   #　创新
            self,
            x,
            multi_scale_feats):

        """将D0、D1、D2统一投影到256维."""

        scale_feats = []

        # ==========================================================
        # D0
        # [N0,64] -> [N0,256]
        # ==========================================================
        d0_feats = [
            self.input_proj(y)
            for y in x
        ]

        scale_feats.append(d0_feats)

        # ==========================================================
        # D1、D2
        # ==========================================================
        for scale_idx, raw_scale in enumerate(
                multi_scale_feats):
            proj = self.extra_scale_proj[
                scale_idx
            ]

            projected = [
                proj(y)
                for y in raw_scale
            ]

            scale_feats.append(projected)

        return scale_feats

    def _residual_multiscale_cross_attention(
            self,
            layer_idx,
            scale_feats,
            queries,
            fine_attn_mask=None):

        """D0主导的D1/D2自适应残差多尺度Query融合."""

        # ==========================================================
        # 1. D0 主分支
        #
        # D0 使用原 OneFormer3D 主 Cross-Attention
        # 并继续使用 fine attention mask
        # ==========================================================
        q_base = self.cross_attn_layers[layer_idx](
            scale_feats[0],
            queries,
            fine_attn_mask
        )

        # ==========================================================
        # 2. D1、D2 辅助分支
        #
        # D1、D2 共用本 Decoder Layer 的辅助 Cross-Attention
        # 但辅助 Attention 与 D0 主 Attention 参数完全独立
        # ==========================================================
        coarse_queries = []

        for scale_idx in range(1, len(scale_feats)):
            q_scale = self.aux_cross_attn_layers[layer_idx](
                scale_feats[scale_idx],
                queries,
                None
            )

            coarse_queries.append(q_scale)

        fused_queries = []
        all_weights = []

        # ==========================================================
        # 3. 对每个 batch 分别进行 Query 自适应尺度融合
        # ==========================================================
        for batch_idx in range(len(queries)):

            base = q_base[batch_idx]

            # ------------------------------------------------------
            # D1 / D2 相对于 D0 主分支 Query 的残差信息
            #
            # 注意：
            # fix_attention=True 时 CrossAttention 后还有
            # residual + LayerNorm，因此这里使用：
            #
            # q_scale - q_base
            #
            # 而不是 q_scale - 原始 queries
            # ------------------------------------------------------
            deltas = []

            for q_scale in coarse_queries:
                delta = (
                        q_scale[batch_idx]
                        - base
                )

                deltas.append(delta)

            # [Q, 2, 256]
            # 2 分别对应 D1、D2
            delta_stack = torch.stack(
                deltas,
                dim=1
            )

            # ------------------------------------------------------
            # 将 D0 主 Query 扩展到两个辅助尺度
            #
            # [Q,256]
            # ->
            # [Q,2,256]
            # ------------------------------------------------------
            base_expand = base.unsqueeze(1).expand(
                -1,
                len(deltas),
                -1
            )

            # ------------------------------------------------------
            # Query-guided adaptive scale weighting
            #
            # 每个 Query 根据：
            # D0 主分支状态 + 当前尺度残差信息
            # 自适应判断 D1 / D2 的贡献
            # ------------------------------------------------------
            gate_input = torch.cat(
                [
                    base_expand,
                    delta_stack
                ],
                dim=-1
            )

            # [Q,2,512] -> [Q,2]
            logits = self.scale_gate(
                gate_input
            ).squeeze(-1)

            # D1 轻微先验：
            # 更倾向保留浅层局部结构信息，
            # 但最终权重仍由网络自适应学习
            scale_prior = logits.new_tensor(
                [0.3, 0.0]
            )

            # D1 + D2 权重和为 1
            weights = torch.softmax(
                logits + scale_prior,
                dim=1
            )

            # ------------------------------------------------------
            # 自适应多尺度辅助残差
            #
            # auxiliary =
            # w1 * delta_D1 + w2 * delta_D2
            # ------------------------------------------------------
            auxiliary = (
                    delta_stack
                    * weights.unsqueeze(-1)
            ).sum(dim=1)

            # ------------------------------------------------------
            # 有界残差强度
            #
            # gamma ∈ (0, 0.15)
            # 初始 gamma ≈ 0.05
            # ------------------------------------------------------
            fusion_strength = (
                    0.15
                    * torch.sigmoid(self.fusion_logit)
            )

            # ======================================================
            # 最终 Query
            #
            # D0 始终作为主体，
            # D1/D2 只作为受限辅助信息加入
            # ======================================================
            fused = (
                    base
                    + fusion_strength
                    * auxiliary
            )

            fused_queries.append(fused)
            all_weights.append(weights)

        # 保存调试信息
        self.last_scale_weights = [
            w.detach()
            for w in all_weights
        ]

        self.last_fusion_strength = (
                0.15
                * torch.sigmoid(
            self.fusion_logit
        )
        ).detach()

        return fused_queries

    def forward_simple_multiscale(  # 创新
            self,
            x,
            queries,
            multi_scale_feats):

        # ============================================================
        # 四尺度统一到256维
        # ============================================================
        scale_feats = self._project_multi_scale(
            x,
            multi_scale_feats
        )

        # ============================================================
        # Mask仍然只使用D0
        # ============================================================
        mask_feats = [
            self.x_mask(y)
            for y in x
        ]

        # 402个Learnable Query
        queries = self._get_queries(
            queries,
            len(x)
        )

        # ============================================================
        # 每一层Decoder：
        #
        # Multi-scale Cross Attention
        # ↓
        # Adaptive Scale Fusion
        # ↓
        # Self Attention
        # ↓
        # FFN
        # ============================================================
        for i in range(len(self.cross_attn_layers)):
            queries = self._residual_multiscale_cross_attention(
                i,
                scale_feats,
                queries,
                fine_attn_mask=None
            )

            queries = self.self_attn_layers[i](
                queries
            )

            queries = self.ffn_layers[i](
                queries
            )

        cls_preds, pred_scores, pred_masks, _ = \
            self._forward_head(
                queries,
                mask_feats
            )

        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores
        )

    def forward_iter_pred_multiscale(
            self,
            x,
            queries,
            multi_scale_feats):

        cls_preds = []
        pred_scores = []
        pred_masks = []

        # ==========================================================
        # D0-D2
        # ==========================================================
        scale_feats = self._project_multi_scale(
            x,
            multi_scale_feats
        )
        if not hasattr(self, '_debug_shape_printed'):
            print('\n========== MULTISCALE DEBUG ==========')

            for s, feats in enumerate(scale_feats):
                print(f'Scale {s}:')
                for b, feat in enumerate(feats):
                    print(
                        f'  batch {b}: '
                        f'shape={tuple(feat.shape)}, '
                        f'mean={feat.mean().item():.6f}, '
                        f'std={feat.std().item():.6f}, '
                        f'norm={feat.norm(dim=-1).mean().item():.6f}'
                    )

            self._debug_shape_printed = False

        # 最终Mask仍使用D0
        mask_feats = [
            self.x_mask(y)
            for y in x
        ]

        queries = self._get_queries(
            queries,
            len(x)
        )

        # ==========================================================
        # 初始预测
        # ==========================================================
        cls_pred, pred_score, pred_mask, attn_mask = \
            self._forward_head(
                queries,
                mask_feats
            )

        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)

        # ==========================================================
        # Decoder
        # ==========================================================
        for i in range(
                len(self.cross_attn_layers)):
            queries = \
                self._residual_multiscale_cross_attention(
                    i,
                    scale_feats,
                    queries,
                    fine_attn_mask=attn_mask
                )

            queries = self.self_attn_layers[i](
                queries
            )

            queries = self.ffn_layers[i](
                queries
            )

            cls_pred, pred_score, \
                pred_mask, attn_mask = \
                self._forward_head(
                    queries,
                    mask_feats
                )

            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {
                'cls_preds': cls_pred,
                'masks': masks,
                'scores': scores
            }
            for cls_pred, scores, masks
            in zip(
                cls_preds[:-1],
                pred_scores[:-1],
                pred_masks[:-1]
            )
        ]

        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs
        )
    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).

        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).

        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    # def forward(self, x, queries=None):
    #     """Forward pass.
    #
    #     Args:
    #         x (List[Tensor]): of len batch_size, each of shape
    #             (n_points_i, in_channels).
    #         queries (List[Tensor], optional): of len batch_size, each of shape
    #             (n_points_i, in_channles).
    #
    #     Returns:
    #         Dict: with labels, masks, scores, and possibly aux_outputs.
    #     """
    #     if self.iter_pred:
    #         return self.forward_iter_pred(x, queries)
    #     else:
    #         return self.forward_simple(x, queries)
    def forward(  # 创新
            self,
            x,
            queries=None,
            multi_scale_feats=None):

        # ============================================================
        # 新的多尺度模式
        # ============================================================
        if (
                self.use_multi_scale
                and multi_scale_feats is not None
        ):

            if self.iter_pred:
                return self.forward_iter_pred_multiscale(
                    x,
                    queries,
                    multi_scale_feats
                )

            else:
                return self.forward_simple_multiscale(
                    x,
                    queries,
                    multi_scale_feats
                )

        # ============================================================
        # 原始OneFormer3D模式
        # 用于做Baseline，也保证其他配置不受影响
        # ============================================================
        if self.iter_pred:
            return self.forward_iter_pred(
                x,
                queries
            )

        else:
            return self.forward_simple(
                x,
                queries
            )


@MODELS.register_module()
class ScanNetQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, **kwargs)
        assert num_semantic_linears in [1, 2]
        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)

    def _forward_head(self, queries, mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks = \
            [], [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).

        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats, last_flag=True)
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).

        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
            self._forward_head(queries, mask_feats, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
                self._forward_head(queries, mask_feats, last_flag)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            dict(
                cls_preds=cls_pred,
                sem_preds=sem_pred,
                masks=masks,
                scores=scores)
            for cls_pred, sem_pred, scores, masks in zip(
                cls_preds[:-1], sem_preds[:-1],
                pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)


@MODELS.register_module()
class OneDataQueryDecoder(BaseModule):
    """Query decoder. The same as above, but for 2 datasets.

    Args:
        num_layers (int): Number of transformer layers.
        num_queries_1dataset (int): Number of queries for the first dataset.
        num_queries_2dataset (int): Number of queries for the second dataset.
        num_classes_1dataset (int): Number of classes in the first dataset.
        num_classes_2dataset (int): Number of classes in the second dataset.
        prefix_1dataset (string): Prefix for the first dataset.
        prefix_2dataset (string): Prefix for the second dataset.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self,
                 num_layers,
                 num_queries_1dataset,
                 num_queries_2dataset,
                 num_classes_1dataset,
                 num_classes_2dataset,
                 prefix_1dataset,
                 prefix_2dataset,
                 in_channels,
                 d_model,
                 num_heads,
                 hidden_dim,
                 dropout,
                 activation_fn,
                 iter_pred,
                 attn_mask,
                 fix_attention,
                 **kwargs):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.num_queries_1dataset = num_queries_1dataset
        self.num_queries_2dataset = num_queries_2dataset

        self.queries_1dataset = nn.Embedding(num_queries_1dataset, d_model)
        self.queries_2dataset = nn.Embedding(num_queries_2dataset, d_model)

        self.prefix_1dataset = prefix_1dataset
        self.prefix_2dataset = prefix_2dataset

        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls_1dataset = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes_1dataset + 1))
        self.out_cls_2dataset = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes_2dataset + 1))
        self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.num_classes_1dataset = num_classes_1dataset
        self.num_classes_2dataset = num_classes_2dataset

    def _get_queries(self, batch_size, scene_names):
        """Get query tensor.

        Args:
            batch_size (int, optional): batch size.
            scene_names (List[string]): list of len batch size, which
                contains scene names.
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """

        result_queries = []
        for i in range(batch_size):
            if self.prefix_1dataset in scene_names[i]:
                result_queries.append(self.queries_1dataset.weight)
            elif self.prefix_2dataset in scene_names[i]:
                result_queries.append(self.queries_2dataset.weight)
            else:
                raise RuntimeError(f'Invalid scene name "{scene_names[i]}".')

        return result_queries

    def _forward_head(self, queries, mask_feats, scene_names):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            scene_names (List[string]): list of len batch size, which
                contains scene names.

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor]: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])

            if self.prefix_1dataset in scene_names[i]:
                cls_preds.append(self.out_cls_1dataset(norm_query))
            elif self.prefix_2dataset in scene_names[i]:
                cls_preds.append(self.out_cls_2dataset(norm_query))
            else:
                raise RuntimeError(f'Invalid scene name "{scene_names[i]}".')


            pred_scores.append(self.out_score(norm_query))
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        return  cls_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, scene_names):
        """Simple forward pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            scene_names (List[string]): list of len batch size, which
                contains scene names.

        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(len(x), scene_names)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats, scene_names)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, scene_names):
        """Iterative forward pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            scene_names (List[string]): list of len batch size, which
                contains scene names.

        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(len(x), scene_names)
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats, scene_names)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats, scene_names)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, scene_names):
        """Forward pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            scene_names (List[string]): list of len batch size, which
                contains scene names.

        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, scene_names)
        else:
            return self.forward_simple(x, scene_names)
