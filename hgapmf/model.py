from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .spatial_refinement_cue import SpatialRefinementCuePredictor


PROGRESSIVE_LEVELS = (2, 4, 8, 14)
OUTPUT_CHANNELS = (3, 5, 9, 15)

# Raw labels are 0 background + 14 foreground anatomical classes.
# EXP-018 uses paper-aligned levels: background + 2/4/8/14 foreground groups.
PROGRESSIVE_LABEL_MAPS: Tuple[Tuple[int, ...], ...] = (
    # P2: supra-tentorial large structures vs posterior/CSF/small structures.
    (0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2),
    # P4: cortical lobes, deep/white matter, cerebellum/brainstem, CSF/optic.
    (0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 3, 4),
    # P8: lobe pairs + deep + white + posterior solid + CSF/optic.
    (0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 7, 8),
    # P14: identity foreground classes.
    tuple(range(15)),
)


@dataclass(frozen=True)
class Config:
    prototype_dim: int = 32
    prototype_nums: Tuple[int, int, int] = (8, 16, 32)
    max_tokens: Tuple[int, int, int] = (64, 64, 128)
    hypergraph_topk: int = 4
    num_heads: int = 4
    temperature: float = 0.07


def _stats_float(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu())


def validate_progressive_label_maps(label_maps: Sequence[Sequence[int]] = PROGRESSIVE_LABEL_MAPS) -> None:
    expected_channels = OUTPUT_CHANNELS
    for idx, (mapping, channels) in enumerate(zip(label_maps, expected_channels)):
        if len(mapping) != 15:
            raise ValueError(f"EXP-018 label map {idx} must have 15 entries, got {len(mapping)}.")
        values = set(int(v) for v in mapping)
        expected = set(range(channels))
        if values != expected:
            raise ValueError(
                f"EXP-018 label map {idx} values must be {sorted(expected)}, got {sorted(values)}."
            )


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ModalityEncoder3D(nn.Module):
    """Lightweight 3D encoder"""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, levels: int = 6) -> None:
        super().__init__()
        channels = [base_channels * (2**i) for i in range(levels)]
        self.output_channels = channels
        self.blocks = nn.ModuleList()
        self.down = nn.AvgPool3d(kernel_size=2, stride=2)
        current = in_channels
        for out_channels in channels:
            self.blocks.append(ConvBlock3D(current, out_channels))
            current = out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            skips.append(x)
            if idx != len(self.blocks) - 1:
                x = self.down(x)
        return skips


class StageAnatomicalPrototypeCodebook(nn.Module):
    """Stage-specific shared anatomical prototype retrieval for CT and MR."""

    def __init__(self, in_channels: int, prototype_dim: int, num_prototypes: int, temperature: float) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.prototype_dim = int(prototype_dim)
        self.num_prototypes = int(num_prototypes)
        self.temperature = float(temperature)
        self.project_in = nn.Conv3d(in_channels, prototype_dim, kernel_size=1)
        self.project_out = nn.Conv3d(prototype_dim, in_channels, kernel_size=1)
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, prototype_dim))
        self.gamma = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.project_in.weight, nonlinearity="linear")
        nn.init.zeros_(self.project_in.bias)
        nn.init.normal_(self.project_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.project_out.bias)
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        z = self.project_in(feature)
        b, c, d, h, w = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        token_norm = F.normalize(tokens, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        logits = token_norm @ proto_norm.t()
        assignment = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        q_tokens = assignment @ self.prototypes
        q = q_tokens.transpose(1, 2).reshape(b, c, d, h, w)
        q_dense = self.gamma * self.project_out(q)
        with torch.no_grad():
            eps = 1e-8
            usage = assignment.detach().mean(dim=(0, 1)).clamp_min(eps)
            usage = usage / usage.sum().clamp_min(eps)
            entropy = -(usage * torch.log(usage)).sum()
            top1 = assignment.detach().argmax(dim=-1).flatten()
            counts = torch.bincount(top1, minlength=self.num_prototypes).float()
            ratio = counts / counts.sum().clamp_min(1.0)
            stats = {
                "assignment_entropy": _stats_float(entropy),
                "effective_num_prototypes": _stats_float(torch.exp(entropy)),
                "mean_assignment_confidence": _stats_float(assignment.detach().max(dim=-1).values.mean()),
                "usage_min": _stats_float(usage.min()),
                "usage_max": _stats_float(usage.max()),
                "usage_mean": _stats_float(usage.mean()),
                "top1_usage_ratio": _stats_float(ratio.max()),
                "gamma": _stats_float(self.gamma),
            }
        return q_dense, assignment, stats


class TokenInteraction(nn.Module):
    """Lightweight intra-/inter-modality token interaction with residual LayerNorm."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        heads = max(1, min(int(num_heads), int(dim)))
        while dim % heads != 0 and heads > 1:
            heads -= 1
        self.ct_proto_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mr_proto_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ct_mr_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mr_ct_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ct_1 = nn.LayerNorm(dim)
        self.norm_mr_1 = nn.LayerNorm(dim)
        self.norm_ct_2 = nn.LayerNorm(dim)
        self.norm_mr_2 = nn.LayerNorm(dim)

    def forward(
        self,
        ct_tokens: torch.Tensor,
        mr_tokens: torch.Tensor,
        prototype_tokens: torch.Tensor,
        available_ct: bool,
        available_mr: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if available_ct:
            ct_msg, _ = self.ct_proto_attn(ct_tokens, prototype_tokens, prototype_tokens, need_weights=False)
            ct_tokens = self.norm_ct_1(ct_tokens + ct_msg)
        if available_mr:
            mr_msg, _ = self.mr_proto_attn(mr_tokens, prototype_tokens, prototype_tokens, need_weights=False)
            mr_tokens = self.norm_mr_1(mr_tokens + mr_msg)
        if available_ct and available_mr:
            ct_cross, _ = self.ct_mr_attn(ct_tokens, mr_tokens, mr_tokens, need_weights=False)
            mr_cross, _ = self.mr_ct_attn(mr_tokens, ct_tokens, ct_tokens, need_weights=False)
            ct_tokens = self.norm_ct_2(ct_tokens + ct_cross)
            mr_tokens = self.norm_mr_2(mr_tokens + mr_cross)
        return ct_tokens, mr_tokens


class HGAPMFBlock(nn.Module):
    """
    HyperGraph-guided Anatomical Prototype Modality Fusion block.

    CT/MR tokens and shared anatomical prototypes are connected by soft/top-k
    token-to-prototype incidence. Semantic priors gate hyperedges; refinement
    cues gate residual message passing and dense feature reprojection.
    """

    def __init__(
        self,
        channels: int,
        semantic_channels: int,
        prototype_dim: int = 32,
        num_prototypes: int = 16,
        max_tokens: int = 64,
        topk: int = 4,
        num_heads: int = 4,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.semantic_channels = int(semantic_channels)
        self.prototype_dim = int(prototype_dim)
        self.num_prototypes = int(num_prototypes)
        self.max_tokens = int(max_tokens)
        self.topk = int(topk)
        self.temperature = float(temperature)

        self.codebook = StageAnatomicalPrototypeCodebook(channels, prototype_dim, num_prototypes, temperature)
        self.token_project = nn.Conv3d(channels, prototype_dim, kernel_size=1)
        self.token_interaction = TokenInteraction(prototype_dim, num_heads)
        self.edge_proj = nn.Sequential(
            nn.Linear(prototype_dim, prototype_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(prototype_dim, prototype_dim),
        )
        self.node_proj = nn.Linear(prototype_dim, prototype_dim)
        self.prototype_proj = nn.Linear(prototype_dim, prototype_dim)
        self.gamma_hg = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.semantic_gate = nn.Sequential(
            nn.Linear(semantic_channels, max(8, semantic_channels * 2)),
            nn.LeakyReLU(inplace=True),
            nn.Linear(max(8, semantic_channels * 2), num_prototypes),
            nn.Sigmoid(),
        )
        self.token_to_dense = nn.Conv3d(prototype_dim, channels, kernel_size=1)
        self.refine_ct = nn.Sequential(nn.Conv3d(channels * 2, channels, kernel_size=1), nn.Sigmoid())
        self.refine_mr = nn.Sequential(nn.Conv3d(channels * 2, channels, kernel_size=1), nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * 2 + semantic_channels + 1, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    @staticmethod
    def _grid_size(max_tokens: int) -> Tuple[int, int, int]:
        side = max(1, round(float(max_tokens) ** (1.0 / 3.0)))
        return side, side, side

    def _tokens(self, feature: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        grid = self._grid_size(self.max_tokens)
        projected = self.token_project(feature)
        pooled = F.adaptive_avg_pool3d(projected, grid)
        return pooled.flatten(2).transpose(1, 2).contiguous(), grid

    def _similarity_to_incidence(self, tokens: torch.Tensor, prototypes: torch.Tensor, available: bool) -> torch.Tensor:
        logits = F.normalize(tokens, dim=-1) @ F.normalize(prototypes, dim=-1).t()
        logits = logits / max(self.temperature, 1e-6)
        if not available:
            return torch.zeros_like(logits)
        if 0 < self.topk < prototypes.shape[0]:
            top_values, top_index = torch.topk(logits, k=self.topk, dim=-1)
            masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
            logits = masked.scatter(-1, top_index, top_values)
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _aggregate_to_edges(h: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        numerator = torch.bmm(h.transpose(1, 2), tokens)
        denominator = h.transpose(1, 2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return numerator / denominator

    def _tokens_to_dense(self, tokens: torch.Tensor, grid: Tuple[int, int, int], spatial_size: Sequence[int]) -> torch.Tensor:
        b = tokens.shape[0]
        dense = tokens.transpose(1, 2).reshape(b, self.prototype_dim, *grid)
        dense = F.interpolate(dense, size=spatial_size, mode="trilinear", align_corners=False)
        return self.token_to_dense(dense)

    def forward(
        self,
        f_ct: torch.Tensor,
        f_mr: torch.Tensor,
        semantic_prior: torch.Tensor,
        refinement_cue: torch.Tensor,
        mode: str = "ct_mr",
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if mode not in {"ct_mr", "ct_only", "mr_only"}:
            raise ValueError(f"EXP-018 unsupported mode={mode}.")
        if semantic_prior.shape[2:] != f_ct.shape[2:]:
            semantic_prior = F.interpolate(semantic_prior, size=f_ct.shape[2:], mode="trilinear", align_corners=False)
        if refinement_cue.shape[2:] != f_ct.shape[2:]:
            refinement_cue = F.interpolate(refinement_cue, size=f_ct.shape[2:], mode="trilinear", align_corners=False)
        if semantic_prior.shape[1] != self.semantic_channels:
            raise ValueError(
                f"EXP-018 semantic prior channels mismatch: got {semantic_prior.shape[1]}, expected {self.semantic_channels}."
            )
        available_ct = mode in {"ct_mr", "ct_only"}
        available_mr = mode in {"ct_mr", "mr_only"}

        f_ct_active = f_ct if available_ct else torch.zeros_like(f_ct)
        f_mr_active = f_mr if available_mr else torch.zeros_like(f_mr)
        q_ct, assign_ct, stats_ct = self.codebook(f_ct_active)
        q_mr, assign_mr, stats_mr = self.codebook(f_mr_active)
        if not available_ct:
            q_ct = torch.zeros_like(q_ct)
            assign_ct = torch.zeros_like(assign_ct)
        if not available_mr:
            q_mr = torch.zeros_like(q_mr)
            assign_mr = torch.zeros_like(assign_mr)

        ct_tokens, grid = self._tokens(f_ct_active)
        mr_tokens, _ = self._tokens(f_mr_active)
        prototypes = self.codebook.prototypes
        prototype_tokens = prototypes.unsqueeze(0).expand(f_ct.shape[0], -1, -1)
        h_ct = self._similarity_to_incidence(ct_tokens, prototypes, available_ct)
        h_mr = self._similarity_to_incidence(mr_tokens, prototypes, available_mr)
        h_p = torch.eye(self.num_prototypes, device=f_ct.device, dtype=f_ct.dtype).unsqueeze(0).expand(f_ct.shape[0], -1, -1)

        ct_tokens, mr_tokens = self.token_interaction(ct_tokens, mr_tokens, prototype_tokens, available_ct, available_mr)
        edge_parts = [self._aggregate_to_edges(h_p, prototype_tokens)]
        if available_ct:
            edge_parts.append(self._aggregate_to_edges(h_ct, ct_tokens))
        if available_mr:
            edge_parts.append(self._aggregate_to_edges(h_mr, mr_tokens))
        edge_msg = torch.stack(edge_parts, dim=0).mean(dim=0)
        semantic_context = semantic_prior.mean(dim=(2, 3, 4))
        edge_gate = self.semantic_gate(semantic_context)
        edge_msg = self.edge_proj(edge_msg) * edge_gate.unsqueeze(-1)

        refine_gate = refinement_cue.mean(dim=(1, 2, 3, 4), keepdim=False).view(-1, 1, 1)
        gamma = self.gamma_hg * refine_gate
        ct_updated = ct_tokens + gamma * self.node_proj(torch.bmm(h_ct, edge_msg)) if available_ct else ct_tokens
        mr_updated = mr_tokens + gamma * self.node_proj(torch.bmm(h_mr, edge_msg)) if available_mr else mr_tokens
        p_updated = prototype_tokens + gamma * self.prototype_proj(torch.bmm(h_p, edge_msg))

        gamma_ct = self._tokens_to_dense(ct_updated, grid, f_ct.shape[2:]) if available_ct else torch.zeros_like(f_ct)
        gamma_mr = self._tokens_to_dense(mr_updated, grid, f_mr.shape[2:]) if available_mr else torch.zeros_like(f_mr)
        ct_bar = f_ct_active + self.refine_ct(torch.cat([f_ct_active, gamma_ct + q_ct], dim=1)) * gamma_ct
        mr_bar = f_mr_active + self.refine_mr(torch.cat([f_mr_active, gamma_mr + q_mr], dim=1)) * gamma_mr

        fused = self.fuse(torch.cat([ct_bar, mr_bar, semantic_prior, refinement_cue], dim=1))
        relation = torch.cat(
            [
                ct_updated.mean(dim=1) if available_ct else torch.zeros_like(p_updated.mean(dim=1)),
                mr_updated.mean(dim=1) if available_mr else torch.zeros_like(p_updated.mean(dim=1)),
                p_updated.mean(dim=1),
                edge_gate.mean(dim=1, keepdim=True),
                edge_gate.std(dim=1, keepdim=True, unbiased=False),
            ],
            dim=1,
        )

        with torch.no_grad():
            incidence_errors = []
            if available_ct:
                incidence_errors.append((h_ct.sum(dim=-1) - 1.0).abs().max())
            if available_mr:
                incidence_errors.append((h_mr.sum(dim=-1) - 1.0).abs().max())
            incidence_error = torch.stack(incidence_errors).max() if incidence_errors else h_p.new_tensor(0.0)
            stats = {
                "ct": stats_ct,
                "mr": stats_mr,
                "assign_ct": assign_ct,
                "assign_mr": assign_mr,
                "prototypes": prototypes,
                "relation": relation,
                "hypergraph": {
                    "mode": mode,
                    "available_ct": bool(available_ct),
                    "available_mr": bool(available_mr),
                    "num_tokens_ct": float(ct_tokens.shape[1]),
                    "num_tokens_mr": float(mr_tokens.shape[1]),
                    "num_prototypes": float(self.num_prototypes),
                    "incidence_sum_error": _stats_float(incidence_error),
                    "semantic_gate_mean": _stats_float(edge_gate.mean()),
                    "semantic_gate_std": _stats_float(edge_gate.std(unbiased=False)),
                    "refinement_gate_mean": _stats_float(refine_gate.mean()),
                    "gamma_hg": _stats_float(self.gamma_hg),
                    "edge_msg_norm": _stats_float(edge_msg.norm(dim=-1).mean()),
                    "ct_delta_norm": _stats_float((ct_updated - ct_tokens).norm(dim=-1).mean()),
                    "mr_delta_norm": _stats_float((mr_updated - mr_tokens).norm(dim=-1).mean()),
                    "prototype_delta_norm": _stats_float((p_updated - prototype_tokens).norm(dim=-1).mean()),
                },
            }
        if not torch.isfinite(fused).all():
            raise FloatingPointError("EXP-018 HG-APMF block produced non-finite fused features.")
        return fused, stats


class HGAPMFNet(nn.Module):
    """Full HG-APMF with progressive anatomical recovery."""

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 15,
        base_channels: int = 16,
        config: Config = Config(),
        ct_encoder: Optional[nn.Module] = None,
        mr_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        validate_progressive_label_maps()
        if in_channels < 1:
            raise ValueError("EXP-018 requires at least one input channel.")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.progressive_levels = PROGRESSIVE_LEVELS
        self.output_channels = OUTPUT_CHANNELS
        self.config = config

        if ct_encoder is None or mr_encoder is None:
            self.ct_encoder = ModalityEncoder3D(1, base_channels=base_channels, levels=6)
            self.mr_encoder = ModalityEncoder3D(1, base_channels=base_channels, levels=6)
            channels = list(self.ct_encoder.output_channels)
            self.encoder_source = "lightweight_fallback"
        else:
            self.ct_encoder = ct_encoder
            self.mr_encoder = mr_encoder
            channels = list(self.ct_encoder.output_channels)
            self.encoder_source = "external_encoder"
        if len(channels) < 6:
            raise ValueError(f"EXP-018 requires at least 6 encoder stages, got {channels}.")

        self.bottleneck_fuse = nn.Conv3d(channels[-1] * 2, channels[-1], kernel_size=1)
        self.skip_channels = [channels[-2], channels[-3], channels[-4], channels[-5], channels[-6]]
        self.skip_fuse = nn.ModuleList([nn.Conv3d(ch * 2, ch, kernel_size=1) for ch in self.skip_channels])

        self.up_4_to_8 = nn.Conv3d(channels[-1], self.skip_channels[0], kernel_size=1)
        self.dec_8 = ConvBlock3D(self.skip_channels[0] * 2, self.skip_channels[0])
        self.head_s2 = nn.Conv3d(self.skip_channels[0], OUTPUT_CHANNELS[0], kernel_size=1)

        stage_channels = self.skip_channels[1:4]
        semantic_channels = OUTPUT_CHANNELS[:3]
        out_channels = OUTPUT_CHANNELS[1:]
        self.up_projections = nn.ModuleList(
            [nn.Conv3d(self.skip_channels[i], stage_channels[i], kernel_size=1) for i in range(3)]
        )
        self.cue_predictors = nn.ModuleList(
            [SpatialRefinementCuePredictor(ch, semantic_channels=sem) for ch, sem in zip(stage_channels, semantic_channels)]
        )
        self.hgapmf_blocks = nn.ModuleList(
            [
                HGAPMFBlock(
                    channels=ch,
                    semantic_channels=sem,
                    prototype_dim=config.prototype_dim,
                    num_prototypes=num_proto,
                    max_tokens=max_tokens,
                    topk=config.hypergraph_topk,
                    num_heads=config.num_heads,
                    temperature=config.temperature,
                )
                for ch, sem, num_proto, max_tokens in zip(
                    stage_channels,
                    semantic_channels,
                    config.prototype_nums,
                    config.max_tokens,
                )
            ]
        )
        self.stage_refine = nn.ModuleList([ConvBlock3D(ch * 2, ch) for ch in stage_channels])
        self.stage_heads = nn.ModuleList([nn.Conv3d(ch, out_ch, kernel_size=1) for ch, out_ch in zip(stage_channels[:2], out_channels[:2])])
        self.up_to_full = nn.Conv3d(self.skip_channels[3], self.skip_channels[4], kernel_size=1)
        self.dec_full = ConvBlock3D(self.skip_channels[4] * 2, self.skip_channels[4])
        self.final_head = nn.Conv3d(self.skip_channels[4], OUTPUT_CHANNELS[-1], kernel_size=1)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == ref.shape[2:]:
            return x
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def _split_inputs(self, x: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"EXP-018 expected [B,C,D,H,W], got {tuple(x.shape)}.")
        ct = x[:, 0:1] if x.shape[1] >= 1 else None
        mr = x[:, 1:2] if x.shape[1] >= 2 else None
        if mode == "ct_mr":
            if ct is None or mr is None:
                raise ValueError("EXP-018 mode=ct_mr requires CT and MR channels.")
        elif mode == "ct_only":
            if ct is None:
                raise ValueError("EXP-018 mode=ct_only requires CT channel.")
            mr = torch.zeros_like(ct)
        elif mode == "mr_only":
            if mr is None:
                if ct is None:
                    raise ValueError("EXP-018 mode=mr_only requires at least one MR-like channel.")
                mr = ct
            ct = torch.zeros_like(mr)
        else:
            raise ValueError(f"EXP-018 unsupported mode={mode}.")
        return ct, mr

    def forward_debug(self, x: torch.Tensor, mode: str = "ct_mr") -> Dict[str, object]:
        x_ct, x_mr = self._split_inputs(x, mode)
        ct_skips = self.ct_encoder(x_ct)
        mr_skips = self.mr_encoder(x_mr)
        fused_skips = [
            fuse(torch.cat([ct_skip, mr_skip], dim=1))
            for fuse, ct_skip, mr_skip in zip(self.skip_fuse, ct_skips[-2::-1], mr_skips[-2::-1])
        ]
        z = self.bottleneck_fuse(torch.cat([ct_skips[-1], mr_skips[-1]], dim=1))
        decoder_feature = F.interpolate(z, size=fused_skips[0].shape[2:], mode="trilinear", align_corners=False)
        decoder_feature = self.up_4_to_8(decoder_feature)
        decoder_feature = self.dec_8(torch.cat([decoder_feature, fused_skips[0]], dim=1))
        s2 = self.head_s2(decoder_feature)

        progressive = [s2]
        refinement: List[torch.Tensor] = []
        stage_stats: List[Dict[str, object]] = []
        relations: List[torch.Tensor] = []
        assignments: Dict[str, torch.Tensor] = {}
        shape_debug: Dict[str, object] = {"mode": mode, "encoder_source": self.encoder_source}

        for stage_index in range(3):
            ct_feature = ct_skips[-(stage_index + 3)]
            mr_feature = mr_skips[-(stage_index + 3)]
            semantic_prior = torch.softmax(progressive[-1], dim=1)
            semantic_prior = self._resize_like(semantic_prior, ct_feature)
            decoder_feature = F.interpolate(decoder_feature, size=ct_feature.shape[2:], mode="trilinear", align_corners=False)
            decoder_feature = self.up_projections[stage_index](decoder_feature)
            cue = self.cue_predictors[stage_index](ct_feature, mr_feature, semantic_prior)
            fused, stats = self.hgapmf_blocks[stage_index](ct_feature, mr_feature, semantic_prior, cue, mode=mode)
            decoder_feature = self.stage_refine[stage_index](torch.cat([decoder_feature, fused], dim=1))
            logits = self.stage_heads[stage_index](decoder_feature) if stage_index < 2 else None
            if logits is not None and not torch.isfinite(logits).all():
                raise FloatingPointError(f"EXP-018 non-finite logits at stage {stage_index}.")
            refinement.append(cue)
            if logits is not None:
                progressive.append(logits)
            stage_stats.append(stats)
            relations.append(stats["relation"])
            assignments[f"stage{stage_index}_ct"] = stats["assign_ct"]
            assignments[f"stage{stage_index}_mr"] = stats["assign_mr"]
            shape_debug[f"stage{stage_index}"] = {
                "ct_feature": tuple(ct_feature.shape),
                "mr_feature": tuple(mr_feature.shape),
                "semantic_prior": tuple(semantic_prior.shape),
                "refinement_cue": tuple(cue.shape),
                "fused_feature": tuple(fused.shape),
                "logits": tuple(logits.shape) if logits is not None else None,
            }

        decoder_feature = F.interpolate(decoder_feature, size=fused_skips[4].shape[2:], mode="trilinear", align_corners=False)
        decoder_feature = self.up_to_full(decoder_feature)
        decoder_feature = self.dec_full(torch.cat([decoder_feature, fused_skips[4]], dim=1))
        s14 = self.final_head(decoder_feature)
        if not torch.isfinite(s14).all():
            raise FloatingPointError("EXP-018 non-finite final S14 logits.")
        progressive.append(s14)
        relation = torch.stack(relations, dim=0).mean(dim=0)
        return {
            "seg": progressive[-1],
            "progressive": progressive,
            "refinement": refinement,
            "stage_stats": stage_stats,
            "assignments": assignments,
            "relations": relations,
            "relation": relation,
            "shape_debug": shape_debug,
            "input_channels_used": {
                "ct": 0,
                "mr": 1 if x.shape[1] > 1 else None,
                "ignored_extra_channels": max(int(x.shape[1]) - 2, 0),
                "mode": mode,
            },
        }

    def forward(self, x: torch.Tensor, return_debug: bool = False, mode: str | None = None):
        if mode is None:
            mode = os.environ.get("INFERENCE_MODE", "ct_mr")
        output = self.forward_debug(x, mode=mode)
        return output if return_debug else output["seg"]
