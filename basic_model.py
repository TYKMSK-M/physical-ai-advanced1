"""PyTorch 自作 SmolVLA の実装 (Basic Section 8 相当).

最終課題(Basic).ipynb Section 8 の教育用簡略 VLA を、
穴埋め TODO 1〜4 を実装済みの状態で移植したもの。

- SinusoidalTimeEmbedding : 時刻埋め込み (旧 TODO 1)
- TransformerBlock        : Self/Cross attention + MLP の Pre-LN Transformer 1 層
- FlowMatchingHead        : 速度場を予測する transformer
- RealVLMEncoder          : SmolVLM2-500M を凍結ロードするラッパ (条件①: VLM=SmolVLM2)
- SmolVLA                 : VLM + state + Flow Matching head の統合
  - compute_loss          : Flow Matching 損失 (旧 TODO 2, 3)
  - sample_actions        : Euler 法 ODE 積分 (旧 TODO 4)

数式の背景・実 SmolVLA との差異の詳細は notebook Section 8 の
markdown を参照。要点:
- VLM (SmolVLM2-500M) は完全凍結、Action head (数 M params) のみ学習
- Flow Matching: x_t = (1-t) x_0 + t a, v* = a - x_0, L = ||v_theta - v*||^2
- 推論は pure noise から Euler 法で t: 0 -> 1 を N=10 ステップ積分
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """スカラー時刻 t を hidden_dim 次元のベクトルに埋め込む."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # sin と cos で半分ずつ埋めるので偶数を要求
        assert hidden_dim % 2 == 0, "hidden_dim は偶数にしてください"
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # 入力 t: (B,) -> 出力: (B, hidden_dim)
        half = self.hidden_dim // 2

        # freqs[i] = 1 / 10000^(i/half) — 高周波 -> 低周波の等比数列
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )

        # ---- 旧 TODO 1: 時刻 x 周波数の外積 (broadcast) ----
        # t[:, None] (B, 1) * freqs[None, :] (1, half) -> (B, half)
        args = t[:, None] * freqs[None, :]

        # sin / cos を横に連結 -> (B, 2*half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TransformerBlock(nn.Module):
    """Transformer decoder 1 層 (Pre-LN: self-attn -> cross-attn -> MLP)."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            "hidden_dim は num_heads の倍数にしてください"
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Q/K/V/O を明示的に分離 (bias=False は LLaMA/SmolVLM 系の慣例)
        self.self_q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # cross-attn: Q は action tokens、K/V は VLM hidden state
        self.cross_q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # position-wise MLP (D -> 4D -> GELU -> D)
        self.mlp_in = nn.Linear(hidden_dim, mlp_ratio * hidden_dim, bias=False)
        self.mlp_out = nn.Linear(mlp_ratio * hidden_dim, hidden_dim, bias=False)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, D) -> (B, num_heads, L, head_dim)
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, num_heads, L, head_dim) -> (B, L, D)
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def _attention(
        self,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        o_proj: nn.Linear,
        q_input: torch.Tensor,
        k_input: torch.Tensor,
        v_input: torch.Tensor,
    ) -> torch.Tensor:
        # Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        # 射影と head 分割は自前、softmax まわりは SDPA に任せる
        Q = self._split_heads(q_proj(q_input))
        K = self._split_heads(k_proj(k_input))
        V = self._split_heads(v_proj(v_input))

        attn_out = F.scaled_dot_product_attention(Q, K, V)

        merged = self._merge_heads(attn_out)
        return o_proj(merged)

    def forward(
        self,
        action_tokens: torch.Tensor,  # (B, chunk_size, D)
        vlm_hidden: torch.Tensor,     # (B, T_vlm, D)
    ) -> torch.Tensor:                # (B, chunk_size, D)
        # Sub-layer 1: self-attention (chunk 内の相互作用)
        h = self.norm1(action_tokens)
        action_tokens = action_tokens + self._attention(
            self.self_q_proj, self.self_k_proj,
            self.self_v_proj, self.self_o_proj,
            h, h, h,
        )

        # Sub-layer 2: cross-attention (VLM 参照 = 条件付け)
        h = self.norm2(action_tokens)
        action_tokens = action_tokens + self._attention(
            self.cross_q_proj, self.cross_k_proj,
            self.cross_v_proj, self.cross_o_proj,
            h,           # Q: action
            vlm_hidden,  # K: VLM
            vlm_hidden,  # V: VLM
        )

        # Sub-layer 3: position-wise MLP
        h = self.norm3(action_tokens)
        mlp_h = F.gelu(self.mlp_in(h))
        action_tokens = action_tokens + self.mlp_out(mlp_h)

        return action_tokens


class FlowMatchingHead(nn.Module):
    """action chunk の velocity 場 v_theta(x_t, t, cond) を予測する Transformer."""

    def __init__(
        self,
        hidden_dim: int,   # D. VLM の hidden 次元と揃える必要はない (project 済み)
        action_dim: int,   # d_a. LIBERO は 7 (6-DoF + gripper)
        chunk_size: int,   # K. 未来何 step ぶんの action を一気に予測するか
        num_layers: int,   # Transformer の層数
        num_heads: int,    # multi-head の head 数 (D を割り切る必要)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        # action 次元 (7) <-> hidden 次元 (例 256) の入出力射影
        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)

        # chunk 内の位置埋め込み (learnable, zero-init)
        self.pos_embed = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))

        # 時刻 t 用の埋め込み器
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)

        self.blocks = nn.ModuleList(
            TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,  # (B, K, d_a) 中間ノイズ状態 x_t
        t: torch.Tensor,              # (B,)        時刻 t ∈ [0, 1]
        vlm_hidden: torch.Tensor,     # (B, T, D)   条件
    ) -> torch.Tensor:                # (B, K, d_a) velocity 予測
        # action 次元 -> hidden 次元 + 位置埋め込み
        x = self.action_in(noisy_actions) + self.pos_embed

        # 時刻埋め込みを chunk 全 step に broadcast 加算
        x = x + self.time_embed(t).to(dtype=x.dtype)[:, None, :]

        # Transformer 層で VLM 情報を吸わせる
        for block in self.blocks:
            x = block(x, vlm_hidden)

        # hidden 次元 -> action 次元へ戻して velocity として出力
        return self.action_out(x)


class RealVLMEncoder(nn.Module):
    """SmolVLM2 をラップ: (画像 list, task 文 list) -> hidden state.

    - VLM は凍結済みなので forward は torch.no_grad
    - processor が chat template を組み立て pixel_values / input_ids に変換
    - 最終 layer の hidden_states (画像 tokens + text tokens の融合表現) を返す
    """

    def __init__(self, model, processor, hidden_dim: int, dtype):
        super().__init__()
        self.model = model
        self.processor = processor
        self.hidden_dim = hidden_dim
        self._dtype = dtype

    @torch.no_grad()
    def forward(
        self, images: Sequence, task_texts: list[str]
    ) -> torch.Tensor:
        from PIL import Image

        # 入力を PIL に統一 (processor は PIL を期待する)
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img.astype(np.uint8)))
            elif isinstance(img, torch.Tensor):
                arr = img.cpu().numpy()
                if arr.ndim == 3 and arr.shape[0] == 3:  # (C,H,W) -> (H,W,C)
                    arr = arr.transpose(1, 2, 0)
                pil_images.append(Image.fromarray(arr.astype(np.uint8)))
            else:
                pil_images.append(img)

        # SmolVLM2 の chat template で「画像 + 指示文」の prompt に組み立てる
        prompts = []
        for txt in task_texts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": txt},
                    ],
                }
            ]
            prompts.append(
                self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            )

        inputs = self.processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )

        device = next(self.model.parameters()).device
        inputs = {
            k: (
                v.to(device=device, dtype=self._dtype)
                if v.dtype in (torch.float16, torch.float32, torch.bfloat16)
                else v.to(device)
            )
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs, output_hidden_states=True)

        # hidden_states[-1] = 最終 layer: (B, L, hidden_dim)
        return outputs.hidden_states[-1]


def build_vlm_encoder(device: str = "cuda"):
    """SmolVLM2-500M-Video-Instruct をロードして凍結された RealVLMEncoder を返す.

    Basic Section 8.4 相当。bf16 対応 GPU なら bf16、それ以外 fp16。
    """
    import torch

    from transformers import AutoModelForImageTextToText, AutoProcessor

    from . import config

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print(f"Loading VLM ({config.VLM_REPO}) in {dtype}...", flush=True)
    vlm_processor = AutoProcessor.from_pretrained(config.VLM_REPO)
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        config.VLM_REPO,
        torch_dtype=dtype,
    ).to(device)

    # 完全 freeze (VLM は学習対象外)
    vlm_model.eval()
    for p in vlm_model.parameters():
        p.requires_grad = False

    hidden_dim = vlm_model.config.text_config.hidden_size
    total = sum(p.numel() for p in vlm_model.parameters())
    trainable = sum(
        p.numel() for p in vlm_model.parameters() if p.requires_grad
    )
    print(
        f"VLM loaded. hidden_dim = {hidden_dim}, "
        f"total {total / 1e6:.1f}M, trainable {trainable / 1e6:.1f}M (0 のはず)"
    )

    return RealVLMEncoder(vlm_model, vlm_processor, hidden_dim, dtype)


class SmolVLA(nn.Module):
    """SmolVLA 全体. VLM は frozen、Action head は trainable、state は projection 経由."""

    def __init__(
        self,
        vlm_encoder: RealVLMEncoder,
        action_dim: int = 7,
        state_dim: int = 8,
        chunk_size: int = 8,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        transformer_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.vlm_encoder = vlm_encoder

        vlm_hidden = vlm_encoder.hidden_dim
        if transformer_hidden_dim is None:
            transformer_hidden_dim = vlm_hidden
        self.transformer_hidden_dim = transformer_hidden_dim

        # VLM 出力 (960) を Transformer 用の次元 (例 256) に線形射影
        if transformer_hidden_dim != vlm_hidden:
            self.vlm_projector = nn.Linear(
                vlm_hidden, transformer_hidden_dim, bias=False
            )
        else:
            self.vlm_projector = nn.Identity()

        # state (8 次元) を 1 トークンに射影 (VLA の "S" 部分)
        self.state_proj = nn.Linear(state_dim, transformer_hidden_dim)

        self.hidden_dim = transformer_hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size

        self.flow_head = FlowMatchingHead(
            hidden_dim=transformer_hidden_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
        )

        # Stage 2 (CHANGES 2026-10-29): state 正規化 stats。
        # LIBERO の eef 位置はスケールが大きく、action だけ正規化では
        # state_proj への入力がアンバランスだった。既定は identity
        # (mean=0, std=1) なので set_state_stats() を呼ばなければ従来動作。
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

    def set_state_stats(
        self, state_mean: torch.Tensor, state_std: torch.Tensor
    ) -> None:
        """state の次元別 mean/std を設定し、以後 state token を z-score 化する."""
        self.state_mean.copy_(
            state_mean.to(self.state_mean.device, dtype=self.state_mean.dtype)
        )
        self.state_std.copy_(
            state_std.to(self.state_std.device, dtype=self.state_std.dtype)
        )

    def _project_cond(
        self, hidden: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        """凍結 VLM の生出力 (B, L, 960) から学習可能な条件を組み立てる.

        C1 の VLM 特徴キャッシュでは、凍結部分 (vlm_encoder) の出力を
        事前計算・保存しておき、この学習可能な部分 (projector / state_proj)
        だけを毎回実行する。
        """
        target_dtype = next(self.flow_head.parameters()).dtype
        hidden = hidden.to(dtype=target_dtype)
        cond_vlm = self.vlm_projector(hidden)

        # state を z-score 正規化 (stats が identity なら実質 no-op)
        states_norm = (states - self.state_mean) / self.state_std
        state_token = self.state_proj(
            states_norm.to(dtype=cond_vlm.dtype, device=cond_vlm.device)
        )
        state_token = state_token.unsqueeze(1)
        return torch.cat([cond_vlm, state_token], dim=1)

    def _encode_cond(
        self, images, task_texts: list[str], states: torch.Tensor
    ) -> torch.Tensor:
        # VLM 融合表現 (勾配なし) -> 学習可能な射影 + state token
        hidden = self.vlm_encoder(images, task_texts)  # (B, L, 960)
        return self._project_cond(hidden, states)

    def compute_loss(
        self,
        images,
        task_texts: list[str],
        states: torch.Tensor,  # (B, state_dim)
        actions: torch.Tensor,  # (B, K, d_a) 正規化済み真 action
        vlm_hidden_raw: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flow Matching 学習損失 L = || v_theta(x_t, t, cond) - (a - x_0) ||^2.

        vlm_hidden_raw を渡すと凍結 VLM の forward をスキップする (C1)。
        その場合 images / task_texts は無視される。
        """
        B = actions.size(0)
        device = actions.device

        if vlm_hidden_raw is None:
            vlm_hidden = self._encode_cond(images, task_texts, states).to(device)
        else:
            vlm_hidden = self._project_cond(vlm_hidden_raw, states).to(device)

        # 学習時は乱数を生成し、validation時は固定値を注入可能にする。
        if noise is None:
            noise = torch.randn_like(actions)  # x_0 ~ N(0, I)
        else:
            if noise.shape != actions.shape:
                raise ValueError("noise shape must match actions")
            noise = noise.to(device=device, dtype=actions.dtype)
        if t is None:
            t = torch.rand(B, device=device)  # t ~ U(0, 1)
        else:
            if t.shape != (B,):
                raise ValueError(f"t shape must be ({B},)")
            t = t.to(device=device, dtype=actions.dtype)

        # Step 2: 線形補間で中間点 x_t を作る
        # ---- 旧 TODO 2: x_t = (1 - t) x_0 + t a ----
        tv = t[:, None, None]  # (B, 1, 1)
        xt = (1.0 - tv) * noise + tv * actions

        # Step 3: 真の速度 (直線パスなので定数場)
        # ---- 旧 TODO 3: v* = a - x_0 ----
        target_v = actions - noise

        # Step 4: ネットワークで速度を予測
        pred_v = self.flow_head(xt, t, vlm_hidden)

        # Step 5: MSE loss
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        task_texts: list[str],
        states: torch.Tensor,  # (B, state_dim)
        num_steps: int = 10,
        solver: str = "euler",
    ) -> torch.Tensor:  # (B, K, d_a) 正規化空間の action chunk
        """ODE を数値積分して action chunk を復元.

        solver="euler": x <- x + v(x, t) * dt            (旧 TODO 4, 1 eval/step)
        solver="heun" : 2次 Heun (modified Euler / trapezoidal)。
            v1 = v(x, t),  x_pred = x + v1*dt,
            v2 = v(x_pred, t+dt),  x <- x + dt*(v1+v2)/2  (2 eval/step)
            同一 num_steps で積分誤差が O(dt^2) に下がり、
            直線パス寄りの Flow Matching でも軌道が安定する (提案 B4)。
        """
        device = next(self.flow_head.parameters()).device

        vlm_hidden = self._encode_cond(images, task_texts, states).to(device)
        B = vlm_hidden.size(0)

        # 初期化: pure noise
        x = torch.randn(B, self.chunk_size, self.action_dim, device=device)

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_now = torch.full((B,), step * dt, device=device)
            v1 = self.flow_head(x, t_now, vlm_hidden)
            if solver == "euler":
                # ---- 旧 TODO 4: Euler の 1 歩 x <- x + v * dt ----
                x = x + v1 * dt
            elif solver == "heun":
                x_pred = x + v1 * dt
                t_next = torch.full((B,), (step + 1) * dt, device=device)
                v2 = self.flow_head(x_pred, t_next, vlm_hidden)
                x = x + dt * (v1 + v2) * 0.5
            else:
                raise ValueError(f"unknown solver: {solver!r}")

        # 正規化空間の値。呼び出し側で denormalize する
        return x


def build_smolvla(vlm_encoder: RealVLMEncoder, device: str = "cuda") -> SmolVLA:
    """教材サイズ (from-scratch x 50 episode 向け) の SmolVLA を構築する."""
    from . import config

    return SmolVLA(
        vlm_encoder=vlm_encoder,
        action_dim=config.PY_ACTION_DIM,
        state_dim=config.PY_STATE_DIM,
        chunk_size=config.PY_CHUNK_SIZE,
        num_transformer_layers=config.PY_NUM_TRANSFORMER_LAYERS,
        num_heads=config.PY_NUM_HEADS,
        transformer_hidden_dim=config.PY_TRANSFORMER_HIDDEN_DIM,
    ).to(device)
