"""Block-parallel SimBa networks for SAC and PPO.

Ports the block-parallel primitives from RoboNuke/Continuous_Force_RL/models/block_simba.py
and exposes:

* :class:`BlockSimBaActor` — squashed-Gaussian (+ optional Bernoulli/force-zero)
  actor used by SAC and PPO for non-hybrid tasks.
* :class:`HybridControlBlockSimBaActor` — hybrid force/position actor whose
  per-axis binary selection gates a position vs a force component, with two
  joint-distribution styles (``"product"`` independent, ``"match"`` selection-
  conditioned; the latter ports the reference's hard ``HybridActionGMM``).
* :class:`BlockSimBaQCritic` — SAC Q(s, a) critic.
* :class:`BlockSimBaValueCritic` — PPO state-value V(s) critic.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


# -----------------------------
#  Squashed Gaussian utilities
# -----------------------------
def squash_log_prob_correction(u: torch.Tensor) -> torch.Tensor:
    # log(1 - tanh(u)^2) summed over last dim; numerically stable form
    return (2.0 * math.log(2.0) - 2.0 * u - 2.0 * F.softplus(-2.0 * u)).sum(dim=-1)


def safe_atanh(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.atanh(torch.clamp(a, -1.0 + eps, 1.0 - eps))


# -----------------------------
#  Block-parallel primitives
# -----------------------------
class BlockLinear(nn.Module):
    def __init__(self, num_blocks: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_blocks, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(num_blocks, out_features))
        for i in range(num_blocks):
            nn.init.kaiming_normal_(self.weight[i])
            nn.init.zeros_(self.bias[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_blocks, batch, in_features)
        return torch.einsum("nbi,noi->nbo", x, self.weight) + self.bias[:, None, :]


class BlockLayerNorm(nn.Module):
    def __init__(self, num_blocks: int, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_blocks, normalized_shape))
        self.bias = nn.Parameter(torch.zeros(num_blocks, normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) * torch.rsqrt(var + self.eps)
        return out * self.weight[:, None, :] + self.bias[:, None, :]


class BlockMLP(nn.Module):
    def __init__(self, num_blocks: int, in_dim: int, hidden_dim: int, out_dim: int, activation=None):
        super().__init__()
        self.fc1 = BlockLinear(num_blocks, in_dim, hidden_dim)
        self.fc2 = BlockLinear(num_blocks, hidden_dim, out_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc2(F.relu(self.fc1(x)))
        if self.activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.activation == "tanh":
            out = torch.tanh(out)
        return out


class BlockResidualBlock(nn.Module):
    def __init__(self, num_blocks: int, dim: int):
        super().__init__()
        self.ln = BlockLayerNorm(num_blocks, dim)
        self.fc1 = BlockLinear(num_blocks, dim, 4 * dim)
        self.fc2 = BlockLinear(num_blocks, 4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.relu(self.fc1(self.ln(x))))


# -----------------------------
#  BlockSimBa backbone
# -----------------------------
class BlockSimBa(nn.Module):
    """Block-parallel SimBa: input proj -> N residual blocks -> LN -> output proj."""

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        hidden_dim: int,
        act_dim: int,
        device,
        num_blocks: int = 2,
        use_state_dependent_std: bool = False,
    ):
        super().__init__()
        self.device = device
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        self.num_blocks = num_blocks
        self.use_state_dependent_std = use_state_dependent_std
        self.std_out_dim = act_dim if use_state_dependent_std else 0

        # Output layout per row (along last dim):
        #   [0 : act_dim)                                  -> action mean
        #   [act_dim : act_dim + std_out_dim)              -> per-action log_std (state-dep std)
        total_out = act_dim + self.std_out_dim
        self.fc_in = BlockLinear(num_agents, obs_dim, hidden_dim)
        self.resblocks = nn.ModuleList(
            [BlockResidualBlock(num_agents, hidden_dim) for _ in range(num_blocks)]
        )
        self.ln_out = BlockLayerNorm(num_agents, hidden_dim)
        self.fc_out = BlockLinear(num_agents, hidden_dim, total_out)

    def forward(self, obs_flat: torch.Tensor, num_envs: int):
        """Return ``(actions, log_std)``.

        ``log_std`` is ``None`` unless ``use_state_dependent_std`` was set; when present its
        shape is ``(num_agents * num_envs, std_out_dim)``.
        """
        obs = obs_flat.view(self.num_agents, num_envs, -1)
        x = self.fc_in(obs)
        for block in self.resblocks:
            x = block(x)
        out = self.fc_out(self.ln_out(x))

        actions = out[..., : self.act_dim]
        if self.std_out_dim > 0:
            log_std = out[..., self.act_dim : self.act_dim + self.std_out_dim].reshape(
                -1, self.std_out_dim
            )
        else:
            log_std = None

        return actions.reshape(-1, actions.shape[-1]), log_std


# -----------------------------
#  Squashed-Gaussian actor
# -----------------------------
class BlockSimBaActor(GaussianMixin, Model):
    """SAC policy: hybrid continuous + discrete (Bernoulli) action distribution,
    block-parallel across agents.

    Most action dims use a tanh-squashed Gaussian (standard SAC). Indices listed
    in ``bernoulli_action_dims`` are sampled from a Bernoulli (binary) instead;
    the {0,1} sample is mapped to {-1,+1} so Isaac Lab's BinaryJointAction sees
    the right sign convention. A straight-through estimator carries the critic's
    gradient back through the soft probability so SAC's reparameterized policy
    gradient still works for those dims.

    Reads ``inputs["observations"]`` per skrl SAC convention. ``act()`` returns
    the (mixed) action vector and a combined log_prob = continuous-squashed-
    Gaussian log_prob + Bernoulli log_prob.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        act_init_std: float = 0.60653066,
        actor_n: int = 2,
        actor_latent: int = 512,
        last_layer_scale: float = 1.0,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
        use_state_dependent_std: bool = False,
        bernoulli_action_dims: list[int] | None = None,
        force_zero_action_dims: list[int] | None = None,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        self.num_agents = num_agents
        self.use_state_dependent_std = use_state_dependent_std

        # Resolve which action dims are continuous vs Bernoulli vs force-zero.
        # Indices into the full action vector that the env consumes (so env-side
        # ordering is preserved when we reassemble below). The three sets are
        # disjoint and partition range(num_actions).
        bdims = sorted(set(bernoulli_action_dims or []))
        zdims = sorted(set(force_zero_action_dims or []))
        for d in bdims:
            if d < 0 or d >= self.num_actions:
                raise ValueError(
                    f"bernoulli_action_dims index {d} out of range [0, {self.num_actions})"
                )
        for d in zdims:
            if d < 0 or d >= self.num_actions:
                raise ValueError(
                    f"force_zero_action_dims index {d} out of range [0, {self.num_actions})"
                )
        if set(bdims) & set(zdims):
            raise ValueError(
                "bernoulli_action_dims and force_zero_action_dims must be disjoint; "
                f"overlap = {sorted(set(bdims) & set(zdims))}"
            )
        self.bernoulli_dims: list[int] = bdims
        self.force_zero_dims: list[int] = zdims
        self.continuous_dims: list[int] = [
            d for d in range(self.num_actions) if d not in bdims and d not in zdims
        ]
        self.num_bernoulli = len(self.bernoulli_dims)
        self.num_force_zero = len(self.force_zero_dims)
        self.num_continuous = len(self.continuous_dims)

        # The backbone produces only `num_continuous + num_bernoulli` action outputs
        # (force-zero dims have no model parameters at all). Within that compressed
        # output, the layout is: [continuous_means | bernoulli_logits].
        self._policy_out_dim = self.num_continuous + self.num_bernoulli
        # Backbone-output indices (where to slice from raw_out).
        self._cont_out_idx = torch.arange(
            0, self.num_continuous, dtype=torch.long, device=device
        )
        self._bern_out_idx = torch.arange(
            self.num_continuous, self._policy_out_dim, dtype=torch.long, device=device
        )
        # Action-vector indices (where to scatter into the env-facing action tensor).
        self._cont_action_idx = torch.as_tensor(self.continuous_dims, dtype=torch.long, device=device)
        self._bern_action_idx = torch.as_tensor(self.bernoulli_dims, dtype=torch.long, device=device)
        self._zero_action_idx = torch.as_tensor(self.force_zero_dims, dtype=torch.long, device=device)

        self.actor_mean = BlockSimBa(
            num_agents=num_agents,
            obs_dim=self.num_observations,
            hidden_dim=actor_latent,
            act_dim=self._policy_out_dim,   # ← shrunk: no params allocated for force-zero dims
            device=device,
            num_blocks=actor_n,
            use_state_dependent_std=use_state_dependent_std,
        ).to(device)

        # log_std parameters cover ONLY continuous dims (Bernoulli has no σ).
        if use_state_dependent_std:
            with torch.no_grad():
                # State-dep std rows live at [act_dim : act_dim + std_out_dim] in the
                # backbone, where act_dim = _policy_out_dim. We restrict consumption
                # to the continuous-only slice (self._cont_out_idx) at runtime.
                self.actor_mean.fc_out.bias[:, self._policy_out_dim:] = math.log(act_init_std)
                self.actor_mean.fc_out.weight[:, self._policy_out_dim:, :] *= 0.1
            self.actor_logstd = None
        else:
            self.actor_logstd = nn.ParameterList(
                [
                    nn.Parameter(torch.ones(1, self.num_continuous) * math.log(act_init_std))
                    for _ in range(num_agents)
                ]
            ).to(device)

        with torch.no_grad():
            # Scale only the action-output rows (first _policy_out_dim).
            self.actor_mean.fc_out.weight[:, : self._policy_out_dim, :] *= last_layer_scale

    def compute(self, inputs, role):
        obs = inputs["observations"]
        num_envs = obs.size(0) // self.num_agents
        raw_out, log_std = self.actor_mean(obs, num_envs)
        # raw_out shape: (N*B, _policy_out_dim) where the layout is
        #   [0 : num_continuous)                          -> continuous Gaussian means
        #   [num_continuous : num_continuous+num_bernoulli) -> Bernoulli logits
        # Force-zero action dims are NOT produced by the model — they're inserted
        # as 0 in the env-facing action vector inside act().

        if not self.use_state_dependent_std:
            batch_size = raw_out.size(0) // self.num_agents
            log_std = torch.cat(
                [p.expand(batch_size, self.num_continuous) for p in self.actor_logstd], dim=0
            )
        elif self.num_continuous < self._policy_out_dim:
            # State-dep std emits one std per backbone-output dim (continuous +
            # bernoulli). Restrict to continuous-only since Bernoulli has no σ.
            log_std = log_std.index_select(-1, self._cont_out_idx)

        outputs = {"log_std": log_std}
        return raw_out, outputs

    def act(self, inputs, *, role: str = ""):
        # Hybrid continuous (squashed Gaussian) + discrete (Bernoulli) sampling.
        # Returns (actions, outputs) per skrl 2.x convention; outputs["log_prob"]
        # is the combined log-prob used by SAC's policy / entropy losses.
        raw_out, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]  # (N*B, num_continuous)

        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, min=self._g_min_log_std, max=self._g_max_log_std)
            outputs["log_std"] = log_std

        taken_actions = inputs.get("taken_actions", None)
        batch = raw_out.shape[0]
        log_prob_parts: list[torch.Tensor] = []
        actions = raw_out.new_zeros((batch, self.num_actions))
        cont_dist = None

        # ---- continuous head (squashed Gaussian on continuous_dims) ----
        # Read from the FIRST num_continuous columns of the (compressed) backbone
        # output; scatter into the env-facing action positions self._cont_action_idx.
        if self.num_continuous > 0:
            cont_mean = raw_out.index_select(-1, self._cont_out_idx)     # (N*B, num_continuous)
            sigma = log_std.exp()
            cont_dist = Normal(cont_mean, sigma)
            self._g_distribution = cont_dist  # for GaussianMixin.get_entropy() compat
            if taken_actions is None:
                u = cont_dist.rsample()
            else:
                # Replay path: recover pre-tanh u from stored continuous actions in (-1, 1).
                taken_cont = taken_actions.index_select(-1, self._cont_action_idx)
                u = safe_atanh(taken_cont)
            a_cont = torch.tanh(u)
            # log p(a_cont) = log p(u) - sum log(1 - tanh^2(u))   (Jacobian correction)
            lp_cont = (
                cont_dist.log_prob(u).sum(dim=-1, keepdim=True)
                - squash_log_prob_correction(u).unsqueeze(-1)
            )
            log_prob_parts.append(lp_cont)
            actions.index_copy_(-1, self._cont_action_idx, a_cont)

        # ---- Bernoulli head (binary on bernoulli_dims, mapped to {-1,+1}) ----
        # Read from the SECOND block of backbone output columns; scatter into the
        # env-facing Bernoulli action positions self._bern_action_idx.
        if self.num_bernoulli > 0:
            bern_logit = raw_out.index_select(-1, self._bern_out_idx)    # (N*B, num_bernoulli)
            bern_prob = torch.sigmoid(bern_logit)
            if taken_actions is None:
                # Fresh sample: draw a Bernoulli sample, route gradient through prob via
                # straight-through estimator. forward = sample, backward = bern_prob.
                with torch.no_grad():
                    bern_sample = (torch.rand_like(bern_prob) < bern_prob).float()
                bern_st = (bern_sample - bern_prob).detach() + bern_prob
                a_bern = 2.0 * bern_st - 1.0                              # {-1, +1} forward
            else:
                # Replay path: stored action is in {-1, +1}; decode to {0, 1} for log_prob.
                # Forward action goes back to env shape; gradient flows through bern_prob
                # via straight-through so the critic Q-grad reaches the policy.
                taken_bern = taken_actions.index_select(-1, self._bern_action_idx)
                bern_sample = ((taken_bern + 1.0) / 2.0).round().clamp(0.0, 1.0)
                bern_st = (bern_sample - bern_prob).detach() + bern_prob
                a_bern = 2.0 * bern_st - 1.0
            bern_dist = torch.distributions.Bernoulli(probs=bern_prob)
            lp_bern = bern_dist.log_prob(bern_sample).sum(dim=-1, keepdim=True)
            log_prob_parts.append(lp_bern)
            actions.index_copy_(-1, self._bern_action_idx, a_bern)

        log_prob = log_prob_parts[0] if len(log_prob_parts) == 1 else sum(log_prob_parts)

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = raw_out
        return actions, outputs

    def get_entropy(self, *, role: str = ""):
        # Continuous-Gaussian entropy as a proxy; the squashed/Bernoulli mixture has
        # no clean closed form and SAC uses log_prob (not entropy) in the gradient.
        if self._g_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._g_distribution.entropy().to(self.device)


# -----------------------------
#  Hybrid force/position actor (selection-gated, two distribution styles)
# -----------------------------
class HybridControlBlockSimBaActor(BlockSimBaActor):
    """Hybrid force/position policy: per-eligible-axis binary **selection** (Bernoulli)
    gates a **position component** vs a **force component**, block-parallel across agents.

    Two joint-distribution styles (``selection_distribution``):

      * ``"product"`` — selection and ALL continuous dims are independent:
        ``log_prob = Σ_continuous logN + Σ_sel logBern``; entropy is the analogous
        independent sum. (This reproduces the plain Bernoulli+Gaussian actor.)
      * ``"match"`` — the continuous density/entropy is **conditioned on the
        selection** (the reference's ``HybridActionGMM``, hard / ``full_CLoP=False``):
        per gated axis only the *selected* component contributes
        (``where(sel>0.5, logp_force, logp_pos)``), and the entropy is the
        selection-probability-weighted mix ``(1-p)·H_pos + p·H_force``. Free
        (non-gated) continuous dims always contribute.

    Adapted from RoboNuke/Continuous_Force_RL's hybrid actor to our **index-based**
    action layout: ``force_zero_action_dims`` carry no params (inserted as 0), and
    selection / position-component / force-component dims are given by explicit
    index lists (typically derived from ``controller_cfg.force_axes`` by the runner).

    Selection actions are emitted in ``{0, 1}`` (the Bernoulli sample via a
    straight-through estimator) so the control wrapper's ``> 0.5`` threshold is
    exact — unlike the gripper Bernoulli head which maps to ``{-1, +1}`` for Isaac
    Lab's ``BinaryJointAction``.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        *,
        selection_dims: list[int],
        pos_component_dims: list[int],
        force_component_dims: list[int],
        selection_distribution: str = "product",
        selection_init_bias: float = 0.0,
        force_zero_action_dims: list[int] | None = None,
        **actor_kwargs,
    ):
        if selection_distribution not in ("product", "match"):
            raise ValueError(
                f"selection_distribution must be 'product' or 'match', got "
                f"{selection_distribution!r}"
            )
        # The selection dims ARE the Bernoulli dims of the base actor; bernoulli must
        # not also be passed via actor_kwargs.
        actor_kwargs.pop("bernoulli_action_dims", None)
        super().__init__(
            observation_space,
            action_space,
            device,
            num_agents=num_agents,
            bernoulli_action_dims=list(selection_dims),
            force_zero_action_dims=force_zero_action_dims,
            **actor_kwargs,
        )
        self.selection_distribution = selection_distribution

        N = len(selection_dims)
        if not (len(pos_component_dims) == len(force_component_dims) == N):
            raise ValueError(
                f"selection_dims ({N}), pos_component_dims ({len(pos_component_dims)}) "
                f"and force_component_dims ({len(force_component_dims)}) must have equal length"
            )
        for d in list(pos_component_dims) + list(force_component_dims):
            if d not in self.continuous_dims:
                raise ValueError(
                    f"gated component dim {d} must be a continuous dim (continuous_dims="
                    f"{self.continuous_dims}); it cannot be a selection or force-zero dim"
                )
        if set(pos_component_dims) & set(force_component_dims):
            raise ValueError(
                "pos_component_dims and force_component_dims must be disjoint; overlap = "
                f"{sorted(set(pos_component_dims) & set(force_component_dims))}"
            )

        # Map action-vector indices -> positions within the compressed continuous /
        # selection backbone outputs (so the gated pairs can be sliced at runtime).
        cont_pos = {d: i for i, d in enumerate(self.continuous_dims)}
        sel_pos = {d: i for i, d in enumerate(self.bernoulli_dims)}
        pair_pos_out = [cont_pos[d] for d in pos_component_dims]
        pair_force_out = [cont_pos[d] for d in force_component_dims]
        pair_sel_out = [sel_pos[d] for d in selection_dims]
        gated = set(pair_pos_out) | set(pair_force_out)
        free_out = [i for i in range(self.num_continuous) if i not in gated]

        # Non-persistent buffers (no leading num_agents dim) — invisible to the
        # block-parallel slicing helpers and to the saved state_dict.
        self.register_buffer(
            "_pair_pos_out", torch.as_tensor(pair_pos_out, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_pair_force_out", torch.as_tensor(pair_force_out, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_pair_sel_out", torch.as_tensor(pair_sel_out, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_free_cont_out", torch.as_tensor(free_out, dtype=torch.long, device=device),
            persistent=False,
        )
        # Per-forward scratch consumed by get_entropy().
        self._last_log_std = None
        self._sel_prob = None
        self._sel_sample = None

        # Bias the selection (Bernoulli) logits at init. last_layer_scale shrinks the
        # logit WEIGHTS toward ~0, so the initial logit is essentially this bias and the
        # initial selection probability is ~= sigmoid(selection_init_bias). A negative
        # default keeps the policy position-dominant at init (force control off until
        # learned); leaving it at 0 gives p=0.5 (coin-flip force/position) which shoves
        # the peg around at init.
        self.selection_init_bias = float(selection_init_bias)
        if self.num_bernoulli > 0:
            with torch.no_grad():
                self.actor_mean.fc_out.bias[:, self.num_continuous : self._policy_out_dim] = (
                    self.selection_init_bias
                )

    def act(self, inputs, *, role: str = ""):
        # Sampling is identical for both styles (the env always needs the full action
        # vector, incl. both position and force components); only the joint log_prob /
        # entropy AGGREGATION differs.
        raw_out, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]  # (N*B, num_continuous)
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, min=self._g_min_log_std, max=self._g_max_log_std)
            outputs["log_std"] = log_std
        self._last_log_std = log_std

        taken_actions = inputs.get("taken_actions", None)
        batch = raw_out.shape[0]
        actions = raw_out.new_zeros((batch, self.num_actions))

        # ---- continuous head: per-dim squashed-Gaussian log_prob (NOT summed) ----
        cont_mean = raw_out.index_select(-1, self._cont_out_idx)     # (N*B, num_continuous)
        sigma = log_std.exp()
        cont_dist = Normal(cont_mean, sigma)
        self._g_distribution = cont_dist  # GaussianMixin / PPO stddev-logging compat
        if taken_actions is None:
            u = cont_dist.rsample()
        else:
            taken_cont = taken_actions.index_select(-1, self._cont_action_idx)
            u = safe_atanh(taken_cont)
        a_cont = torch.tanh(u)
        actions.index_copy_(-1, self._cont_action_idx, a_cont)
        # per-dim Jacobian correction log(1 - tanh^2(u)) (same form as squash_log_prob_correction, unsummed)
        corr = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))      # (N*B, num_continuous)
        lp_cont_per_dim = cont_dist.log_prob(u) - corr               # (N*B, num_continuous)

        # ---- selection head: Bernoulli in {0,1} via straight-through ----
        if self.num_bernoulli > 0:
            sel_logit = raw_out.index_select(-1, self._bern_out_idx)  # (N*B, num_sel)
            sel_prob = torch.sigmoid(sel_logit)
            if taken_actions is None:
                with torch.no_grad():
                    sel_sample = (torch.rand_like(sel_prob) < sel_prob).float()
            else:
                taken_sel = taken_actions.index_select(-1, self._bern_action_idx)
                sel_sample = taken_sel.round().clamp(0.0, 1.0)        # stored as {0,1}
            sel_st = (sel_sample - sel_prob).detach() + sel_prob      # forward={0,1}, grad->prob
            actions.index_copy_(-1, self._bern_action_idx, sel_st)    # emit {0,1}
            lp_sel_per_dim = torch.distributions.Bernoulli(probs=sel_prob).log_prob(sel_sample)
            self._sel_prob = sel_prob
            self._sel_sample = sel_sample
        else:
            lp_sel_per_dim = raw_out.new_zeros((batch, 0))
            self._sel_prob = None
            self._sel_sample = None

        # ---- aggregate joint log_prob per style ----
        if self.selection_distribution == "product":
            log_prob = (
                lp_cont_per_dim.sum(dim=-1, keepdim=True)
                + lp_sel_per_dim.sum(dim=-1, keepdim=True)
            )
        else:  # match: condition the continuous density on the (sampled) selection
            free = lp_cont_per_dim.index_select(-1, self._free_cont_out).sum(dim=-1, keepdim=True)
            if self._pair_sel_out.numel() > 0:
                sel_pairs = self._sel_sample.index_select(-1, self._pair_sel_out)   # (N*B, n_pairs)
                lp_pos = lp_cont_per_dim.index_select(-1, self._pair_pos_out)
                lp_force = lp_cont_per_dim.index_select(-1, self._pair_force_out)
                gated = torch.where(sel_pairs > 0.5, lp_force, lp_pos).sum(dim=-1, keepdim=True)
            else:
                gated = free.new_zeros((batch, 1))
            log_prob = free + gated + lp_sel_per_dim.sum(dim=-1, keepdim=True)

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = raw_out
        return actions, outputs

    def get_entropy(self, *, role: str = ""):
        # Closed-form (proxy) entropy following the active style. Continuous uses the
        # unsquashed-Gaussian differential entropy 0.5*log(2*pi*e) + log_std (the same
        # proxy the plain actor uses); selection uses the Bernoulli entropy.
        if self._last_log_std is None:
            return torch.tensor(0.0, device=self.device)
        log_std = self._last_log_std
        H_cont = 0.5 * math.log(2.0 * math.pi * math.e) + log_std    # (N*B, num_continuous)
        if self.num_bernoulli > 0 and self._sel_prob is not None:
            p = self._sel_prob
            H_sel = -(p * torch.log(p + 1e-8) + (1.0 - p) * torch.log(1.0 - p + 1e-8))
        else:
            H_sel = H_cont.new_zeros((H_cont.shape[0], 0))

        if self.selection_distribution == "product":
            H = H_cont.sum(dim=-1, keepdim=True) + H_sel.sum(dim=-1, keepdim=True)
        else:  # match: H(S) + E_S[H(cont|S)]  (p-weighted mix of the two components)
            free = H_cont.index_select(-1, self._free_cont_out).sum(dim=-1, keepdim=True)
            if self._pair_sel_out.numel() > 0:
                p_pairs = self._sel_prob.index_select(-1, self._pair_sel_out)       # (N*B, n_pairs)
                H_pos = H_cont.index_select(-1, self._pair_pos_out)
                H_force = H_cont.index_select(-1, self._pair_force_out)
                gated = ((1.0 - p_pairs) * H_pos + p_pairs * H_force).sum(dim=-1, keepdim=True)
            else:
                gated = free.new_zeros((free.shape[0], 1))
            H = free + gated + H_sel.sum(dim=-1, keepdim=True)
        return H


# -----------------------------
#  Q-critic (state, action -> scalar)
# -----------------------------
class BlockSimBaQCritic(DeterministicMixin, Model):
    """SAC Q-function: concatenates observation and action, returns scalar Q per (o, a).

    skrl SAC calls this via `critic.act({**inputs, "taken_actions": actions})`, where
    `inputs["observations"]` carries the observation and `inputs["taken_actions"]` the action.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        critic_output_init_mean: float = 0.0,
        critic_n: int = 2,
        critic_latent: int = 512,
        clip_actions: bool = False,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.num_agents = num_agents
        self.q_net = BlockSimBa(
            num_agents=num_agents,
            obs_dim=self.num_observations + self.num_actions,
            hidden_dim=critic_latent,
            act_dim=1,
            device=device,
            num_blocks=critic_n,
            use_state_dependent_std=False,
        ).to(device)

        torch.nn.init.constant_(self.q_net.fc_out.bias, critic_output_init_mean)

    def compute(self, inputs, role):
        obs = inputs["observations"]
        actions = inputs["taken_actions"]
        x = torch.cat([obs, actions], dim=-1)
        num_envs = x.size(0) // self.num_agents
        value, _ = self.q_net(x, num_envs)  # backbone returns (out, log_std)
        return value, {}


# -----------------------------
#  State-value critic (state -> scalar) — PPO
# -----------------------------
class BlockSimBaValueCritic(DeterministicMixin, Model):
    """PPO state-value function V(obs) -> scalar, block-parallel across agents.

    Mirrors :class:`BlockSimBaQCritic` but consumes observations ONLY (no action
    concatenation), since PPO's critic estimates V(s) rather than Q(s, a). Reuses
    the same ``BlockSimBa`` backbone and accepts the same ``model_cfg.critic``
    kwargs, so the per-agent save/load slicing helpers apply unchanged.

    skrl PPO calls this via ``value.act({"observations": obs}, role="value")``.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        critic_output_init_mean: float = 0.0,
        critic_n: int = 2,
        critic_latent: int = 512,
        clip_actions: bool = False,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.num_agents = num_agents
        self.v_net = BlockSimBa(
            num_agents=num_agents,
            obs_dim=self.num_observations,   # obs only — no +num_actions
            hidden_dim=critic_latent,
            act_dim=1,
            device=device,
            num_blocks=critic_n,
            use_state_dependent_std=False,
        ).to(device)

        torch.nn.init.constant_(self.v_net.fc_out.bias, critic_output_init_mean)

    def compute(self, inputs, role):
        obs = inputs["observations"]
        num_envs = obs.size(0) // self.num_agents
        value, _ = self.v_net(obs, num_envs)  # backbone returns (out, log_std)
        return value, {}


# -----------------------------
#  Per-agent save/load slicing helpers
# -----------------------------
def _is_block_tensor(t, num_agents: int) -> bool:
    """A tensor is "block-parallel" if its leading dim equals num_agents."""
    return torch.is_tensor(t) and t.dim() >= 1 and t.shape[0] == num_agents


def _per_agent_paramlist_prefixes(block_module: nn.Module, num_agents: int) -> list[str]:
    """Find dotted prefixes of nn.ParameterList children that have ``num_agents`` entries.

    Used by the state_dict slicer to recognize per-agent ParameterList keys (e.g.
    ``actor_logstd.0``, ``actor_logstd.1`` for N=2) so they can be filtered + renumbered
    when slicing for a single agent.
    """
    prefixes = []
    for mod_name, mod in block_module.named_modules():
        if isinstance(mod, nn.ParameterList) and len(mod) == num_agents:
            prefixes.append(mod_name)  # e.g. "actor_logstd"
    return prefixes


def slice_block_state_dict(block_module: nn.Module, agent_idx: int, num_agents: int) -> dict:
    """Return a state_dict with every dim-0 == num_agents tensor sliced to agent_idx.

    Output tensors have one less leading dim than the source (e.g. ``(N, out, in)`` ->
    ``(out, in)``). Per-agent ``nn.ParameterList`` entries (length == num_agents) keep
    only the ``agent_idx``-th entry, renumbered to ``0`` so the result is shaped like a
    single-agent module's state_dict. Non-block tensors are passed through unchanged.
    """
    pl_prefixes = _per_agent_paramlist_prefixes(block_module, num_agents)
    sliced = {}
    for name, param in block_module.state_dict().items():
        # Handle per-agent ParameterList entries: keep only agent_idx, renumber to 0.
        matched_prefix = None
        for pre in pl_prefixes:
            if name.startswith(pre + "."):
                matched_prefix = pre
                break
        if matched_prefix is not None:
            tail = name[len(matched_prefix) + 1 :]
            head, _, rest = tail.partition(".")
            if not head.isdigit():
                raise ValueError(
                    f"Expected integer index after ParameterList prefix '{matched_prefix}.' "
                    f"in state_dict key '{name}', got '{head}'"
                )
            if int(head) != agent_idx:
                continue  # drop other agents' entries
            new_name = f"{matched_prefix}.0" + (("." + rest) if rest else "")
            sliced[new_name] = param.detach().clone().cpu() if torch.is_tensor(param) else param
            continue

        if _is_block_tensor(param, num_agents):
            sliced[name] = param[agent_idx].detach().clone().cpu()
        else:
            sliced[name] = param.detach().clone().cpu() if torch.is_tensor(param) else param
    return sliced


def assign_block_slice(
    block_module: nn.Module, agent_idx: int, num_agents: int, agent_state_dict: dict
) -> None:
    """Write ``agent_state_dict`` (single-agent shape) into block_module's slot ``agent_idx``.

    Inverts what :func:`slice_block_state_dict` did:
      * For each block-parallel param in block_module, copies the source tensor into
        ``param.data[agent_idx]``.
      * For per-agent ParameterList entries renamed ``prefix.0`` on save, writes them
        back to ``prefix.{agent_idx}`` of the destination.
      * Non-block params are copied wholesale.
    """
    pl_prefixes = _per_agent_paramlist_prefixes(block_module, num_agents)
    block_state = block_module.state_dict()

    # Translate "saved key" -> "destination block key" via the renumber-back step.
    remapped = {}
    for name, val in agent_state_dict.items():
        matched_prefix = None
        for pre in pl_prefixes:
            if name.startswith(pre + "."):
                matched_prefix = pre
                break
        if matched_prefix is not None:
            tail = name[len(matched_prefix) + 1 :]
            head, _, rest = tail.partition(".")
            if head == "0":
                new_name = f"{matched_prefix}.{agent_idx}" + (("." + rest) if rest else "")
                remapped[new_name] = val
                continue
        remapped[name] = val

    # Source must be a subset of dest's keys (other agents' ParameterList entries are
    # legitimately absent from a single-agent slice).
    extra = set(remapped.keys()) - set(block_state.keys())
    if extra:
        raise KeyError(f"Unexpected keys in single-agent state_dict: {sorted(extra)}")

    paramlist_keys = {k for k in block_state.keys()
                      if any(k.startswith(p + ".") for p in pl_prefixes)}

    with torch.no_grad():
        for name, agent_param in remapped.items():
            block_param = block_state[name]
            # ParameterList entries are written wholesale into their dedicated slot
            # (they don't have a leading num_agents dim of their own).
            if name in paramlist_keys:
                if torch.is_tensor(block_param):
                    block_param.copy_(agent_param.to(block_param.device))
                continue
            if _is_block_tensor(block_param, num_agents):
                block_param[agent_idx].copy_(agent_param.to(block_param.device))
            else:
                if torch.is_tensor(block_param):
                    block_param.copy_(agent_param.to(block_param.device))


def slice_optimizer_state(
    opt_state_dict: dict, agent_idx: int, num_agents: int
) -> dict:
    """Slice every dim-0 == num_agents tensor in an optimizer state_dict to agent_idx.

    Non-block tensors (e.g. Adam's scalar ``step``, or ``actor_logstd[i]`` whose
    own leading dim is 1) and non-tensor entries are passed through unchanged.
    ``param_groups`` is preserved verbatim. A sidecar ``_sliced_keys`` set records
    which (param_id, key) pairs were sliced so :func:`merge_optimizer_states` can
    unambiguously restack them later.
    """
    if "state" not in opt_state_dict:
        raise KeyError("Optimizer state_dict missing required key 'state'")
    if "param_groups" not in opt_state_dict:
        raise KeyError("Optimizer state_dict missing required key 'param_groups'")

    out_state = {}
    sliced_keys = set()
    for param_id, param_state in opt_state_dict["state"].items():
        new_state = {}
        for k, v in param_state.items():
            if _is_block_tensor(v, num_agents):
                new_state[k] = v[agent_idx].detach().clone().cpu()
                sliced_keys.add((param_id, k))
            elif torch.is_tensor(v):
                new_state[k] = v.detach().clone().cpu()
            else:
                new_state[k] = v
        out_state[param_id] = new_state
    return {
        "state": out_state,
        "param_groups": opt_state_dict["param_groups"],
        "_sliced_keys": sliced_keys,
    }


def merge_optimizer_states(per_agent_state_dicts: list, num_agents: int) -> dict:
    """Stack per-agent optimizer state_dicts back into a block-shaped state_dict.

    Uses each per-agent dict's ``_sliced_keys`` sidecar (written by
    :func:`slice_optimizer_state`) to know exactly which (param_id, key) pairs were
    sliced on save and therefore must be re-stacked. All other tensor entries are
    taken from agent 0 verbatim.
    """
    if len(per_agent_state_dicts) != num_agents:
        raise ValueError(
            f"Expected {num_agents} per-agent state_dicts, got {len(per_agent_state_dicts)}"
        )
    a0 = per_agent_state_dicts[0]
    if "_sliced_keys" not in a0:
        raise KeyError(
            "Per-agent optimizer state_dict missing '_sliced_keys' sidecar; this is "
            "required to know which entries were block-parallel and must be re-stacked. "
            "Did you produce the slice with slice_optimizer_state()?"
        )
    if "state" not in a0 or "param_groups" not in a0:
        raise KeyError("Per-agent optimizer state_dict missing 'state' or 'param_groups'")

    sliced_keys = a0["_sliced_keys"]
    out_state = {}
    for param_id, param_state in a0["state"].items():
        new_state = {}
        for k, v0 in param_state.items():
            if (param_id, k) in sliced_keys:
                new_state[k] = torch.stack(
                    [per_agent_state_dicts[i]["state"][param_id][k] for i in range(num_agents)],
                    dim=0,
                )
            else:
                new_state[k] = v0
        out_state[param_id] = new_state
    return {"state": out_state, "param_groups": a0["param_groups"]}
