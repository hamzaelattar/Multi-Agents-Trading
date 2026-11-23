# COMMON CORE — copied into each standalone training script.
import argparse
import copy
import json
import math
import random
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _canon(key):
    return re.sub(r'[^a-z0-9]', '', str(key).lower())


def _col(df, *names):
    names_by_canonical = {_canon(k): k for k in df.columns}
    for name in names:
        if _canon(name) in names_by_canonical:
            return pd.to_numeric(df[names_by_canonical[_canon(name)]], errors='coerce')
    return None


class Market:
    """Causal features; train-only standardization; next-open execution with costs."""
    def __init__(self, csv_path, lookback=32, spread_bps=2.0, fee_bps=0.0,
                 train_fraction=0.70, val_fraction=0.15, max_rows=0):
        df = pd.read_csv(csv_path)
        if len(df) < max(120, 3 * lookback + 30):
            raise ValueError('CSV too short for train/validation/test and lookback.')
        time_col = next((c for c in df.columns if _canon(c) in
                         ('time', 'timestamp', 'datetime', 'date')), None)
        self.times = None
        if time_col:
            df['__parsed_time__'] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
            if df['__parsed_time__'].isna().any():
                raise ValueError('Some timestamps are invalid; fix the CSV first.')
            df = df.sort_values('__parsed_time__').drop_duplicates('__parsed_time__')
            self.times = df['__parsed_time__'].reset_index(drop=True)
        if max_rows > 0 and len(df) > max_rows:
            df = df.iloc[-max_rows:].copy()
            if self.times is not None:
                self.times = df['__parsed_time__'].reset_index(drop=True)
            print(f'INFO: using ONLY last {len(df)} rows because --max-rows was set.')
        df = df.reset_index(drop=True)

        price = {}
        for kind in ('open', 'high', 'low', 'close'):
            standard = _col(df, kind)
            bid = _col(df, 'bid_' + kind, kind + '_bid')
            ask = _col(df, 'ask_' + kind, kind + '_ask')
            if standard is not None:
                series = standard
            elif bid is not None and ask is not None:
                series = (bid + ask) / 2.0
            elif bid is not None:
                series = bid
                print(f'INFO: {kind} uses BID quote only; bid/ask execution is approximated.')
            else:
                raise ValueError(f'Missing {kind} column (or bid_{kind}/ask_{kind}).')
            price[kind] = series.to_numpy(dtype=np.float64)
        self.open, self.high = price['open'], price['high']
        self.low, self.close = price['low'], price['close']
        if (not np.isfinite(np.stack(list(price.values()))).all() or
                (np.stack(list(price.values())) <= 0).any()):
            raise ValueError('Prices contain null, non-finite, or non-positive values.')
        if (self.high < self.low).any():
            raise ValueError('high must be >= low.')
        self.n = len(df)
        self.lookback = lookback
        self.fee_fraction = fee_bps / 10000.0

        bid_open = _col(df, 'bid_open', 'open_bid')
        ask_open = _col(df, 'ask_open', 'open_ask')
        spread = _col(df, 'spread')
        if bid_open is not None and ask_open is not None:
            self.spread_frac = ((ask_open - bid_open).to_numpy(dtype=np.float64) /
                                np.maximum(self.open, 1e-12))
        elif spread is not None:
            # Explicit convention: a spread column is an ABSOLUTE price difference.
            self.spread_frac = spread.to_numpy(dtype=np.float64) / self.open
        else:
            self.spread_frac = np.full(self.n, spread_bps / 10000.0)
            print(f'INFO: no bid/ask or spread found; assuming {spread_bps} bps full spread.')
        if not np.isfinite(self.spread_frac).all() or (self.spread_frac < 0).any():
            raise ValueError('Invalid spread; expected nonnegative bid/ask difference.')

        volume = _col(df, 'volume', 'tick_volume', 'tickvolume')
        if volume is None:
            volume = pd.Series(np.zeros(self.n, dtype=np.float64))
            print('INFO: volume missing; volume-derived feature is set to zero.')
        v = volume.fillna(0).clip(lower=0).to_numpy(dtype=np.float64)
        lc = np.log(self.close)
        ret = np.r_[0.0, np.diff(lc)]
        ema20 = pd.Series(self.close).ewm(span=20, adjust=False).mean().to_numpy()
        rv = pd.Series(ret).rolling(20, min_periods=2).std().fillna(0).to_numpy()
        lv = np.log1p(v)
        vdiff = np.r_[0.0, np.diff(lv)]
        raw = np.column_stack([
            ret, np.log(self.close / self.open), np.log(self.high / self.low),
            (self.high - np.maximum(self.open, self.close)) / self.close,
            (np.minimum(self.open, self.close) - self.low) / self.close,
            np.log(self.close / ema20), rv, vdiff, self.spread_frac,
        ])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and
                train_fraction + val_fraction < 1):
            raise ValueError('Invalid train and validation fractions.')
        self.train_end = int(self.n * train_fraction)
        self.val_end = int(self.n * (train_fraction + val_fraction))
        if min(self.train_end - lookback, self.val_end - self.train_end,
               self.n - self.val_end) < 20:
            raise ValueError('Train, validation and test each need enough bars.')
        mu = raw[:self.train_end].mean(axis=0)
        std = raw[:self.train_end].std(axis=0)
        std[std < 1e-9] = 1.0
        self.mean, self.std = mu.astype(np.float32), std.astype(np.float32)
        self.features = np.clip((raw - mu) / std, -8, 8).astype(np.float32)
        self.feature_count = self.features.shape[1]
        # Target for a decision made after bar t: return inside NEXT bar (open[t+1]->close[t+1]).
        self.target_bps = np.zeros(self.n, dtype=np.float32)
        self.target_bps[:-1] = (10000.0 * np.log(self.close[1:] / self.open[1:])).astype(np.float32)
        self.bars_per_year = 252 * 288
        if self.times is not None and len(self.times) > 2:
            dt = self.times.diff().dt.total_seconds().dropna()
            median = dt[dt > 0].median()
            if np.isfinite(median) and median > 0:
                # Forex-style 252 trading days x 24 hours, adjusted for detected bar size.
                self.bars_per_year = max(1, int(252 * 24 * 3600 / median))

    def window(self, t):
        return self.features[t - self.lookback + 1:t + 1]

    def obs(self, t, pos, equity):
        return np.concatenate((self.window(t).reshape(-1),
                               np.asarray([pos, np.log(max(equity, 1e-12))], dtype=np.float32)))

    def transition(self, t, old_pos, target_pos, equity):
        """Action at bar t close; rebalance at t+1 open, mark at t+1 close.

        Fractional position/linear exposure approximation, NOT a broker margin simulator.
        """
        j = t + 1
        target_pos = float(np.clip(target_pos, -1.0, 1.0))
        gap_return = self.open[j] / self.close[t] - 1.0
        bar_return = self.close[j] / self.open[j] - 1.0
        turnover = abs(target_pos - old_pos)
        cost = turnover * (self.fee_fraction + 0.5 * self.spread_frac[j])
        simple = old_pos * gap_return + target_pos * bar_return - cost
        ratio = max(1e-8, 1.0 + simple)
        new_equity = float(equity * ratio)
        log_reward = float(np.log(ratio))
        return new_equity, log_reward, turnover

    def partitions(self):
        return {'train': (self.lookback - 1, self.train_end - 1),
                'validation': (self.train_end - 1, self.val_end - 1),
                'test': (self.val_end - 1, self.n - 1)}


def metrics(equity_curve, log_rewards, turnover, bars_per_year):
    equity_curve = np.asarray(equity_curve, dtype=float)
    lr = np.asarray(log_rewards, dtype=float)
    period_returns = np.expm1(lr)
    if len(period_returns) > 1 and period_returns.std() > 1e-12:
        sharpe = float(np.sqrt(bars_per_year) * period_returns.mean() /
                       period_returns.std(ddof=1))
    else:
        sharpe = 0.0
    peaks = np.maximum.accumulate(equity_curve)
    max_drawdown = float(np.min(equity_curve / np.maximum(peaks, 1e-12) - 1.0))
    return {'total_return': float(equity_curve[-1] / equity_curve[0] - 1.0),
            'sharpe_annualized': sharpe, 'max_drawdown': max_drawdown,
            'turnover': float(turnover), 'bars': int(len(lr)),
            'ending_equity': float(equity_curve[-1])}


@torch.no_grad()
def backtest(market, policy, start, end, bars_per_year=None):
    """policy(observation, t, position, equity) -> scalar target exposure."""
    pos, equity = 0.0, 1.0
    curve, rewards, turnover = [equity], [], 0.0
    for t in range(start, end):
        obs = market.obs(t, pos, equity)
        action = float(policy(obs, t, pos, equity))
        equity, lr, tv = market.transition(t, pos, action, equity)
        pos = float(np.clip(action, -1, 1))
        curve.append(equity)
        rewards.append(lr)
        turnover += tv
    return metrics(curve, rewards, turnover,
                   bars_per_year or market.bars_per_year), curve


class Replay:
    def __init__(self, capacity, obs_dim, action_dim):
        self.cap, self.i, self.count = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros((capacity, action_dim), np.float32)
        self.r = np.zeros((capacity, 1), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)

    def push(self, s, a, r, s2, done):
        i = self.i
        self.s[i], self.a[i], self.r[i, 0] = s, a, r
        self.s2[i], self.done[i, 0] = s2, float(done)
        self.i = (i + 1) % self.cap
        self.count = min(self.count + 1, self.cap)

    def sample(self, batch, device):
        idx = np.random.randint(0, self.count, size=batch)
        return tuple(torch.as_tensor(arr[idx], device=device) for arr in
                     (self.s, self.a, self.r, self.s2, self.done))


def mlp(inputs, outputs, hidden=128, depth=2):
    layers = []
    dim = inputs
    for _ in range(depth):
        layers.extend([nn.Linear(dim, hidden), nn.ReLU()])
        dim = hidden
    layers.append(nn.Linear(dim, outputs))
    return nn.Sequential(*layers)


def write_results(out, config, results):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'metrics.json').open('w', encoding='utf8') as f:
        json.dump({'configuration': vars(config), 'results': results}, f, indent=2, default=str)
    print('METRICS:', json.dumps(results, indent=2))
    print('Output directory:', out.resolve())


def common_parser(description, default_lookback):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--csv', required=True, help='Chronologically ordered XAUUSD OHLC CSV.')
    p.add_argument('--out', default='runs/trading')
    p.add_argument('--lookback', type=int, default=default_lookback)
    p.add_argument('--max-rows', type=int, default=0, help='0=all rows; otherwise use last N rows.')
    p.add_argument('--spread-bps', type=float, default=2.0, help='Full spread if absent from CSV.')
    p.add_argument('--fee-bps', type=float, default=0.0, help='Additional per-side commission in bps.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--bars-per-year', type=int, default=0, help='0=derive from timestamps, forex convention.')
    return p


def build_market(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    m = Market(args.csv, args.lookback, args.spread_bps, args.fee_bps,
               max_rows=args.max_rows)
    if args.bars_per_year > 0:
        m.bars_per_year = args.bars_per_year
    print(f'rows={m.n:,}; features={m.feature_count}; lookback={m.lookback}; '
          f'splits={m.partitions()}; device={device}')
    return m, device


def sample_windows(market, low, high, batch, device):
    """Indexes uniformly in [low,high) with next-bar targets; no future beyond high."""
    indices = np.random.randint(low, high, size=batch)
    xx = np.stack([market.window(int(t)) for t in indices])
    yy = market.target_bps[indices]
    return indices, torch.from_numpy(xx).to(device), torch.from_numpy(yy).to(device).unsqueeze(-1)

# ===== Approach 1: StockFormer-inspired Predictive Coding + SAC ===============
# Research basis: NEUR1, Gao et al., IJCAI 2023. This is an ADAPTATION:
# one XAUUSD instrument, so the paper's inter-asset relational branch becomes
# a cross-feature relationship branch. Uses normal Transformer attention rather
# than the paper's diversified multi-head FFN (DMH-Attn). Not an exact reproduction.

class SFBackbone(nn.Module):
    def __init__(self, n_features, lookback, d_model=32, heads=4, layers=1, output_dim=64):
        super().__init__()
        self.n_features, self.lookback = n_features, lookback
        self.input_embed = nn.Linear(n_features, d_model)
        self.time_pos = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
        self.short_pos = nn.Parameter(torch.randn(1, min(8, lookback), d_model) * 0.02)
        self.rel_input = nn.Linear(lookback, d_model)
        self.feat_pos = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
        def transformer():
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=heads,
                                           dim_feedforward=4*d_model, dropout=0.0,
                                           batch_first=True, activation='gelu'),
                num_layers=layers, enable_nested_tensor=False)
        self.long_branch = transformer()
        self.short_branch = transformer()
        self.relation_branch = transformer()
        self.portfolio_proj = nn.Linear(2, d_model)
        self.fusion_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d_model, output_dim), nn.LayerNorm(output_dim), nn.Tanh())

    def forward_branches(self, seq, portfolio):
        # seq = (batch, lookback, features); all observations are <= bar t.
        sl = self.long_branch(self.input_embed(seq) + self.time_pos).mean(dim=1)
        n = min(8, self.lookback)
        ss = self.short_branch(self.input_embed(seq[:, -n:]) + self.short_pos).mean(dim=1)
        # Cross-feature relations replace cross-asset relations for XAUUSD.
        sr = self.relation_branch(self.rel_input(seq.transpose(1, 2)) + self.feat_pos).mean(dim=1)
        tokens = torch.stack((sl, ss, sr, self.portfolio_proj(portfolio)), dim=1)
        fused, _ = self.fusion_attn(tokens, tokens, tokens, need_weights=False)
        return self.out(fused.mean(dim=1)), sl, ss, sr

    def forward(self, observation):
        batch = observation.shape[0]
        seq = observation[:, :-2].reshape(batch, self.lookback, self.n_features)
        return self.forward_branches(seq, observation[:, -2:])[0]


class PretrainHeads(nn.Module):
    def __init__(self, backbone, d_model, n_features):
        super().__init__()
        self.backbone = backbone
        self.pred_short = nn.Linear(d_model, 1)
        self.pred_long = nn.Linear(d_model, 1)
        self.reconstruct_features = nn.Linear(d_model, n_features)

    def forward(self, seq, mask):
        masked = seq.masked_fill(mask[:, None, :], 0.0)
        p = torch.zeros((seq.shape[0], 2), device=seq.device)
        _, sl, ss, sr = self.backbone.forward_branches(masked, p)
        return self.pred_short(ss), self.pred_long(sl), self.reconstruct_features(sr)


class SquashedGaussianActor(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = mlp(latent_dim, 2, hidden=128)

    def sample(self, latents, deterministic=False):
        x = self.net(latents)
        mean, logstd = x[:, :1], torch.clamp(x[:, 1:], -5, 1)
        std = logstd.exp()
        z = mean if deterministic else mean + std * torch.randn_like(mean)
        a = torch.tanh(z)
        # Log probability corrected for Tanh squashing.
        log_prob = (-0.5 * (((z - mean) / std) ** 2 + 2 * logstd +
                            math.log(2 * math.pi)) - torch.log(1 - a.square() + 1e-6)).sum(1, keepdim=True)
        return a, log_prob


class StockFormerSAC:
    def __init__(self, backbone, device, lr=3e-4, gamma=0.99, tau=0.005):
        self.device, self.gamma, self.tau = device, gamma, tau
        self.encoder = backbone.to(device)
        # As in the paper's training philosophy: policy-learning gradients
        # alter relational representations; temporal predictive branches frozen.
        for module in (self.encoder.long_branch, self.encoder.short_branch,
                       self.encoder.input_embed):
            for p in module.parameters():
                p.requires_grad_(False)
        self.encoder.time_pos.requires_grad_(False)
        self.encoder.short_pos.requires_grad_(False)
        self.actor = SquashedGaussianActor().to(device)
        self.q1 = mlp(65, 1, 128).to(device)
        self.q2 = mlp(65, 1, 128).to(device)
        self.encoder_target = copy.deepcopy(self.encoder).to(device).eval()
        self.q1_target = copy.deepcopy(self.q1).to(device).eval()
        self.q2_target = copy.deepcopy(self.q2).to(device).eval()
        self.optim_q = torch.optim.Adam(
            [p for p in self.encoder.parameters() if p.requires_grad] +
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.log_alpha = torch.nn.Parameter(torch.tensor(-1.6, device=device))
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        x = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        self.encoder.eval()
        a, _ = self.actor.sample(self.encoder(x), deterministic)
        return float(a.item())

    def update(self, replay, batch):
        s, a, r, s2, done = replay.sample(batch, self.device)
        self.encoder.train()
        with torch.no_grad():
            z2 = self.encoder_target(s2)
            a2, lp2 = self.actor.sample(z2)
            q2 = torch.minimum(self.q1_target(torch.cat([z2, a2], dim=1)),
                               self.q2_target(torch.cat([z2, a2], dim=1)))
            target = r + (1.0 - done) * self.gamma * (q2 - self.log_alpha.exp() * lp2)
        z = self.encoder(s)
        q1, q2 = self.q1(torch.cat([z, a], dim=1)), self.q2(torch.cat([z, a], dim=1))
        qloss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.optim_q.zero_grad(set_to_none=True)
        qloss.backward()
        nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.q1.parameters()) +
                                 list(self.q2.parameters()), 1.0)
        self.optim_q.step()
        z = self.encoder(s).detach()
        an, lp = self.actor.sample(z)
        qn = torch.minimum(self.q1(torch.cat([z, an], dim=1)),
                           self.q2(torch.cat([z, an], dim=1)))
        actor_loss = (self.log_alpha.exp().detach() * lp - qn).mean()
        self.optim_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim_actor.step()
        alpha_loss = -(self.log_alpha * (lp.detach() + (-1.0))).mean()
        self.optim_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.optim_alpha.step()
        with torch.no_grad():
            for source, target_model in ((self.encoder, self.encoder_target),
                                         (self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, tp in zip(source.parameters(), target_model.parameters()):
                    tp.mul_(1 - self.tau).add_(self.tau * p)
        return float(qloss.item()), float(actor_loss.item())


def pretrain(m, args, device, encoder):
    net = PretrainHeads(encoder, args.d_model, m.feature_count).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.pretrain_lr)
    low = m.lookback - 1
    horizon = max(2, args.long_horizon)
    high = m.train_end - horizon
    if high <= low:
        raise ValueError('Not enough training rows for long horizon.')
    for update in range(args.pretrain_updates):
        indices, x, short_target = sample_windows(m, low, high, args.batch_size, device)
        future_ix = indices + horizon
        long_bps = 10000.0 * np.log(m.close[future_ix] / m.close[indices])
        long_target = torch.tensor(long_bps, dtype=torch.float32, device=device).view(-1, 1)
        mask = torch.rand((args.batch_size, m.feature_count), device=device) < 0.25
        pred_s, pred_l, reconstruction = net(x, mask)
        recon_err = (reconstruction - x[:, -1, :]).square()
        recon_loss = recon_err[mask].mean() if mask.any() else recon_err.mean()
        # price-return losses use bps; long horizon naturally has larger variance.
        loss = F.smooth_l1_loss(pred_s, short_target) + 0.25 * F.smooth_l1_loss(
            pred_l, long_target) + 0.1 * recon_loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        if (update + 1) % max(1, args.pretrain_updates // 5) == 0:
            print(f'pretrain {update + 1}/{args.pretrain_updates}: loss={loss.item():.5f}')
    return encoder


def train_sac(m, args, device, agent):
    start_train, end_train = m.partitions()['train']
    dim = m.lookback * m.feature_count + 2
    replay = Replay(args.buffer_size, dim, 1)
    pos, equity, t = 0.0, 1.0, end_train
    for step in range(args.steps):
        if step == 0 or t >= ep_end:
            ep_length = min(args.episode_bars, end_train - start_train)
            t = int(np.random.randint(start_train, end_train - ep_length + 1))
            ep_end = t + ep_length
            pos, equity = 0.0, 1.0
        obs = m.obs(t, pos, equity)
        action = float(np.random.uniform(-1, 1)) if step < args.warmup else agent.act(obs)
        equity2, log_r, _ = m.transition(t, pos, action, equity)
        next_obs = m.obs(t + 1, action, equity2)
        done = (t + 1 >= ep_end)
        replay.push(obs, [action], log_r * args.reward_scale, next_obs, done)
        t, pos, equity = t + 1, action, equity2
        if step >= args.warmup and replay.count >= args.batch_size:
            for _ in range(args.updates_per_step):
                qloss, aloss = agent.update(replay, args.batch_size)
        if (step + 1) % max(1, args.steps // 10) == 0:
            print(f'SAC {step + 1}/{args.steps}: replay={replay.count} '
                  f'last_reward={log_r:.6f}' +
                  (f' qloss={qloss:.4f} aloss={aloss:.4f}' if step >= args.warmup and replay.count >= args.batch_size else ''))


def main():
    parser = common_parser('StockFormer-inspired three-branch Transformer + SAC for XAUUSD', 32)
    parser.add_argument('--d-model', type=int, default=32)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--long-horizon', type=int, default=12)
    parser.add_argument('--pretrain-updates', type=int, default=500)
    parser.add_argument('--pretrain-lr', type=float, default=3e-4)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--episode-bars', type=int, default=1024)
    parser.add_argument('--buffer-size', type=int, default=30000)
    parser.add_argument('--updates-per-step', type=int, default=1)
    parser.add_argument('--reward-scale', type=float, default=1000.0)
    args = parser.parse_args()
    if args.d_model % args.heads:
        parser.error('--d-model must be divisible by --heads')
    m, device = build_market(args)
    backbone = SFBackbone(m.feature_count, m.lookback, args.d_model,
                          args.heads, args.layers).to(device)
    pretrain(m, args, device, backbone)
    agent = StockFormerSAC(backbone, device)
    train_sac(m, args, device, agent)
    results = {}
    for segment in ('validation', 'test'):
        start, end = m.partitions()[segment]
        stat, _ = backtest(m, lambda s, *_: agent.act(s, deterministic=True), start, end)
        results[segment] = stat
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'encoder': agent.encoder.state_dict(), 'actor': agent.actor.state_dict(),
                'q1': agent.q1.state_dict(), 'q2': agent.q2.state_dict(),
                'scaler_mean': m.mean, 'scaler_std': m.std, 'feature_count': m.feature_count,
                'lookback': m.lookback, 'd_model': args.d_model, 'heads': args.heads,
                'layers': args.layers}, out / 'stockformer_sac.pt')
    write_results(out, args, results)


if __name__ == '__main__':
    main()
