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
"""
Fixe les graines aléatoires de Python, NumPy et PyTorch pour rendre
les expériences reproductibles. La même seed permet de retrouver
les mêmes séquences aléatoires dans les mêmes conditions.
Cela ne bloque pas l'apprentissage des neurones.
"""

def _canon(key):
    return re.sub(r'[^a-z0-9]', '', str(key).lower())

"""
Standardise les noms des colonnes en les convertissant en minuscules
et en supprimant les espaces, symboles et caractères spéciaux.
Exemple : 'Bid_Open' devient 'bidopen'.
"""

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

# ===== Approach 2: DRL-DEM-inspired 6 predictors + actor-critic =============
# Research basis: NEUR2, Lin/Xie/Xu, Electronics 2023.
# Faithful ingredients: 6 GRU/ALSTM/Transformer predictors (64,128 each),
# MSE vs SmoothL1 objectives, actor with THREE 3-linear-layer MLP branches,
# softmax predictor weights, actor-critic replay, supervised+trading fine-tuning.
# XAUUSD adaptation: predicted future return is mapped to a bounded exposure;
# original paper used a top-q stock portfolio, which requires multiple stocks.

class Predictor(nn.Module):
    def __init__(self, family, features, hidden, lookback):
        super().__init__()
        self.family, self.lookback = family, lookback
        if family in ('GRU', 'ALSTM'):
            self.rnn = nn.GRU(features, hidden, batch_first=True)
            if family == 'ALSTM':
                self.attn = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(),
                                          nn.Linear(hidden, 1))
        else:
            self.embed = nn.Linear(features, hidden)
            self.pos = nn.Parameter(torch.randn(1, lookback, hidden) * 0.02)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden, nhead=4,
                                           dim_feedforward=2*hidden, dropout=0.0,
                                           batch_first=True, activation='gelu'),
                num_layers=1, enable_nested_tensor=False)
        self.output = mlp(hidden, 1, hidden=hidden, depth=1)

    def forward(self, x):
        if self.family in ('GRU', 'ALSTM'):
            seq, _ = self.rnn(x)
            if self.family == 'ALSTM':
                attn = torch.softmax(self.attn(seq), dim=1)
                h = (seq * attn).sum(dim=1)
            else:
                h = seq[:, -1]
        else:
            h = self.transformer(self.embed(x) + self.pos).mean(dim=1)
        return self.output(h)


def build_predictors(m, device):
    configs = [('GRU', 64, 'mse'), ('GRU', 128, 'smooth_l1'),
               ('ALSTM', 64, 'mse'), ('ALSTM', 128, 'smooth_l1'),
               ('Transformer', 64, 'mse'), ('Transformer', 128, 'smooth_l1')]
    return nn.ModuleList([Predictor(name, m.feature_count, hidden, m.lookback)
                          for name, hidden, _ in configs]).to(device), configs


def pretrain_predictors(m, models, configs, args, device):
    low, high = m.lookback - 1, m.train_end - 1
    for i, (model, (_, hidden, lossname)) in enumerate(zip(models, configs)):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.predictor_lr)
        for u in range(args.predictor_updates):
            _, x, y = sample_windows(m, low, high, args.batch_size, device)
            yp = model(x)
            loss = F.mse_loss(yp, y) if lossname == 'mse' else F.smooth_l1_loss(yp, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(f'predictor {i + 1}/6: {configs[i][0]} hidden={hidden} '
              f'loss={lossname} final_train_loss={loss.item():.5f}')


@torch.no_grad()
def predict_all(m, models, device, batch=256):
    for model in models:
        model.eval()
    result = np.zeros((m.n, 6), np.float32)
    for i in range(m.lookback - 1, m.n - 1, batch):
        ids = range(i, min(m.n - 1, i + batch))
        xx = torch.as_tensor(np.stack([m.window(t) for t in ids]), device=device)
        y = torch.cat([model(xx) for model in models], dim=1)
        result[i:i + len(ids)] = y.cpu().numpy()
    # At bar t: abs error for prediction made at t-1; result of bar t is known.
    errors = np.zeros_like(result)
    errors[1:] = np.abs(result[:-1] - m.target_bps[:-1, None])
    return result, errors


class WeightActor(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.pred_mlp = mlp(6, 64, hidden=64, depth=2)  # 3 linear + 2 ReLU
        self.error_mlp = mlp(6, 64, hidden=64, depth=2)
        self.fusion_mlp = mlp(130, 6, hidden=64, depth=2)
        self.temp = temp

    def forward(self, s):
        predictions = self.pred_mlp(s[:, :6])
        losses = self.error_mlp(s[:, 6:12])
        logits = self.fusion_mlp(torch.cat([predictions, losses, s[:, 12:]], dim=1))
        return torch.softmax(logits / self.temp, dim=1)


class DDPGWeightAgent:
    def __init__(self, device, lr=1e-4, gamma=0.99, tau=0.005, temp=1.0):
        self.device, self.gamma, self.tau = device, gamma, tau
        self.actor = WeightActor(temp).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = mlp(14+6, 1, hidden=128).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs, explore=False, noise=0.10):
        s = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        self.actor.eval()
        a = self.actor(s).cpu().numpy()[0]
        if explore:
            a = np.maximum(1e-6, a + np.random.normal(0, noise, size=6))
            a = a / a.sum()
        return a.astype(np.float32)

    def update(self, replay, batch):
        s, a, r, s2, done = replay.sample(batch, self.device)
        with torch.no_grad():
            a2 = self.actor_target(s2)
            target = r + (1.0 - done) * self.gamma * self.critic_target(
                torch.cat([s2, a2], dim=1))
        pred = self.critic(torch.cat([s, a], dim=1))
        critic_loss = F.mse_loss(pred, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()
        weights = self.actor(s)
        actor_loss = -self.critic(torch.cat([s, weights], dim=1)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()
        with torch.no_grad():
            for model, target_model in ((self.actor, self.actor_target),
                                        (self.critic, self.critic_target)):
                for p, pt in zip(model.parameters(), target_model.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)


class MetaSim:
    def __init__(self, market, predictions, old_errors, signal_scale):
        self.m, self.p, self.e, self.scale = market, predictions, old_errors, signal_scale

    def state(self, t, pos, capital):
        return np.r_[self.p[t]/10.0, np.clip(self.e[t]/10.0, 0, 10),
                     pos, np.log(max(capital, 1e-12))].astype(np.float32)

    def signal(self, t, weights):
        predicted_bps = float(np.dot(weights, self.p[t]))
        return float(np.tanh(predicted_bps / self.scale))


def train_meta(m, sim, agent, args, start_step, steps):
    train_start, train_end = m.partitions()['train']
    replay = Replay(args.buffer_size, 14, 6)
    t, pos, equity = train_end, 0.0, 1.0
    for step in range(steps):
        if step == 0 or t >= ep_end:
            span = min(args.episode_bars, train_end - train_start)
            t = int(np.random.randint(train_start, train_end-span+1))
            ep_end = t + span
            pos, equity = 0.0, 1.0
        s = sim.state(t, pos, equity)
        weights = np.random.dirichlet(np.ones(6)).astype(np.float32) \
                  if step + start_step < args.warmup else agent.act(s, explore=True)
        exposure = sim.signal(t, weights)
        equity2, lr, _ = m.transition(t, pos, exposure, equity)
        s2 = sim.state(t+1, exposure, equity2)
        terminal = t+1 >= ep_end
        replay.push(s, weights, lr * args.reward_scale, s2, terminal)
        t, pos, equity = t + 1, exposure, equity2
        if replay.count >= args.batch_size and step + start_step >= args.warmup:
            agent.update(replay, args.batch_size)
        if (step+1) % max(1, steps//5) == 0:
            print(f'meta-RL {start_step+step+1}/{args.steps}: last_reward={lr:.6f}, '
                  f'replay={replay.count}')


def joint_fine_tune(m, models, sim, agent, args, device):
    """Alternating adjustment of the six predictors using PnL + prediction losses.

    Unlike ordinary frozen-encoder DDPG, gradients flow from differentiable
    one-step trading PnL THROUGH the trained weight actor to base predictors.
    The critic is not backpropagated into predictors; partial paper adaptation.
    """
    if args.joint_updates <= 0:
        return
    print(f'joint tuning predictors for {args.joint_updates} steps (train split ONLY)')
    for p in agent.actor.parameters():
        p.requires_grad_(False)
    agent.actor.eval()
    for model in models:
        model.train()
    optimizer = torch.optim.Adam(models.parameters(), lr=args.joint_lr)
    low, high = m.lookback, m.train_end - 1
    for u in range(args.joint_updates):
        ids, x, y = sample_windows(m, low, high, args.batch_size, device)
        current_preds = torch.cat([model(x) for model in models], dim=1)
        known_errors = torch.tensor(np.clip(sim.e[ids]/10., 0, 10),
                                    dtype=torch.float32, device=device)
        # Previous exposure estimated causally using last bar's predictions.
        previous_states = np.stack([sim.state(int(t-1), 0.0, 1.0) for t in ids])
        with torch.no_grad():
            previous_states = torch.tensor(previous_states, dtype=torch.float32, device=device)
            weights_old = agent.actor(previous_states)
            previous_bps = torch.tensor(sim.p[ids-1], device=device)
            old_pos = torch.tanh((weights_old * previous_bps).sum(1) / args.signal_scale)
        current_state = torch.cat((current_preds / 10., known_errors,
                                   old_pos[:, None].detach(), torch.zeros((len(ids), 1), device=device)), dim=1)
        current_weights = agent.actor(current_state)
        signal = torch.tanh((current_weights * current_preds).sum(1) / args.signal_scale)
        gap = torch.tensor(m.open[ids+1] / m.close[ids] - 1.0,
                           dtype=torch.float32, device=device)
        intrabar = torch.tensor(m.close[ids+1] / m.open[ids+1] - 1.0,
                                dtype=torch.float32, device=device)
        transaction_cost = torch.tensor(m.spread_frac[ids+1] * .5 + m.fee_fraction,
                                        dtype=torch.float32, device=device)
        net = (1.0 + old_pos * gap + signal * intrabar -
               torch.abs(signal - old_pos) * transaction_cost)
        pnl_loss = -torch.log(torch.clamp(net, min=1e-5)).mean() * args.reward_scale
        supervised_loss = F.smooth_l1_loss(current_preds, y.expand(-1, 6))
        loss = pnl_loss + args.supervised_weight * supervised_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(models.parameters(), 1.0)
        optimizer.step()
        if (u+1) % max(1, args.joint_updates//5) == 0:
            print(f'joint {u+1}/{args.joint_updates}: pnl_loss={pnl_loss.item():.5f} '
                  f'supervised_loss={supervised_loss.item():.5f}')
    for p in agent.actor.parameters():
        p.requires_grad_(True)


def main():
    parser = common_parser('DRL-DEM-inspired GRU/ALSTM/Transformer + weighting actor-critic', 15)
    parser.add_argument('--predictor-updates', type=int, default=500,
                        help='SGD updates PER predictor (six predictors).')
    parser.add_argument('--predictor-lr', type=float, default=3e-4)
    parser.add_argument('--joint-updates', type=int, default=150,
                        help='Trading-reward fine tuning of predictors between RL stages.')
    parser.add_argument('--joint-lr', type=float, default=1e-5)
    parser.add_argument('--supervised-weight', type=float, default=0.05)
    parser.add_argument('--steps', type=int, default=30000, help='Total DDPG-style agent transitions.')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--buffer-size', type=int, default=30000)
    parser.add_argument('--episode-bars', type=int, default=1024)
    parser.add_argument('--signal-scale', type=float, default=8.0,
                        help='Predicted return (bps) divided by this before tanh position.')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--reward-scale', type=float, default=1000.0)
    args = parser.parse_args()
    m, device = build_market(args)
    models, configs = build_predictors(m, device)
    pretrain_predictors(m, models, configs, args, device)
    predictions, errors = predict_all(m, models, device)
    sim = MetaSim(m, predictions, errors, args.signal_scale)
    agent = DDPGWeightAgent(device, temp=args.temperature)
    first_steps = args.steps//2 if args.joint_updates else args.steps
    train_meta(m, sim, agent, args, 0, first_steps)
    if args.joint_updates:
        joint_fine_tune(m, models, sim, agent, args, device)
        predictions, errors = predict_all(m, models, device)
        sim = MetaSim(m, predictions, errors, args.signal_scale)
        train_meta(m, sim, agent, args, first_steps, args.steps-first_steps)

    results = {}
    for segment in ('validation', 'test'):
        start, end = m.partitions()[segment]
        stat, _ = backtest(m, lambda _obs, t, pos, eq: sim.signal(
            t, agent.act(sim.state(t, pos, eq), explore=False)), start, end)
        results[segment] = stat
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'predictors': models.state_dict(), 'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(), 'predictor_configs': configs,
                'scaler_mean': m.mean, 'scaler_std': m.std,
                'lookback': m.lookback, 'signal_scale': args.signal_scale,
                'feature_count': m.feature_count}, out/'drl_dem.pt')
    write_results(out, args, results)


if __name__ == '__main__':
    main()
