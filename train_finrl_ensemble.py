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

# ===== Approach 3: FinRL-style Ensemble PPO + A2C + DDPG ====================
# Research basis: v8 FinRL framework and v10 ensemble strategy.
# Adaptation for ONE XAUUSD market: each policy chooses continuous exposure
# [-1, 1] rather than separate share quantities for 30 stocks.
# Weights are selected by Sharpe on a trailing, strictly PAST window;
# the paper also periodically retrains models, whereas here models stay fixed
# after training to give a clean held-out test.

class GaussianPolicyValue(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.policy = mlp(obs_dim, 1, hidden=128)
        self.log_std = nn.Parameter(torch.tensor([-0.7]))
        self.value = mlp(obs_dim, 1, hidden=128)

    def sample(self, observation, deterministic=False):
        mean = self.policy(observation)
        std = self.log_std.clamp(-5, 1.5).exp()
        z = mean if deterministic else mean + std * torch.randn_like(mean)
        a = torch.tanh(z)
        logp = self.log_prob(observation, a)
        return a, logp, self.value(observation)

    def log_prob(self, obs, action):
        mean = self.policy(obs)
        std = self.log_std.clamp(-5, 1.5).exp()
        x = action.clamp(-1+1e-6, 1-1e-6)
        unscaled = torch.atanh(x)
        logp = (-0.5*((unscaled-mean)/std).square() - torch.log(std)
                - 0.5 * math.log(2 * math.pi) - torch.log(1-x.square()+1e-6))
        return logp.sum(-1, keepdim=True)


class DDPGPolicy:
    def __init__(self, obs_dim, device, lr=3e-4, gamma=0.99, tau=0.005):
        self.device, self.gamma, self.tau = device, gamma, tau
        self.actor = nn.Sequential(mlp(obs_dim, 1, 128), nn.Tanh()).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = mlp(obs_dim+1, 1, 128).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        s = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
        return float(self.actor(s).item())

    def update(self, replay, batch):
        s, a, r, s2, d = replay.sample(batch, self.device)
        with torch.no_grad():
            target_q = r + self.gamma * (1-d) * self.critic_target(
                torch.cat((s2, self.actor_target(s2)), dim=1))
        q = self.critic(torch.cat((s, a), dim=1))
        loss_q = F.mse_loss(q, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        loss_a = -self.critic(torch.cat((s, self.actor(s)), dim=1)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        with torch.no_grad():
            for src, dest in ((self.actor, self.actor_target),
                              (self.critic, self.critic_target)):
                for p, pt in zip(src.parameters(), dest.parameters()):
                    pt.mul_(1-self.tau).add_(self.tau*p)


class StochasticPolicy:
    def __init__(self, obs_dim, device, lr=3e-4):
        self.device = device
        self.model = GaussianPolicyValue(obs_dim).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        x = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
        a, _, _ = self.model.sample(x, deterministic=deterministic)
        return float(a.item())


def random_episode(m, length):
    start, end = m.partitions()['train']
    length = min(length, end-start)
    s = int(np.random.randint(start, end-length+1))
    return s, s+length


def train_ddpg(m, args, device, obs_dim):
    ag = DDPGPolicy(obs_dim, device, lr=args.lr, gamma=args.gamma)
    buffer = Replay(args.buffer_size, obs_dim, 1)
    t = end = 0
    position, equity = 0., 1.
    for step in range(args.steps):
        if step == 0 or t >= end:
            t, end = random_episode(m, args.episode_bars)
            position, equity = 0., 1.
        s = m.obs(t, position, equity)
        if step < args.warmup:
            a = float(np.random.uniform(-1, 1))
        else:
            a = float(np.clip(ag.act(s) + np.random.normal(0, args.ddpg_noise), -1, 1))
        new_equity, lr, _ = m.transition(t, position, a, equity)
        next_s = m.obs(t+1, a, new_equity)
        buffer.push(s, [a], lr * args.reward_scale, next_s, t+1 >= end)
        t, position, equity = t+1, a, new_equity
        if step >= args.warmup and buffer.count >= args.batch_size:
            ag.update(buffer, args.batch_size)
        if (step+1) % max(1, args.steps//5) == 0:
            print(f'DDPG: {step+1}/{args.steps} transitions')
    return ag


def train_policy_gradient(m, args, device, obs_dim, method):
    agent = StochasticPolicy(obs_dim, device, lr=args.lr)
    total = 0
    while total < args.steps:
        t, ep_end = random_episode(m, args.episode_bars)
        pos, equity = 0.0, 1.0
        while t < ep_end and total < args.steps:
            trajectory = []
            roll = min(args.rollout, ep_end-t, args.steps-total)
            for _ in range(roll):
                obs = m.obs(t, pos, equity)
                tensor = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
                with torch.no_grad():
                    at, lp, v = agent.model.sample(tensor)
                action = float(at.item())
                new_equity, reward, _ = m.transition(t, pos, action, equity)
                done = t+1 >= ep_end
                trajectory.append((obs, action, float(lp.item()), float(v.item()),
                                   reward * args.reward_scale, float(done)))
                t, pos, equity = t+1, action, new_equity
                total += 1
            with torch.no_grad():
                last_value = 0.0 if t >= ep_end else float(agent.model.value(
                    torch.as_tensor(m.obs(t, pos, equity)[None],
                                    dtype=torch.float32, device=device)).item())
            # Generalized Advantage Estimation on causally collected trajectories.
            advantages = np.zeros(len(trajectory), dtype=np.float32)
            gae = 0.0
            for j in reversed(range(len(trajectory))):
                _, _, _, val, reward, done = trajectory[j]
                next_value = last_value if j == len(trajectory)-1 else trajectory[j+1][3]
                td = reward + args.gamma * (1-done) * next_value - val
                gae = td + args.gamma * args.gae_lambda * (1-done) * gae
                advantages[j] = gae
            obs_batch = torch.as_tensor(np.stack([x[0] for x in trajectory]), device=device)
            actions = torch.tensor([[x[1]] for x in trajectory], dtype=torch.float32, device=device)
            old_logp = torch.tensor([[x[2]] for x in trajectory], dtype=torch.float32, device=device)
            vals_old = np.array([x[3] for x in trajectory], dtype=np.float32)
            ret = torch.as_tensor((advantages + vals_old)[:, None], device=device)
            adv = torch.as_tensor(advantages[:, None], device=device)
            if len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            # A2C: one synchronous actor-critic update. PPO: clipped multi-epoch SGD.
            epochs = args.ppo_epochs if method == 'PPO' else 1
            batch = args.batch_size if method == 'PPO' else len(trajectory)
            for _ in range(epochs):
                indices = np.random.permutation(len(trajectory))
                for i in range(0, len(indices), batch):
                    ix = indices[i:i+batch]
                    new_logp = agent.model.log_prob(obs_batch[ix], actions[ix])
                    values = agent.model.value(obs_batch[ix])
                    ratio = torch.exp((new_logp - old_logp[ix]).clamp(-10, 10))
                    if method == 'PPO':
                        clipped = torch.clamp(ratio, 1-args.clip, 1+args.clip)
                        loss_policy = -torch.minimum(ratio*adv[ix], clipped*adv[ix]).mean()
                    else:
                        loss_policy = -(new_logp * adv[ix]).mean()
                    loss_value = F.mse_loss(values, ret[ix])
                    # Gaussian entropy bonus: exact (pre-tanh) normal entropy.
                    entropy = (agent.model.log_std.clamp(-5, 1.5) +
                               0.5 * math.log(2 * math.pi * math.e)).mean()
                    loss = loss_policy + args.value_coef*loss_value - args.entropy_coef*entropy
                    agent.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.model.parameters(), args.grad_clip)
                    agent.opt.step()
            if total % max(1, args.steps//5) < roll:
                print(f'{method}: {total}/{args.steps} transitions, last_value_loss={loss_value.item():.5f}')
    return agent


def rolling_selector(m, agents, args):
    """At each decision block, select using ONLY preceding bars' Sharpe."""
    t, end = m.partitions()['test']
    pos, equity = 0.0, 1.0
    curve, reward_log, turnover = [equity], [], 0.0
    decisions = []
    while t < end:
        # Initial selection window lies in validation. Later windows may include
        # already-observed test history, never a future bar.
        past_start = max(m.train_end-1, t-args.selection_bars)
        if t-past_start < 2:
            raise ValueError('Selection window too short.')
        scores = {}
        for name, ag in agents.items():
            stats, _ = backtest(m, lambda s, *_: ag.act(s), past_start, t)
            scores[name] = stats
        name = max(scores, key=lambda k: (scores[k]['sharpe_annualized'],
                                          scores[k]['total_return']))
        target_end = min(t+args.rebalance_bars, end)
        decisions.append({'decision_start_bar': int(t), 'end_bar': int(target_end),
                          'selected': name, 'past_sharpe': {k: v['sharpe_annualized']
                                                            for k, v in scores.items()}})
        ag = agents[name]
        for step in range(t, target_end):
            s = m.obs(step, pos, equity)
            target = float(ag.act(s))
            next_equity, r, tv = m.transition(step, pos, target, equity)
            curve.append(next_equity)
            reward_log.append(r)
            turnover += tv
            equity, pos = next_equity, target
        print(f'ensemble: bars {t}..{target_end}, selected {name}; '
              f'past Sharpe={scores[name]["sharpe_annualized"]:.3f}')
        t = target_end
    return metrics(curve, reward_log, turnover, m.bars_per_year), decisions, curve


def main():
    parser = common_parser('FinRL-inspired PPO / A2C / DDPG with rolling Sharpe selection', 32)
    parser.add_argument('--steps', type=int, default=30000,
                        help='Training transitions PER agent, i.e. 3 x this value.')
    parser.add_argument('--rollout', type=int, default=256)
    parser.add_argument('--ppo-epochs', type=int, default=4)
    parser.add_argument('--episode-bars', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--ddpg-noise', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.002)
    parser.add_argument('--grad-clip', type=float, default=0.5)
    parser.add_argument('--reward-scale', type=float, default=1000.0)
    parser.add_argument('--selection-bars', type=int, default=1000,
                        help='Length of past Sharpe window (bars, not days).')
    parser.add_argument('--rebalance-bars', type=int, default=1000,
                        help='Execute selected policy this many bars before reselecting.')
    args = parser.parse_args()
    if args.selection_bars < 2 or args.rebalance_bars < 1:
        parser.error('Invalid selection/rebalance period.')
    m, device = build_market(args)
    obs_dim = m.lookback * m.feature_count + 2
    print('Training PPO...')
    ppo = train_policy_gradient(m, args, device, obs_dim, 'PPO')
    print('Training A2C...')
    a2c = train_policy_gradient(m, args, device, obs_dim, 'A2C')
    print('Training DDPG...')
    ddpg = train_ddpg(m, args, device, obs_dim)
    agents = {'PPO': ppo, 'A2C': a2c, 'DDPG': ddpg}
    results = {'individual': {}}
    for name, agent in agents.items():
        results['individual'][name] = {}
        for segment in ('validation', 'test'):
            start, end = m.partitions()[segment]
            stats, _ = backtest(m, lambda s, *_: agent.act(s), start, end)
            results['individual'][name][segment] = stats
    result_ensemble, decisions, _ = rolling_selector(m, agents, args)
    results['ensemble_test'] = result_ensemble
    results['selection_counts'] = {k: sum(d['selected'] == k for d in decisions)
                                   for k in agents}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': ppo.model.state_dict(), 'lookback': m.lookback,
                'scaler_mean': m.mean, 'scaler_std': m.std}, out/'ppo.pt')
    torch.save({'state_dict': a2c.model.state_dict(), 'lookback': m.lookback,
                'scaler_mean': m.mean, 'scaler_std': m.std}, out/'a2c.pt')
    torch.save({'actor': ddpg.actor.state_dict(), 'critic': ddpg.critic.state_dict(),
                'lookback': m.lookback, 'scaler_mean': m.mean,
                'scaler_std': m.std}, out/'ddpg.pt')
    with (out/'selection_history.json').open('w', encoding='utf8') as f:
        json.dump(decisions, f, indent=2)
    write_results(out, args, results)


if __name__ == '__main__':
    main()
