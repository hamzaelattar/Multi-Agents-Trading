#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# XAUUSDT Binance, M1, 35-column schema, 50-bar lookback

"""COMMON CORE embedded into each standalone training script.

This file is a build-time source fragment only: the three delivered scripts each
include this code, so end users do not need shared local imports.
"""
import argparse
import copy
import json
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# EXACT column order provided by the project owner: 35 CSV columns.
CSV_COLUMNS = [
    'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'sma_5', 'sma_10', 'sma_20', 'sma_50',
    'ema_5', 'ema_10', 'ema_20', 'ema_50',
    'rsi_7', 'rsi_14',
    'macd_5_13_4', 'macd_signal_5_13_4', 'macd_hist_5_13_4',
    'macd_5_26_4', 'macd_signal_5_26_4', 'macd_hist_5_26_4',
    'stoch_k_9_3', 'stoch_d_9_3', 'stoch_k_14_3', 'stoch_d_14_3',
    'cci_14', 'atr_7', 'atr_14',
    'bb_upper_20_2', 'bb_middle_20_2', 'bb_lower_20_2',
    'obv_diff_10', 'obv_diff_20',
]
FEATURE_NAMES = CSV_COLUMNS[2:]  # exactly 33 numerical features
PRICE_COLUMNS = ['open', 'high', 'low', 'close']
LEVEL_COLUMNS = [name for name in FEATURE_NAMES if name.startswith(('sma_', 'ema_', 'bb_'))]
RATIO_COLUMNS = [name for name in FEATURE_NAMES if name.startswith(('macd_', 'atr_'))]
OBV_COLUMNS = ['obv_diff_10', 'obv_diff_20']


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Market:
    """One-symbol M1 market: causally scaled 33 features and 50-bar sequences.

    Data contract:
    - CSV contains ALL 35 named columns in the exact supplied order.
    - Every row is a one-minute Binance XAUUSDT candle, timestamp UTC.
    - Technical indicator NaNs are permitted ONLY as a leading warm-up prefix.
    - Indicators must have been computed causally from past/current closed bars.
    - Decision occurs AFTER close of t; execution is approximated at OPEN t+1.
    - No bid/ask columns: assumed full spread and fees are user-configurable.
    - This is an exposure simulator, NOT a matching-engine or futures simulator.
    """

    def __init__(self, csv_path, lookback=50, spread_bps=2.0, fee_bps=0.0,
                 slippage_bps=0.0, train_fraction=0.70, val_fraction=0.15,
                 max_rows=0, allow_gaps=False):
        if lookback < 2:
            raise ValueError('lookback must be >=2')
        if min(spread_bps, fee_bps, slippage_bps) < 0:
            raise ValueError('Spread, fee and slippage must be >= 0 bps.')
        if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and
                train_fraction + val_fraction < 1):
            raise ValueError('Invalid train/validation fractions.')
        df = pd.read_csv(csv_path, dtype={'symbol': 'string'}, low_memory=False)
        if list(df.columns) != CSV_COLUMNS:
            missing = [c for c in CSV_COLUMNS if c not in df.columns]
            extra = [c for c in df.columns if c not in CSV_COLUMNS]
            raise ValueError(f'CSV schema does not match the exact 35 columns. '
                             f'Missing={missing}; extra={extra}; '
                             f'wrong_order={not (missing or extra)}')
        if len(df) < max(160, lookback * 3 + 30):
            raise ValueError('CSV is too short for 50-bar lookback and 3 chronological splits.')
        if df['symbol'].isna().any() or set(df['symbol']) != {'XAUUSDT'}:
            raise ValueError("Expected exactly one market symbol: 'XAUUSDT'.")
        ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        if ts.isna().any():
            raise ValueError('Invalid UTC timestamps in CSV.')
        df = df.assign(timestamp=ts).sort_values('timestamp').reset_index(drop=True)
        if max_rows > 0 and len(df) > max_rows:
            df = df.iloc[-max_rows:].reset_index(drop=True)
            print(f'INFO: --max-rows limits this run to the last {len(df):,} CSV rows.')
        if df['timestamp'].duplicated().any():
            raise ValueError('Duplicated (timestamp, symbol) rows; correct source data first.')
        deltas = df['timestamp'].diff().dt.total_seconds().iloc[1:]
        gaps = (deltas != 60).sum()
        if gaps:
            explanation = (f'{gaps} timestamp gap(s): expected exactly one row per minute. '
                           'Fix the dataset, or pass --allow-gaps to knowingly use irregular bars.')
            if not allow_gaps:
                raise ValueError(explanation)
            print('WARNING:', explanation)

        for name in FEATURE_NAMES:
            df[name] = pd.to_numeric(df[name], errors='coerce')
        if not np.isfinite(df[PRICE_COLUMNS + ['volume']].to_numpy(np.float64)).all():
            raise ValueError('OHLCV contains missing or nonfinite values.')
        if (df[PRICE_COLUMNS] <= 0).any().any() or (df['volume'] < 0).any():
            raise ValueError('OHLC must be >0 and volume nonnegative.')
        if ((df['high'] < df[PRICE_COLUMNS].max(axis=1)) |
                (df['low'] > df[PRICE_COLUMNS].min(axis=1))).any():
            raise ValueError('Inconsistent OHLC (high/low do not bound open/close).')
        complete = np.isfinite(df[FEATURE_NAMES].to_numpy(np.float64)).all(axis=1)
        good = np.flatnonzero(complete)
        if len(good) == 0:
            raise ValueError('No fully populated indicator row. Please check indicator generation.')
        warmup = int(good[0])
        if not complete[warmup:].all():
            raise ValueError('Indicator NaN/inf after warm-up. Only a leading NaN prefix is allowed.')
        if warmup:
            print(f'INFO: dropping {warmup} leading candle(s) with incomplete indicators.')
        df = df.iloc[warmup:].reset_index(drop=True)
        if len(df) < max(120, 3 * lookback + 30):
            raise ValueError('Not enough COMPLETE indicator bars left for 3 splits.')

        self.times = df['timestamp']
        self.symbol = 'XAUUSDT'
        self.open = df['open'].to_numpy(np.float64)
        self.high = df['high'].to_numpy(np.float64)
        self.low = df['low'].to_numpy(np.float64)
        self.close = df['close'].to_numpy(np.float64)
        self.n = len(df)
        self.lookback = lookback
        self.fee_fraction = fee_bps / 10000.0
        self.slippage_fraction = slippage_bps / 10000.0
        self.spread_frac = np.full(self.n, spread_bps / 10000.0)
        print(f'ASSUMED COSTS: full spread={spread_bps} bps, fee={fee_bps} bps '
              f'per side, slippage={slippage_bps} bps per side; NO measured bid/ask.')

        # One transformed feature per numerical CSV column (33 total).
        # Price levels are expressed relative to the current close or previous close
        # to reduce nonstationarity. These computations use only bars <= t.
        close = df['close'].astype('float64')
        prev_close = close.shift(1).fillna(close)
        raw_cols = []
        for name in FEATURE_NAMES:
            v = df[name].to_numpy(np.float64)
            if name in PRICE_COLUMNS:
                x = np.log(v / prev_close.to_numpy(np.float64))
            elif name == 'volume':
                x = np.log1p(v)
            elif name in LEVEL_COLUMNS:
                x = np.log(v / self.close)
            elif name in RATIO_COLUMNS:
                x = 10000.0 * v / self.close
            elif name in OBV_COLUMNS:
                x = np.sign(v) * np.log1p(np.abs(v))
            else:
                x = v  # RSI, stochastic, CCI
            raw_cols.append(x)
        raw = np.column_stack(raw_cols)
        if not np.isfinite(raw).all():
            raise ValueError('Feature transformation produced NaN/inf; inspect data scale.')
        self.train_end = int(self.n * train_fraction)
        self.val_end = int(self.n * (train_fraction + val_fraction))
        if min(self.train_end - lookback, self.val_end-self.train_end,
               self.n-self.val_end) < 20:
            raise ValueError('Each train/validation/test segment needs >= 20 decisions.')
        mu = raw[:self.train_end].mean(axis=0)
        std = raw[:self.train_end].std(axis=0)
        std[std < 1e-9] = 1.0
        self.mean, self.std = mu.astype(np.float32), std.astype(np.float32)
        self.features = np.clip((raw - mu) / std, -8., 8.).astype(np.float32)
        self.feature_count = self.features.shape[1]
        assert self.feature_count == 33

        # Label at t: NEXT bar open-to-close return (unknown until after t+1).
        self.target_bps = np.zeros(self.n, np.float32)
        self.target_bps[:-1] = (10000.0 * np.log(self.close[1:] / self.open[1:])).astype(np.float32)
        # Binance-style always-on M1 assumption. Override if market/session differs.
        self.bars_per_year = 365 * 24 * 60
        print(f'ROWS={self.n:,}; chronological splits train={self.train_end:,}, '
              f'val={self.val_end-self.train_end:,}, test={self.n-self.val_end:,}; '
              f'input_features=33, lookback={lookback}, bars/year={self.bars_per_year:,}')
        print(f'UTC RANGE: {self.times.iloc[0]} -> {self.times.iloc[-1]}')
        print('NOTICE: XAUUSDT (Binance) is NOT spot XAUUSD (Dukascopy). '
              'No margin, leverage, funding, liquidation or order-book execution is modeled.')

    def window(self, t):
        return self.features[t-self.lookback+1:t+1]

    def obs(self, t, pos, equity):
        return np.concatenate((self.window(t).reshape(-1),
                               np.array([pos, np.log(max(equity, 1e-12))], np.float32)))

    def transition(self, t, old_pos, target_pos, equity):
        """Close(t) decision, OPEN(t+1) rebalance, CLOSE(t+1) mark-to-market.

        target_pos in [-1,1]: -1 short, 0 flat, +1 long. Simulated exposure;
        shorting is hypothetical and may not be available on the user's market.
        """
        j = t + 1
        target_pos = float(np.clip(target_pos, -1.0, 1.0))
        gap_return = self.open[j] / self.close[t] - 1.0
        bar_return = self.close[j] / self.open[j] - 1.0
        turnover = abs(target_pos-old_pos)
        cost = turnover * (self.fee_fraction + self.slippage_fraction +
                           0.5*self.spread_frac[j])
        simple = old_pos*gap_return + target_pos*bar_return - cost
        ratio = max(1e-8, 1.0+simple)
        new_equity = float(equity*ratio)
        return new_equity, float(np.log(ratio)), turnover

    def partitions(self):
        return {'train': (self.lookback-1, self.train_end-1),
                'validation': (self.train_end-1, self.val_end-1),
                'test': (self.val_end-1, self.n-1)}


def metrics(equity_curve, log_rewards, turnover, bars_per_year):
    curve = np.asarray(equity_curve, dtype=float)
    returns = np.expm1(np.asarray(log_rewards, dtype=float))
    if len(returns)>1 and returns.std(ddof=1)>1e-12:
        sharpe = float(np.sqrt(bars_per_year)*returns.mean()/returns.std(ddof=1))
    else:
        sharpe = 0.0
    peaks = np.maximum.accumulate(curve)
    return {'total_return': float(curve[-1]/curve[0]-1.),
            'sharpe_annualized': sharpe,
            'max_drawdown': float((curve/np.maximum(peaks, 1e-12)-1.).min()),
            'turnover': float(turnover), 'bars': int(len(returns)),
            'ending_equity': float(curve[-1])}


@torch.no_grad()
def backtest(market, policy, start, end, bars_per_year=None):
    pos, equity = 0.0, 1.0
    curve, rewards, turnover = [1.0], [], 0.0
    for t in range(start, end):
        observation = market.obs(t, pos, equity)
        action = float(policy(observation, t, pos, equity))
        equity, reward, tv = market.transition(t, pos, action, equity)
        pos = float(np.clip(action, -1, 1))
        curve.append(equity)
        rewards.append(reward)
        turnover += tv
    return metrics(curve, rewards, turnover, bars_per_year or market.bars_per_year), curve


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
        self.s[i], self.a[i], self.r[i,0] = s,a,r
        self.s2[i], self.done[i,0] = s2,float(done)
        self.i = (i+1)%self.cap
        self.count = min(self.count+1,self.cap)

    def sample(self, batch, device):
        idx = np.random.randint(0,self.count,size=batch)
        return tuple(torch.as_tensor(arr[idx],device=device) for arr in
                     (self.s,self.a,self.r,self.s2,self.done))


def mlp(inputs,outputs,hidden=128,depth=2):
    layers=[]
    dim=inputs
    for _ in range(depth):
        layers += [nn.Linear(dim,hidden),nn.ReLU()]
        dim=hidden
    layers.append(nn.Linear(dim,outputs))
    return nn.Sequential(*layers)


def write_results(out, config, results):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'metrics.json').open('w',encoding='utf8') as f:
        json.dump({'configuration':vars(config), 'expected_csv_columns':CSV_COLUMNS,
                   'model_features':FEATURE_NAMES,
                   'feature_preprocessing':'price returns/levels relative to close; volume log1p; momentum raw; MACD/ATR relative in bps; OBV signed log1p; train-only z-score',
                   'execution':'bar t closed -> trade at next open -> mark at next close; assumed spread/fees/slippage',
                   'results':results},f,indent=2,default=str)
    print('METRICS:',json.dumps(results,indent=2))
    print('Saved:',out.resolve())


def common_parser(description, default_lookback=50):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--csv',required=True,help='Exact 35-column Binance XAUUSDT M1 CSV.')
    p.add_argument('--out',default='runs/finrl_ensemble')
    p.add_argument('--lookback',type=int,default=default_lookback,
                   help='Number of last closed M1 candles; project default=50.')
    p.add_argument('--max-rows',type=int,default=0,
                   help='0=all rows; otherwise last N rows (for pilot runs).')
    p.add_argument('--allow-gaps',action='store_true',
                   help='Allow non-1-minute time gaps; use carefully.')
    p.add_argument('--spread-bps',type=float,default=2.0,
                   help='ASSUMED full bid/ask spread, not observable in this CSV.')
    p.add_argument('--fee-bps',type=float,default=0.0,
                   help='ASSUMED fee per side, in basis points.')
    p.add_argument('--slippage-bps',type=float,default=0.0,
                   help='ASSUMED slippage per side, in basis points.')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--bars-per-year',type=int,default=0,
                   help='0=365*24*60 for 24/7 M1. Adjust for session schedule.')
    return p


def build_market(args):
    seed_everything(args.seed)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available()
                        else 'cpu' if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    m=Market(args.csv,lookback=args.lookback,spread_bps=args.spread_bps,
             fee_bps=args.fee_bps,slippage_bps=args.slippage_bps,
             max_rows=args.max_rows,allow_gaps=args.allow_gaps)
    if args.bars_per_year>0:
        m.bars_per_year=args.bars_per_year
    print(f'Device={device}; split decision ranges: {m.partitions()}')
    return m,device


def sample_windows(market,low,high,batch,device):
    """Random causally observed windows, labels known within training split only."""
    ids=np.random.randint(low,high,size=batch)
    xx=np.stack([market.window(int(t)) for t in ids])
    yy=market.target_bps[ids]
    return ids,torch.from_numpy(xx).to(device),torch.from_numpy(yy).to(device).unsqueeze(-1)



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
    parser = common_parser('FinRL-inspired PPO / A2C / DDPG for Binance XAUUSDT M1', 50)
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
                'scaler_mean': m.mean, 'scaler_std': m.std, 'feature_names': FEATURE_NAMES}, out/'ppo.pt')
    torch.save({'state_dict': a2c.model.state_dict(), 'lookback': m.lookback,
                'scaler_mean': m.mean, 'scaler_std': m.std, 'feature_names': FEATURE_NAMES}, out/'a2c.pt')
    torch.save({'actor': ddpg.actor.state_dict(), 'critic': ddpg.critic.state_dict(),
                'lookback': m.lookback, 'scaler_mean': m.mean,
                'scaler_std': m.std}, out/'ddpg.pt')
    with (out/'selection_history.json').open('w', encoding='utf8') as f:
        json.dump(decisions, f, indent=2)
    write_results(out, args, results)


if __name__ == '__main__':
    main()
