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
    - No bid/ask columns: no spread is included; fees/slippage are configurable.
    - This is an exposure simulator, NOT a matching-engine or futures simulator.
    """

    def __init__(self, csv_path, lookback=50, fee_bps=0.0,
                 slippage_bps=0.0, train_fraction=0.70, val_fraction=0.15,
                 max_rows=0, allow_gaps=False):
        if lookback < 2:
            raise ValueError('lookback must be >=2')
        if min(fee_bps, slippage_bps) < 0:
            raise ValueError('Fee and slippage must be >= 0 bps.')
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
        print(f'COSTS: spread NOT modeled (CSV has no bid/ask); '
              f'assumed fee={fee_bps} bps per side, slippage={slippage_bps} bps per side.')

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
        cost = turnover * (self.fee_fraction + self.slippage_fraction)
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
                   'execution':'bar t closed -> trade at next open -> mark at next close; spread not modeled; optional assumed fees/slippage',
                   'results':results},f,indent=2,default=str)
    print('METRICS:',json.dumps(results,indent=2))
    print('Saved:',out.resolve())


def common_parser(description, default_lookback=50):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--csv',required=True,help='Exact 35-column Binance XAUUSDT M1 CSV.')
    p.add_argument('--out',default='runs/drl_dem')
    p.add_argument('--lookback',type=int,default=default_lookback,
                   help='Number of last closed M1 candles; project default=50.')
    p.add_argument('--max-rows',type=int,default=0,
                   help='0=all rows; otherwise last N rows (for pilot runs).')
    p.add_argument('--allow-gaps',action='store_true',
                   help='Allow non-1-minute time gaps; use carefully.')
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
    m=Market(args.csv,lookback=args.lookback,
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
        transaction_cost = torch.tensor(m.fee_fraction + m.slippage_fraction,
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
    parser = common_parser('DRL-DEM-inspired GRU/ALSTM/Transformer + weighting actor-critic for XAUUSDT M1', 50)
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
                'scaler_mean': m.mean, 'scaler_std': m.std, 'feature_names': FEATURE_NAMES,
                'lookback': m.lookback, 'signal_scale': args.signal_scale,
                'feature_count': m.feature_count}, out/'drl_dem.pt')
    write_results(out, args, results)


if __name__ == '__main__':
    main()
