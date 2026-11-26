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
    p.add_argument('--out',default=Path(__file__).resolve().parents[1] / 'data' / 'training_runs' / 'stockformer')
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



# ===== Approach 1: StockFormer-inspired Predictive Coding + SAC ===============
# Research basis: NEUR1, Gao et al., IJCAI 2023. This is an ADAPTATION:
# one XAUUSDT instrument, so the paper's inter-asset relational branch becomes
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
        # Cross-feature relations replace cross-asset relations for XAUUSDT.
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
    parser = common_parser('StockFormer-inspired three-branch Transformer + SAC for XAUUSDT M1', 50)
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
                'scaler_mean': m.mean, 'scaler_std': m.std, 'feature_names': FEATURE_NAMES, 'feature_count': m.feature_count,
                'lookback': m.lookback, 'd_model': args.d_model, 'heads': args.heads,
                'layers': args.layers}, out / 'stockformer_sac.pt')
    write_results(out, args, results)


if __name__ == '__main__':
    main()
