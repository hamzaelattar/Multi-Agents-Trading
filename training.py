
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import math


class ALSTM_PPO(nn.Module):

    def __init__(self, n_features=33, hidden=128):
        super().__init__()

        # 1. LSTM
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            batch_first=True
        )

        # 2. Attention
        self.attention = nn.Linear(hidden, 1)

        # 3. Actor : mean et log_std
        self.actor = nn.Sequential(
            nn.Linear(hidden + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        # 4. Critic : V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, window, portfolio):

        # 1. LSTM
        sequence, _ = self.lstm(window)

        # 2. Attention
        scores = self.attention(sequence)
        weights = torch.softmax(scores, dim=1)

        # 3. Représentation finale
        h = (sequence * weights).sum(dim=1)

        # 4. Ajouter les 4 informations du portefeuille
        state = torch.cat([h, portfolio], dim=1)

        # 5. Actor
        mean, log_std = self.actor(state).chunk(2, dim=1)
        std = torch.exp(torch.clamp(log_std, -5, 1))

        # 6. Critic
        value = self.critic(state)

        return mean, std, value

    @torch.no_grad()
    def act(self, window, portfolio):

        mean, std, value = self.forward(window, portfolio)

        # Distribution des actions
        dist = Normal(mean, std)

        # Choisir une action
        raw_action = dist.sample()
        action = torch.tanh(raw_action)

        # Log-probabilité corrigée après tanh
        correction = 2 * (
            math.log(2)
            - raw_action
            - F.softplus(-2 * raw_action)
        )

        log_prob = (
            dist.log_prob(raw_action) - correction
        ).sum(dim=1)

        return action, raw_action, log_prob, value




def build_portfolio_state(portfolio, current_price):

    cash = portfolio["cash"]
    gold = portfolio["gold_quantity"]
    avg_price = portfolio["avg_buy_price"]

    if current_price <= 0:
        raise ValueError("Le prix doit être positif.")

    # 1. Calculer le bilan total
    equity = cash + gold * current_price

    if equity <= 0:
        raise ValueError("Le bilan doit être positif.")

    # 2. Proportion du portefeuille investie en or
    gold_ratio = gold * current_price / equity

    # 3. Prix d'achat moyen par rapport au prix actuel
    entry_price_ratio = (
        avg_price / current_price if gold > 0 else 0.0
    )

    # 4. Construire les 4 valeurs
    portfolio_state = torch.tensor([
        gold_ratio,
        np.log(equity),
        gold,
        entry_price_ratio
    ], dtype=torch.float32)

    return portfolio_state


def transition(portfolio, action, close_t, open_next,
               close_next, fee=0.0001, slippage=0.0001,
               timestamp=None):

    if min(close_t, open_next, close_next) <= 0:
        raise ValueError("Les prix doivent être positifs.")

    if fee < 0 or not (0 <= slippage < 1):
        raise ValueError("Frais ou slippage invalides.")

    # 1. Récupérer le portefeuille
    cash = portfolio["cash"]
    gold = portfolio["gold_quantity"]
    avg_price = portfolio["avg_buy_price"]

    # 2. Bilan avant la transaction
    old_equity = cash + gold * close_t

    if old_equity <= 0:
        raise ValueError("Le bilan doit être positif.")

    # 3. Limiter l'action
    action = float(np.clip(action, -1, 1))

    transaction = None

    # 4. ACHAT : pourcentage du cash disponible
    if action > 0 and cash > 1e-10:

        budget = cash * action

        price = open_next * (1 + slippage)

        # Quantité achetée, frais inclus dans le budget
        quantity = budget / (price * (1 + fee))

        fees = quantity * price * fee

        # Nouveau prix d'achat moyen
        new_gold = gold + quantity

        avg_price = (
            gold * avg_price + quantity * price
        ) / new_gold

        # Mise à jour du portefeuille
        cash -= quantity * price + fees
        gold = new_gold

        transaction = {
            "type": "BUY",
            "quantity": quantity,
            "price": price,
            "fees": fees,
            "timestamp": timestamp
        }

    # 5. VENTE : pourcentage de l'or possédé
    elif action < 0 and gold > 1e-10:

        quantity = gold * abs(action)

        price = open_next * (1 - slippage)

        fees = quantity * price * fee

        # Mise à jour du portefeuille
        cash += quantity * price - fees
        gold -= quantity

        # Si tout l'or a été vendu
        if gold < 1e-10:
            gold = 0.0
            avg_price = 0.0

        transaction = {
            "type": "SELL",
            "quantity": quantity,
            "price": price,
            "fees": fees,
            "timestamp": timestamp
        }

    # 6. Valeur actuelle de l'or
    gold_value = gold * close_next

    # 7. Nouveau bilan total
    new_equity = cash + gold_value

    # 8. Reward pour PPO
    reward = np.log(
        max(new_equity, 1e-12) / old_equity
    )

    # 9. Sauvegarder le portefeuille
    portfolio["cash"] = max(0.0, cash)
    portfolio["gold_quantity"] = max(0.0, gold)
    portfolio["avg_buy_price"] = avg_price
    portfolio["equity"] = new_equity

    # 10. Enregistrer la transaction
    if transaction is not None:
        portfolio["transactions"].append(transaction)

    return portfolio, float(reward)



class PPOBuffer:

    def __init__(self):
        self.data = []

    def add(self, window, portfolio_state,
            raw_action, log_prob, value, reward, done):

        experience = {
            "window": window.detach().clone(),
            "portfolio": portfolio_state.detach().clone(),
            "raw_action": raw_action.detach().clone(),
            "log_prob": log_prob.detach().clone(),
            "value": value.detach().clone(),
            "reward": float(reward),
            "done": bool(done)
        }

        self.data.append(experience)

    def clear(self):
        self.data.clear()



def compute_gae(buffer, last_value=0.0,
                gamma=0.99, gae_lambda=0.95):

    # 1. Récupérer les expériences
    data = buffer.data
    n = len(data)

    # 2. Préparer les résultats
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)

    gae = 0.0
    next_value = float(last_value)

    # 3. Parcourir les expériences à l'envers
    for t in reversed(range(n)):

        reward = data[t]["reward"]
        value = data[t]["value"].item()
        done = data[t]["done"]

        # 4. Vérifier si l'épisode continue
        mask = 1.0 - float(done)

        # 5. Erreur de prédiction du Critic
        delta = reward + gamma * next_value * mask - value

        # 6. Calculer l'Advantage
        gae = delta + gamma * gae_lambda * mask * gae

        advantages[t] = gae

        # 7. Cible pour entraîner le Critic
        returns[t] = gae + value

        next_value = value

    return (
        torch.tensor(advantages),
        torch.tensor(returns)
    )



def update_ppo(model, optimizer, buffer, advantages, returns,
               clip_eps=0.2, epochs=4):

    device = next(model.parameters()).device

    # 1. Récupérer les expériences
    windows = torch.cat(
        [e["window"] for e in buffer.data], dim=0
    ).to(device)

    portfolios = torch.cat(
        [e["portfolio"] for e in buffer.data], dim=0
    ).to(device)

    raw_actions = torch.cat(
        [e["raw_action"] for e in buffer.data], dim=0
    ).to(device)

    old_log_probs = torch.cat(
        [e["log_prob"] for e in buffer.data], dim=0
    ).to(device)

    advantages = advantages.to(device)
    returns = returns.to(device)

    # 2. Normaliser les avantages
    advantages = (
        (advantages - advantages.mean())
        / (advantages.std(unbiased=False) + 1e-8)
    )

    # 3. Entraîner plusieurs fois sur les expériences
    for epoch in range(epochs):

        # Nouvelle sortie de notre ALSTM + PPO
        mean, std, values = model(windows, portfolios)

        # Nouvelle distribution
        dist = Normal(mean, std)

        # Nouvelle log-probabilité des mêmes actions
        correction = 2 * (
            math.log(2)
            - raw_actions
            - F.softplus(-2 * raw_actions)
        )

        new_log_probs = (
            dist.log_prob(raw_actions) - correction
        ).sum(dim=1)

        # 4. Rapport entre nouvelle et ancienne politique
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 5. Loss de l'Actor (PPO Clipping)
        surr1 = ratio * advantages

        surr2 = torch.clamp(
            ratio, 1 - clip_eps, 1 + clip_eps
        ) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()

        # 6. Loss du Critic
        critic_loss = ((values.squeeze(1) - returns) ** 2).mean()

        # 7. Loss totale
        loss = actor_loss + 0.5 * critic_loss

        # 8. Modifier les poids du réseau
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=0.5
        )

        optimizer.step()

    return actor_loss.item(), critic_loss.item()



class TradingEnv:

    def __init__(self, features, opens, closes,
                 start=49, end=None):

        self.features = features
        self.opens = opens
        self.closes = closes

        self.start = start
        self.end = len(features) - 1 if end is None else end

        assert features.shape[1] == 33
        assert len(features) == len(opens) == len(closes)
        assert 49 <= self.start < self.end < len(features)

    # 1. Démarrer un épisode
    def reset(self):

        self.t = self.start

        self.portfolio = {
            "cash": 10000.0,
            "gold_quantity": 0.0,
            "avg_buy_price": 0.0,
            "equity": 10000.0,
            "transactions": []
        }

        return self.get_observation()

    # 2. Construire l'observation actuelle
    def get_observation(self):

        window = self.features[self.t-49:self.t+1]

        portfolio_state = build_portfolio_state(
            self.portfolio,
            self.closes[self.t]
        )

        window = torch.tensor(
            window, dtype=torch.float32
        ).unsqueeze(0)

        portfolio_state = portfolio_state.unsqueeze(0)

        return window, portfolio_state

    # 3. Exécuter une action PPO
    def step(self, action):

        if self.t >= self.end:
            raise RuntimeError("Épisode terminé : utiliser reset().")

        # Exécuter la transaction sur la bougie suivante
        self.portfolio, reward = transition(
            portfolio=self.portfolio,
            action=action,
            close_t=self.closes[self.t],
            open_next=self.opens[self.t+1],
            close_next=self.closes[self.t+1]
        )

        # Passer à la bougie suivante
        self.t += 1

        # Vérifier la fin de l'épisode
        done = self.t >= self.end

        # Nouvelle observation
        next_window, next_portfolio = self.get_observation()

        return next_window, next_portfolio, reward, done
