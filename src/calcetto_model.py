import numpy as np
import pyro
import pyro.distributions as dist
import torch

from .calcetto_data import CalcettoData

DEFAULT_K = 5.0
INCLUDE_K = False
MODEL_VICTORY = False
MODEL_GOALS = True
PRIOR = {
    "mu_skill": 6.0,
    "sigma_skill": 1,
    "loc_log_k": 0.0,
    "scale_log_k": 3.0,
}


def model(data: CalcettoData):
    mu_pior = torch.tensor(PRIOR["mu_skill"])
    sigma_pior = torch.tensor(PRIOR["sigma_skill"])

    skill = {
        p: pyro.sample(f"skill_{p}", dist.Normal(loc=mu_pior, scale=sigma_pior))
        for p in data.get_players()
    }

    k = 1.0
    if INCLUDE_K:
        k = torch.exp(
            pyro.sample(
                "log_k",
                dist.Normal(
                    loc=torch.tensor(PRIOR["loc_log_k"]),
                    scale=torch.tensor(PRIOR["scale_log_k"]),
                ),
            )
        )

    for i, m in enumerate(data.get_matches()):
        skill_team_a = sum(skill[p] for p in m.team_a) / len(m.team_a)
        skill_team_b = sum(skill[p] for p in m.team_b) / len(m.team_b)

        logits_A = skill_team_a - skill_team_b
        if INCLUDE_K:
            logits_A *= k

        logits_A *= DEFAULT_K

        if MODEL_GOALS:
            pyro.sample(
                name=f"match_{i+1}",
                fn=dist.Binomial(total_count=m.goals_a + m.goals_b, logits=logits_A),
                obs=torch.tensor(m.goals_a),
            )

        if MODEL_VICTORY:
            pyro.sample(
                name=f"match_{i+1}_winner",
                fn=dist.ContinuousBernoulli(logits=logits_A),
                obs=torch.tensor(
                    1.0
                    if m.goals_a > m.goals_b
                    else (0.5 if m.goals_a == m.goals_b else 0.0)
                ),
            )
