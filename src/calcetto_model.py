import numpy as np
import pyro
import pyro.distributions as dist
import torch

from .calcetto_data import CalcettoData

PRIOR = {
    "mu_skill": 6.0,
    "sigma_skill": 1.0,
    "loc_log_k": 3.0,
    "scale_log_k": 2.0,
}

INCLUDE_K = False


def model(data: CalcettoData):
    mu_pior = torch.tensor(PRIOR["mu_skill"])
    sigma_pior = torch.tensor(PRIOR["sigma_skill"])

    skill = {
        p: pyro.sample(f"skill_{p}", dist.Normal(loc=mu_pior, scale=sigma_pior))
        for p in data.get_players()
    }

    k = 1.0
    if INCLUDE_K:
        k = pyro.sample(
            "k",
            dist.LogNormal(
                loc=torch.tensor(PRIOR["loc_log_k"]),
                scale=torch.tensor(PRIOR["scale_log_k"]),
            ),
        )

    for i, m in enumerate(data.get_matches()):
        skill_team_a = sum(skill[p] for p in m.team_a) / len(m.team_a)
        skill_team_b = sum(skill[p] for p in m.team_b) / len(m.team_b)

        logits_A = skill_team_a - skill_team_b
        if INCLUDE_K:
            logits_A *= k

        pyro.sample(
            name=f"match_{i+1}",
            fn=dist.Binomial(total_count=m.goals_a + m.goals_b, logits=logits_A),
            obs=torch.tensor(m.goals_a),
        )
