import numpy as np
import pyro
import pyro.distributions as dist
import torch

from calcetto_data import CalcettoData

PRIOR = {
    "mu_log_skill": np.log(70),
    "sigma_log_skill": 0.1,
    "mu_log_k": 3.0,
    "sigma_log_k": 2.0,
}


def model(data: CalcettoData):
    mu_pior = torch.tensor(PRIOR["mu_log_skill"])
    sigma_pior = torch.tensor(PRIOR["sigma_log_skill"])

    latent_skill = {
        p: pyro.sample(
            f"latent_skill_{p}",
            dist.LogNormal(loc=mu_pior, scale=sigma_pior)
            # if p != "Umberto L"
            # else dist.LogNormal(mu_pior, torch.tensor(1e-4)),
        )
        for p in data.get_players()
    }

    k = pyro.sample(
        "k",
        dist.LogNormal(
            loc=torch.tensor(PRIOR["mu_log_k"]),
            scale=torch.tensor(PRIOR["sigma_log_k"]),
        ),
    )

    for i, m in enumerate(data.get_matches()):
        latent_skill_team_a = sum(latent_skill[p] for p in m.team_a)
        latent_skill_team_b = sum(latent_skill[p] for p in m.team_b)

        skill_ratio = latent_skill_team_a / latent_skill_team_b
        colder_skill_ratio = torch.pow(skill_ratio, k)
        prob_A = colder_skill_ratio / (colder_skill_ratio + 1)

        pyro.sample(
            name=f"match_{i+1}",
            fn=dist.Binomial(total_count=m.goals_a + m.goals_b, probs=prob_A),
            obs=torch.tensor(m.goals_a),
        )
