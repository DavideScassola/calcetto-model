import pyro.distributions as dist
import pyro
import torch
from calcetto_data import CalcettoData
from pyro.infer.autoguide.guides import AutoMultivariateNormal
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd

# PRIOR = {"mu_log2_skill": -0.29, "sigma_log2_skill": 0.1}
PRIOR = {"mu_log2_skill": 0.0, "sigma_log2_skill": 0.5}


def model(data: CalcettoData):
    mu_pior = torch.tensor(PRIOR["mu_log2_skill"])
    sigma_pior = torch.tensor(PRIOR["sigma_log2_skill"])

    # latent_log2_skill[data.reference_player] = 1.0  # TODO: maybe I have to put an array
    latent_log2_skill = {
        p: pyro.sample(
            f"latent_log2_skill_{p}",
            dist.Normal(loc=mu_pior, scale=sigma_pior)
            if p != "Umberto L"
            else dist.Normal(torch.tensor(0.0), torch.tensor(1e-4)),
        )
        for p in data.get_players()
        # for p in data.players(include_reference_player=False)
    }

    latent_skill = {name: torch.exp2(ls) for name, ls in latent_log2_skill.items()}

    for i, m in enumerate(data.get_matches()):
        latent_skill_team_a = sum(latent_skill[p] for p in m.team_a)
        latent_skill_team_b = sum(latent_skill[p] for p in m.team_b)
        prob_A = latent_skill_team_a / (latent_skill_team_a + latent_skill_team_b)
        pyro.sample(
            name=f"match_{i+1}",
            fn=dist.Binomial(total_count=m.goals_a + m.goals_b, probs=prob_A),
            obs=torch.tensor(m.goals_a),
        )


"""    
def guide(data: CalcettoData):
    # register the two variational parameters with Pyro.
    l = len(data.get_players())
    mean_q = pyro.param("mean_q", torch.zeros(l))
    cov_q = pyro.param("cov_q", torch.eye(l))
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    pyro.sample("latent_fairness", dist.MultivariateNormal(mean_q, cov_q))
    
    dist.MultivariateNormal()
"""


data = CalcettoData("dataset/log.csv")

guide = AutoMultivariateNormal(model=model)
n_steps = 1000


# setup the optimizer
adam_params = {"lr": 0.001}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(
    model, guide, optimizer, loss=Trace_ELBO(vectorize_particles=True, num_particles=3)
)

losses = np.zeros(n_steps)

# do gradient steps
for i in range(n_steps):
    losses[i] = svi.step(data)
    if i % 100 == 0:
        plt.plot(losses[:i])
        plt.savefig("loss.png")
        plt.close()

        mean = pyro.get_param_store()["AutoMultivariateNormal.loc"]
        std = pyro.get_param_store()["AutoMultivariateNormal.scale"]
        corr = pyro.get_param_store()["AutoMultivariateNormal.scale_tril"]

        stats = {
            p: {"median": np.exp2(mean[i].item()), "std": std[i].item()}
            for i, p in enumerate(data.get_players())
        }

        print("stats: ")
        print(json.dumps(stats, indent=4))

        print(f"{corr=}: ")

        players = np.array(data.get_players())
        medians = np.exp2(mean.detach().numpy())
        order = np.argsort(-medians)
        y_pos = np.arange(len(mean))

        plt.barh(y=y_pos, width=medians[order])
        plt.yticks(y_pos, labels=players[order])
        plt.savefig("stats.png")
        plt.close()
