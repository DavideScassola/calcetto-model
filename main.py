import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import seaborn as sns
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide.guides import AutoMultivariateNormal
from pyro.optim import Adam, SGD
from tqdm import tqdm
from scipy.stats import norm

from calcetto_data import CalcettoData


# PRIOR = {"mu_log2_skill": -0.29, "sigma_log2_skill": 0.1}
PRIOR = {
    "mu_log2_skill": 0.0,
    "sigma_log2_skill": 1,
    "mu_log2_k": 0.0,
    "sigma_log2_k": 3,
}


def model(data: CalcettoData):
    mu_pior = torch.tensor(PRIOR["mu_log2_skill"])
    sigma_pior = torch.tensor(PRIOR["sigma_log2_skill"])

    # latent_log2_skill[data.reference_player] = 1.0  # TODO: maybe I have to put an array
    latent_skill = {
        p: pyro.sample(
            f"latent_log2_skill_{p}",
            dist.LogNormal(loc=mu_pior, scale=sigma_pior)
            if p != "Umberto L"
            else dist.LogNormal(torch.tensor(0.0), torch.tensor(1e-4)),
        )
        for p in data.get_players()
        # for p in data.players(include_reference_player=False)
    }

    k = pyro.sample(
        "k",
        dist.LogNormal(
            loc=torch.tensor(PRIOR["mu_log2_k"]),
            scale=torch.tensor(PRIOR["sigma_log2_k"]),
        ),
    )

    for i, m in enumerate(data.get_matches()):
        latent_skill_team_a = sum(latent_skill[p] for p in m.team_a)
        latent_skill_team_b = sum(latent_skill[p] for p in m.team_b)

        skill_ratio = latent_skill_team_a / latent_skill_team_b
        colder_skill_ratio = torch.pow(skill_ratio, k)
        prob_A = colder_skill_ratio / (colder_skill_ratio + 1)

        # prob_A = latent_skill_team_a / (latent_skill_team_a + latent_skill_team_b)

        # assert prob_A < 1, f"{latent_skill}"
        pyro.sample(
            name=f"match_{i+1}",
            fn=dist.Binomial(total_count=m.goals_a + m.goals_b, probs=prob_A),
            obs=torch.tensor(m.goals_a),
        )


data = CalcettoData("dataset/log.csv")

guide = AutoMultivariateNormal(model=model)
n_steps = 10000


# setup the optimizer
opt_params = {"lr": 0.003}
optimizer = Adam(opt_params)

# setup the inference algorithm
svi = SVI(
    model,
    guide,
    optimizer,
    loss=Trace_ELBO(vectorize_particles=True, num_particles=100),
)

losses = np.zeros(n_steps)

# do gradient steps
for i in range(n_steps):
    losses[i] = svi.step(data)
    print(losses[i])

    if i % 100 == 0:
        plt.plot(np.log2(losses[:i]))
        plt.savefig("log2_loss.png")
        plt.close()

        mean = pyro.get_param_store()["AutoMultivariateNormal.loc"][:-1]
        std = pyro.get_param_store()["AutoMultivariateNormal.scale"][:-1]
        corr = pyro.get_param_store()["AutoMultivariateNormal.scale_tril"][:-1, :-1]

        k_mean = pyro.get_param_store()["AutoMultivariateNormal.loc"][-1]
        k_std = pyro.get_param_store()["AutoMultivariateNormal.scale"][-1]

        stats = {
            p: {"median": np.exp2(mean[i].item()), "std": std[i].item()}
            for i, p in enumerate(data.get_players())
        }

        stats["k"] = {"median": np.exp2(k_mean.item()), "std": k_std.item()}

        print("stats: ")
        print(json.dumps(stats, indent=4))

        print(f"{corr=}: ")

        players = np.array(data.get_players())
        medians = np.exp2(mean.detach().numpy())
        quantiles_10 = np.exp2(norm(mean.detach(), std.detach()).ppf(0.1))
        quantiles_90 = np.exp2(norm(mean.detach(), std.detach()).ppf(0.9))

        aestetic_rescale = 80 / np.max(medians)

        medians *= aestetic_rescale
        quantiles_10 *= aestetic_rescale
        quantiles_90 *= aestetic_rescale

        order = np.argsort(-medians)
        y_pos = np.arange(len(mean))

        plt.tight_layout()
        plt.figure(figsize=(8, 6))

        plt.barh(y=y_pos, width=quantiles_90[order], alpha=0.8, color="red")
        plt.barh(y=y_pos, width=medians[order], alpha=0.8, color="blue")
        plt.barh(y=y_pos, width=quantiles_10[order], alpha=0.8, color="#FFFFFF")

        plt.yticks(y_pos, labels=players[order])
        plt.xlim(np.min(quantiles_10) / 1.05, np.max(quantiles_90) * 1.05)
        plt.savefig("stats.png")
        plt.close()

        if i % 200 == 0:
            corr_for_plot = np.round(corr.detach().numpy(), 2)
            corr_for_plot += corr_for_plot.T
            np.fill_diagonal(corr_for_plot, np.nan)
            plt.tight_layout()
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                pd.DataFrame(corr_for_plot, index=players, columns=players),
                annot=False,
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                alpha=0.8,
            )
            plt.savefig("corr.png")
            plt.close()
