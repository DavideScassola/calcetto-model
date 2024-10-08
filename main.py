import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import seaborn as sns
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide.guides import AutoMultivariateNormal
from pyro.optim import SGD, RAdam
from scipy.stats import lognorm, norm

from src.calcetto_data import CalcettoData
from src.calcetto_model import INCLUDE_K, model
from src.util import store_json

DATASET = (
    "dataset/log.csv"
    if os.path.exists("dataset/log.csv")
    else "dataset_example/log.csv"
)
RESULTS_FOLDER = "results/"
IMAGE_TYPE = "png"
SHOW_EXP = False

plt.rcParams.update({"figure.autolayout": True})


def loss_plot(losses, *, path, log=False):
    y = losses if not log else np.log2(losses)
    plt.plot(y)
    plt.savefig(RESULTS_FOLDER + f"{'log2_' if log else ''}loss.{IMAGE_TYPE}")
    plt.close()


def correlations_plot(corr, data, mp_minimum=4):
    players = np.array(data.get_players())
    corr_for_plot = np.round(corr.detach().numpy(), 2)
    corr_for_plot += corr_for_plot.T
    np.fill_diagonal(corr_for_plot, np.nan)
    # plt.tight_layout()
    plt.figure(figsize=(12, 10))

    p_mask = data.get_player_statistics()["MP"] >= mp_minimum
    p_mask["PRIOR"] = False

    df = pd.DataFrame(corr_for_plot, index=players, columns=players).loc[p_mask, p_mask]
    ax = sns.heatmap(
        df,
        annot=False,
        cmap=sns.color_palette("coolwarm", as_cmap=True),
        alpha=1.0,
        vmin=-1.0,
        vmax=1.0,
    )
    plt.savefig(RESULTS_FOLDER + f"corr.{IMAGE_TYPE}")
    plt.close()


def marginal_skills_plot(data, mean, std, f, mp_minimum=3):
    players = np.array(data.get_players())
    medians = f(mean.detach().numpy())
    quantiles_05 = f(norm(mean.detach(), std.detach()).ppf(0.05))
    quantiles_95 = f(norm(mean.detach(), std.detach()).ppf(0.95))

    # aestetic_rescale = 80 / np.max(medians)
    aestetic_rescale = 1

    medians *= aestetic_rescale
    quantiles_05 *= aestetic_rescale
    quantiles_95 *= aestetic_rescale

    order = np.argsort(-medians)

    skill_df = pd.DataFrame(
        {
            "q95": quantiles_95[order],
            "q05": quantiles_05[order],
            "medians": medians[order],
        },
        index=players[order],
    )

    mp = data.get_player_statistics()["MP"] >= mp_minimum
    mp["PRIOR"] = True
    mask = [p for p in skill_df.index if mp[p]]
    skill_df = skill_df.loc[mask]

    y_pos = np.arange(len(skill_df))

    plt.barh(y=y_pos, width=skill_df["q95"], alpha=0.8, color="red")
    plt.barh(y=y_pos, width=skill_df["medians"], alpha=0.8, color="blue")
    plt.barh(y=y_pos, width=skill_df["q05"], alpha=0.8, color="#FFFFFF")

    plt.yticks(y_pos, labels=skill_df.index)
    plt.xlim(np.min(quantiles_05) / 1.01, np.max(quantiles_95) * 1.01)
    plt.grid(alpha=0.8)
    # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
    plt.savefig(RESULTS_FOLDER + f"stats.{IMAGE_TYPE}")
    plt.close()

def store_posterior(players, mean, std, corr):
    posterior = {'players': players,
                 'mean': mean.detach().numpy(),
                 'std': std.detach().numpy(),
                 'corr': corr.detach().numpy()}
    np.save(RESULTS_FOLDER + 'posterior.npy', posterior)

if __name__ == "__main__":
    data = CalcettoData(DATASET)

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # creating csv of players statistics
    data.to_markdown(telegram=True, path=RESULTS_FOLDER)

    guide = AutoMultivariateNormal(model=model)

    # setup the optimizer
    opt_params = {"lr": 0.002}
    optimizer = RAdam(opt_params)
    n_steps = 10000
    num_particles = 10

    # setup the inference algorithm
    svi = SVI(
        model,
        guide,
        optimizer,
        loss=Trace_ELBO(vectorize_particles=True, num_particles=num_particles),
    )

    losses = np.zeros(n_steps)

    # do gradient steps
    for i in range(n_steps):
        losses[i] = svi.step(data)
        print("ELBO: ", losses[i])

        if i % 100 == 0:
            loss_plot(
                losses=losses[:i], path=RESULTS_FOLDER + f"loss.{IMAGE_TYPE}", log=False
            )

            skills_slice = slice(None, -1) if INCLUDE_K else slice(None, None)

            mean = pyro.get_param_store()["AutoMultivariateNormal.loc"][skills_slice]
            std = pyro.get_param_store()["AutoMultivariateNormal.scale"][skills_slice]
            corr = pyro.get_param_store()["AutoMultivariateNormal.scale_tril"][
                skills_slice, skills_slice
            ]

            if INCLUDE_K:
                k_mean = pyro.get_param_store()["AutoMultivariateNormal.loc"][-1]
                k_std = pyro.get_param_store()["AutoMultivariateNormal.scale"][-1]

            f = np.exp if SHOW_EXP else lambda x: x

            stats = {
                p: {"median": f(mean[i].item()), "std": std[i].item()}
                for i, p in enumerate(data.get_players())
            }

            if INCLUDE_K:
                stats["k"] = {
                    "median(k)": np.exp(k_mean.item()),
                    "std(log_k)": k_std.item(),
                }
                store_json(stats["k"], file=RESULTS_FOLDER + "k.json")

            marginal_skills_plot(data, mean, std, f)
            
            store_posterior(data.get_players(), mean, std, corr)

            if i % 200 == 0:
                correlations_plot(corr, data)
