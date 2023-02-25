import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import seaborn as sns
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide.guides import AutoMultivariateNormal
from pyro.optim import SGD, Adam
from scipy.stats import norm
from tqdm import tqdm

from calcetto_data import CalcettoData
from calcetto_model import model

if __name__ == "__main__":
    data = CalcettoData("dataset/log.csv")

    guide = AutoMultivariateNormal(model=model)

    # setup the optimizer
    opt_params = {"lr": 0.005}
    optimizer = Adam(opt_params)
    n_steps = 10000
    num_particles = 20

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
            quantiles_05 = np.exp2(norm(mean.detach(), std.detach()).ppf(0.05))
            quantiles_95 = np.exp2(norm(mean.detach(), std.detach()).ppf(0.95))

            # aestetic_rescale = 80 / np.max(medians)
            aestetic_rescale = 1

            medians *= aestetic_rescale
            quantiles_05 *= aestetic_rescale
            quantiles_95 *= aestetic_rescale

            order = np.argsort(-medians)
            y_pos = np.arange(len(mean))

            plt.tight_layout()
            plt.figure(figsize=(8, 6))

            plt.barh(y=y_pos, width=quantiles_95[order], alpha=0.8, color="red")
            plt.barh(y=y_pos, width=medians[order], alpha=0.8, color="blue")
            plt.barh(y=y_pos, width=quantiles_05[order], alpha=0.8, color="#FFFFFF")

            plt.yticks(y_pos, labels=players[order])
            plt.xlim(np.min(quantiles_05) / 1.01, np.max(quantiles_95) * 1.01)
            plt.grid(alpha=0.8)
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
