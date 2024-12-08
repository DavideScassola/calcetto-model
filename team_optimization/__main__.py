import argparse
import random
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
import yaml


def get_indexes(subset: list, all: list):
    return [all.index(x) for x in subset]


def get_mean_cov(players: list | pd.Series) -> tuple:
    if isinstance(players, list):
        posterior = np.load("results/posterior.npy", allow_pickle=True).item()
        i = get_indexes(players, posterior["players"])
        mean = posterior["mean"][i]
        std = posterior["std"][i]
        corr = posterior["corr"][i][:, i]
        corr = corr + corr.T - np.diag(np.diag(corr))
        cov = corr * std.reshape(-1, 1) * std.reshape(1, -1)

        s = pd.Series(mean, index=players)
        s -= s.mean()
        print(s.sort_values().round(2))

    else:
        mean = players.values
        cov = np.zeros((len(players), len(players)))
        print(players.sort_values().round(2))

    return mean, cov


def get_diff_mean_var(players: list | pd.Series, partitions: np.ndarray) -> tuple:
    mean, cov = get_mean_cov(players)
    a = np.where(partitions, 1, -1)
    mean_diff = a @ mean
    var_diff = np.array([a[i] @ cov @ a[i].T for i in range(len(a))])  # TODO: vectorize

    return mean_diff, var_diff


def unknown_result_criterion(
    players: list | pd.Series, partitions: np.ndarray
) -> np.ndarray:
    mean_diff, var_diff = get_diff_mean_var(players, partitions)
    return -mean_diff / var_diff**0.5


def extreme_result_criterion(
    players: list | pd.Series, partitions: np.ndarray
) -> np.ndarray:
    mean_diff, var_diff = get_diff_mean_var(players, partitions)
    return -(mean_diff**2 + var_diff)


def balance_criterion(players: list | pd.Series, partitions: np.ndarray) -> np.ndarray:
    mean_diff, _ = get_diff_mean_var(players, partitions)
    return -(mean_diff**2)


def similar_couples_criterion(
    players: list | pd.Series, partitions: np.ndarray
) -> np.ndarray:
    """
    Returns the sum of the squared errors between the skill of the ith best player of team A and
    the skill of the ith best player of team B.
    """

    # Check this code
    mean, _ = get_mean_cov(players)
    mean_mod = mean - mean.min() + 1
    a = np.where(partitions, 1, -1)
    a = a * mean_mod.reshape(1, len(mean))
    team1 = np.sort(a[a > 0].reshape(len(a), -1), axis=-1)
    team2 = np.sort(-a[a < 0].reshape(len(a), -1), axis=-1)

    return -np.sum((team1 - team2) ** 2, axis=-1)


def similar_couples_and_balance_criterion(
    players: list | pd.Series, partitions: np.ndarray, balance_weight=0.5
) -> np.ndarray:
    return (1 - balance_weight) * similar_couples_criterion(
        players, partitions
    ) + balance_weight * balance_criterion(players, partitions)


def consensus_criterion(
    players: list | pd.Series, partitions: np.ndarray
) -> np.ndarray:
    preferences = get_preferences(players)
    satisfaction = partitions @ preferences.T

    satisfaction = np.where(
        partitions, satisfaction, np.sum(preferences, axis=1) - satisfaction
    )
    return np.min(satisfaction, axis=1)


def load_players(path: str):
    with open(path, "r") as file:
        players = yaml.safe_load(file)
    return players if isinstance(players, list) else pd.Series(players)


def get_preferences(players: list) -> np.ndarray:
    d = {
        "A": [0.1, -1000, 0.01, 0.2, -0.1, 0.3, 0.05, 0.0, 0, -0.05],
        "B": [0.1, 0, 0.02, 0.02, 0.0, 0.6, 0.04, 0.03, -1000, 0],
    }

    preferences = np.full((len(players), len(players)), 1 / len(players))

    for c in d:
        preferences[players.index(c)] = np.array(d[c])

    return preferences


def load_criterion(name: str, players: list) -> Callable:
    return partial(eval(name), players)


def generate_partitions(n: int) -> np.ndarray:
    assert n % 2 == 0, "Number of players must be even"
    d = np.arange(2**n)
    p = (((d[:, None] & (1 << np.arange(n)))) > 0).astype(int)  # all partitions
    p = p[p.sum(axis=1) == n // 2]  # all partitions with equal size
    p = p[p[:, 0] == 1]  # removes redundant half
    return p == 1


def optimize_team(players: list, criterion: Callable, randomize=True) -> pd.Series:
    partitions = generate_partitions(len(players))
    randomization = (
        np.random.randn(len(partitions)) * 1e-6
        if randomize
        else np.zeros(len(partitions))
    )
    optimal_partition_index = np.argsort(criterion(partitions) + randomization)[
        -1
    ]  # optimal_partition_index = np.argmax(criterion(partitions))

    optimal_partition = partitions[optimal_partition_index]

    return pd.Series(
        optimal_partition, index=players if isinstance(players, list) else players.index
    )


def show_results(players: list | pd.Series, optimal_teams: dict | pd.Series) -> None:
    if isinstance(players, list):
        mean = get_mean_cov(optimal_teams.index.to_list())[0].round(2)
    else:
        mean = players.round(2)
    df = pd.DataFrame(
        {"team": optimal_teams.values, "mean": mean}, index=optimal_teams.index
    )
    df["team"] = df["team"].map({True: "A", False: "B"})
    df.sort_values(["team", "mean"], ascending=False, inplace=True)

    teamA = df[df["team"] == "A"].index
    teamB = df[df["team"] == "B"].index

    teams = [teamA, teamB]
    random.shuffle(teams)

    separator = ", "
    teams_strings = [t.str.cat(sep=separator) for t in teams]

    print("\n```")
    print(f"Bianchi: {teams_strings[0]}")
    print(f"Neri:    {teams_strings[1]}")
    print("```\n")

    print(f"Mean Bianchi: ", df.loc[teams[0]]["mean"].mean())
    print(f"Mean Neri: ", df.loc[teams[1]]["mean"].mean())


def run(players_file: str, criterion_name: str):
    players = load_players(players_file)
    criterion = load_criterion(criterion_name, players)
    optimal_teams = optimize_team(players, criterion)
    show_results(players, optimal_teams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Team Optimization")
    parser.add_argument(
        "--players",
        type=str,
        required=False,
        help="Path to the .json file listing the players",
        default="team_optimization/players.yaml",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        required=False,
        help="Name of the criterion for building teams",
        default="similar_couples_and_balance_criterion",
    )

    args = parser.parse_args()
    run(players_file=args.players, criterion_name=args.criterion)
