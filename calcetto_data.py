from dataclasses import dataclass
from typing import List

import pandas as pd

TEAM_A = "Squadra A"
TEAM_B = "Squadra B"
GOALS_A = "Gol A"
GOALS_B = "Gol B"
RESULT = "risultato"


def from_notion_csv(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    to_set = lambda c: c.str.split(", ").apply(set)
    df[TEAM_A] = to_set(df[TEAM_A])
    df[TEAM_B] = to_set(df[TEAM_B])
    useful_columns = [TEAM_A, TEAM_B, GOALS_A, GOALS_B, RESULT]
    return df[useful_columns]


@dataclass
class Match:
    team_a: set
    team_b: set
    goals_a: int
    goals_b: int
    result: str

    def total_goals(self) -> int:
        return self.goals_a + self.goals_b

    def valid(self) -> int:
        return self.result != "Nulla"


class CalcettoData:
    def __init__(self, file: str) -> None:
        self.df = from_notion_csv(file)
        self.__build_players_list(self.df)
        self.__build_matches_list(self.df)
        self.player_index = {
            name: index for index, name in enumerate(self.get_players())
        }

    def __build_players_list(self, df: pd.DataFrame):
        self.players = sorted(list(set().union(*df[TEAM_A], *df[TEAM_B])))

    def __build_matches_list(self, df: pd.DataFrame):
        def from_df_row(row: dict):
            return Match(
                team_a=row[TEAM_A],
                team_b=row[TEAM_B],
                goals_a=row[GOALS_A],
                goals_b=row[GOALS_B],
                result=row[RESULT],
            )

        self.matches = [from_df_row(r) for r in df.iloc]

    def get_players(self) -> list:
        return self.players + ["PRIOR"]

    def get_matches(self) -> List[Match]:
        return self.matches

    def get_player_statistics(self) -> pd.DataFrame:
        df = pd.DataFrame(index=self.get_players())

        df["GF"] = 0
        df["GA"] = 0
        df["W"] = 0
        df["D"] = 0
        df["L"] = 0

        def assign_result(player: str, result: str, team_a: bool):
            if result == "A":
                if team_a:
                    player["W"] += 1
                else:
                    player["L"] += 1
            if result == "B":
                if team_a:
                    player["L"] += 1
                else:
                    player["W"] += 1
            if result == "D":
                player["D"] += 1

        for m in self.matches:
            if m.valid():
                for p in m.team_a:
                    df.loc[p]["GF"] += m.goals_a
                    df.loc[p]["GA"] += m.goals_b
                    assign_result(df.loc[p], m.result, True)

                for p in m.team_b:
                    df.loc[p]["GF"] += m.goals_b
                    df.loc[p]["GA"] += m.goals_a
                    assign_result(df.loc[p], m.result, False)

        df["MP"] = df["W"] + df["D"] + df["L"]
        df["WR"] = df["W"] / df["MP"]
        df["GD"] = df["GF"] - df["GA"]
        df["GR"] = df["GF"] / df["GA"]

        return round(df.sort_values(by="GR", ascending=False), 2)
