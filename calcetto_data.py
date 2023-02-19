from dataclasses import dataclass
from typing import List
import pandas as pd

TEAM_A = "Squadra A"
TEAM_B = "Squadra B"
GOALS_A = "Gol A"
GOALS_B = "Gol B"
WINNER = "risultato"


def from_notion_csv(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    to_list = lambda c: c.str.split(", ")
    df[TEAM_A] = to_list(df[TEAM_A])
    df[TEAM_B] = to_list(df[TEAM_B])
    useful_columns = [TEAM_A, TEAM_B, GOALS_A, GOALS_B, WINNER]
    return df[useful_columns]


@dataclass
class Match:
    team_a: list
    team_b: list
    goals_a: int
    goals_b: int

    def total_goals(self) -> int:
        return self.goals_a + self.team_b


class CalcettoData:
    def __init__(self, file: str) -> None:
        self.df = from_notion_csv(file)
        self.build_players_list(self.df)
        self.build_matches_list(self.df)

    def build_players_list(self, df: pd.DataFrame):
        self.players = set().union(*df[TEAM_A], *df[TEAM_B])

    def build_matches_list(self, df: pd.DataFrame):
        def from_df_row(row: dict):
            return Match(
                team_a=row[TEAM_A],
                team_b=row[TEAM_B],
                goals_a=row[GOALS_A],
                goals_b=row[TEAM_B],
            )

        self.matches = [from_df_row(r) for r in df.iloc]

    def players(self) -> list:
        return self.players

    def matches(self) -> List[Match]:
        pass
