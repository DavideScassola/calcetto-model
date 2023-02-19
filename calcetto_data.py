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
    to_set = lambda c: c.str.split(", ").apply(set)
    df[TEAM_A] = to_set(df[TEAM_A])
    df[TEAM_B] = to_set(df[TEAM_B])
    useful_columns = [TEAM_A, TEAM_B, GOALS_A, GOALS_B, WINNER]
    return df[useful_columns]


@dataclass
class Match:
    team_a: set
    team_b: set
    goals_a: int
    goals_b: int

    def total_goals(self) -> int:
        return self.goals_a + self.goals_b


class CalcettoData:
    def __init__(self, file: str) -> None:
        self.df = from_notion_csv(file)
        self.__build_players_list(self.df)
        self.__build_matches_list(self.df)

    def __build_players_list(self, df: pd.DataFrame):
        self.players = sorted(list(set().union(*df[TEAM_A], *df[TEAM_B])))

    def __build_matches_list(self, df: pd.DataFrame):
        def from_df_row(row: dict):
            return Match(
                team_a=row[TEAM_A],
                team_b=row[TEAM_B],
                goals_a=row[GOALS_A],
                goals_b=row[GOALS_B],
            )

        self.matches = [from_df_row(r) for r in df.iloc]

    def get_players(self) -> list:
        return self.players

    def get_matches(self) -> List[Match]:
        return self.matches
