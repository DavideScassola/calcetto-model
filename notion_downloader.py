# Initialisation
import json
import requests
import pandas as pd

DATABASE_URL = "https://www.notion.so/6a006d0f698a476eb4bcd18d22e653b5?v=822e48882f7e4421be4c66d04bdb86b0"
NOTION_TOKEN_FILE = "notion_token.txt"
CSV_TARGET = "dataset/log.csv"

def get_database_id(url: str) -> str:
    return url.split("/")[-1].split("?")[0]


# Response a Database
def responseDatabase(databaseID, headers):
    readUrl = f"https://api.notion.com/v1/databases/{databaseID}"
    res = requests.request("GET", readUrl, headers=headers)
    print(res.status_code)


def readDatabase(databaseID, headers, to_json=False):
    readUrl = f"https://api.notion.com/v1/databases/{databaseID}/query"
    res = requests.request("POST", readUrl, headers=headers)
    data = res.json()
    #print(res.status_code)

    if to_json:
        with open("./full-properties.json", "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False)
        return data
    
    return res.json()

def notion_json_to_df(d: dict) -> pd.DataFrame:
    
    def parse_team(team: dict):
        names = [d['name'] for d in team['multi_select']]
        return ", ".join(names)
    
    def property_to_entry(p: dict) -> dict:
        return {"Data": p['Data']['date']['start'],
                "Campo": p['Campo']['multi_select'][0]['name'],
                "Squadra A": parse_team(p["Squadra A"]),
                "Squadra B": parse_team(p["Squadra B"]),
                "Gol A": p["Gol A"]["number"],
                "Gol B": p["Gol B"]["number"],
                "risultato": p["risultato"]["select"]["name"],
                "note": p["note"]["rich_text"][0]["plain_text"] if len(p["note"]["rich_text"]) > 0 else ""}
        
    df = pd.DataFrame([property_to_entry(p['properties']) for p in d['results']])
    return df.sort_values(by="Data")
    
if __name__=="__main__":
    
    databaseID = get_database_id(DATABASE_URL)
    token = next(open(NOTION_TOKEN_FILE)).strip()
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "Notion-Version": "2022-02-22",
    }

    data = readDatabase(databaseID, headers)
    df = notion_json_to_df(data)
    df.to_csv(CSV_TARGET, index=False)
    