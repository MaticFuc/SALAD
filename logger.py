import pandas as pd
from os import path

def log(file_path, results, filename="results.csv"):
    df = pd.DataFrame(data=results)
    csv_mode = "a"
    if not path.isfile(f"{file_path}/{filename}"):
        csv_mode = "w"
    print(f"{file_path}/{filename}")
    df.to_csv(
        f"{file_path}/{filename}",
        mode=csv_mode,
        index=False,
        header=csv_mode == "w",
    )
