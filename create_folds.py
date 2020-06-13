import os
import pandas as pd
from sklearn import model_selection

if __name__ = "__main__":
    input_path = "/home/Jared/Downloads/PyTorch Skin Detection/input/"
    df = pd.read.csv(os.path.join(input_path, "train.csv"))
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedFold(n_splits=5)
    for fold_, (_,_) in enumerate(kf.split(X=df, y=y)):
        df.loc[:, "kfold"] = fold
    df.to_csv(os.path.join(input_path, "train_folds.csv"), index=False)