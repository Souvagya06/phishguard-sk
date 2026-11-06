import glob, pandas as pd
f = glob.glob(r"E:\Hackathons\Phisguard-Z\phish\data\raw\url_features_extract*")[0]
lp = pd.read_csv(f)
print(f, lp.shape); print(lp.columns.tolist()); print(lp.iloc[:, -1].value_counts())