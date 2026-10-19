import pandas as pd, matplotlib.pyplot as plt

d = pd.read_csv("bulletins/Delhi_AQIBulletins.csv", parse_dates=["date"])
s = d.groupby("date")["Index Value"].mean().asfreq("D")
x, y = s.iloc[:-1].values, s.iloc[1:].values
ok = ~(pd.isna(x) | pd.isna(y))

print("correlation, today vs tomorrow:", round(pd.Series(x[ok]).corr(pd.Series(y[ok])), 2))
plt.scatter(x, y, s=5)
plt.plot([0, 500], [0, 500], "r")
plt.xlabel("AQI today"); plt.ylabel("AQI tomorrow")
plt.savefig("plot.png", dpi=150, bbox_inches="tight")
plt.show()