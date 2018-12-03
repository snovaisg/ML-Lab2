from utils.utils import load_data
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

plt.rcParams.update({'figure.autolayout': True})

df = load_data()

print("\n", df.head(n=5))
print("\ntype\n", df["type"].value_counts())

print("\n", df.loc[df['type'] == "nuclear explosion"])

df.fault_names.value_counts().plot("bar")
plt.title("Histogram of quakes by fault lines")
plt.savefig("../media/kmeans/hist_faults.png")
plt.show()

