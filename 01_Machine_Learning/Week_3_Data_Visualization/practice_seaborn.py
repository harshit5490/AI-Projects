import seaborn as sns
import matplotlib.pyplot as plt


tips = sns.load_dataset("tips")

# sns.scatterplot(
#     data=tips,
#     x="total_bill",
#     y="tip"
# )

# plt.show()

#Line Chart

plt.figure(figsize=(8,5))

# sns.lineplot(
#     data=tips,
#     x="size",
#     y="total_bill",
#     markers="o",
#     linewidth=3,
#     hue="sex",
#     style="sex",
#     errorbar=None
# )
# plt.show()

# Bar Chart

# order = [
#     "Thur",
#     "Fri",
#     "Sat",
#     "Sun"
# ]
# # for horizontal bar simply swap x and y
# sns.barplot(
#     data=tips,
#     x="day",
#     y="total_bill",
#     hue="sex",
#     errorbar=None,
#     order=order
# )
# plt.show()

# Histogram Chart

# sns.histplot(
#     data=tips,
#     x="total_bill",
#     kde=True,
#     hue="sex",
#     multiple="layer", # There are option in multiple
#     stat="probability" # use to choose y axis
# )
# plt.show()

# Box Plot

# sns.barplot(
#     data=tips,
#     x="day",
#     y="total_bill",
#     hue="sex"
# )
# plt.show()

# Count Plot

# ax = sns.countplot(
#     data=tips,
#     x="day",
#     hue="sex",
#     palette="viridis"
# )

# for container in ax.containers:
#     ax.bar_label(container)

# plt.show()

# Pair Plot

# sns.pairplot(tips,hue="sex",diag_kind="hist",corner=True,
#             #   vars=[""] this is used to display graph of listed columns
#             )
# plt.show()

# Heatmap Plot

iris = sns.load_dataset("iris")

corr = iris.corr(numeric_only=True)
print(corr)
sns.heatmap(corr,annot=True,cmap="coolwarm",vmin=-1,vmax=1,linewidths=1)
plt.show()