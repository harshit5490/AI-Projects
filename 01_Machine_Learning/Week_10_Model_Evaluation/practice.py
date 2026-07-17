from sklearn.model_selection import cross_val_score,StratifiedKFold,KFold,learning_curve,validation_curve,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

data = load_breast_cancer()

X = data.data
y = data.target

model = LogisticRegression(max_iter=1000)

# scores = cross_val_score(
#     model,
#     # X,
#     # y,
#     cv=5 # for run 5 time on different subsets of data
# )

# print(scores)

# kf = KFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=42
# )

# scores = cross_val_score(
#     model,
#     # X,
#     # y,
#     cv=kf # for run 5 time on different subsets of data
# )

# skf = StratifiedKFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=42
# )

# scores = cross_val_score(
#     model,
#     # X,
#     # y,
#     cv=skf # for run 5 time on different subsets of data
# )

# model = LogisticRegression(max_iter=1000)

# train_sizes, train_scores, validation_scores = learning_curve(
#     model,
#     X,
#     y,
#     cv=5,
#     scoring="accuracy",
#     train_sizes=np.linspace(0.1,1.0,5)
# )

# train_mean = train_scores.mean(axis=1)
# validation_mean = validation_scores.mean(axis=1)

# plt.plot(train_sizes, train_mean, label="Training Score")
# plt.plot(train_sizes, validation_mean, label="Validation Score")

# plt.xlabel("Training Samples")
# plt.ylabel("Accuracy")
# plt.title("Learning Curve")
# plt.legend()

# plt.show()

# model = DecisionTreeClassifier(random_state=42)

# train_scores, validation_scores = validation_curve(
#     estimator=model,
#     X=X,
#     y=y,
#     param_name="max_depth",
#     param_range=np.arange(1,11),
#     cv=5,
#     scoring="accuracy"
# )

# train_mean = train_scores.mean(axis=1)
# validation_mean = validation_scores.mean(axis=1)

# depth = np.arange(1,11)
# plt.plot(depth, train_mean, label="Training Score")
# plt.plot(depth, validation_mean, label="Validation Score")

# plt.xlabel("max_depth")
# plt.ylabel("Accuracy")
# plt.title("Validation Curve")

# plt.legend()

# plt.show()

# model = DecisionTreeClassifier(random_state=42)

# param_grid = {
#     "max_depth": [2,4,6,8],
#     "criterion": ["gini","entropy"]
# }

# grid = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     cv=5,
#     scoring="accuracy"
# )

# grid.fit(X, y)
# print(grid.best_params_)
# print(grid.best_score_)
# print(grid.best_estimator_)

model = DecisionTreeClassifier(random_state=42)

param_dist = {
    "max_depth":[2,4,6,8,10],
    "criterion":["gini","entropy"]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=5,
    cv=5,
    scoring="accuracy",
    random_state=42
)

random_search.fit(X, y)
print(random_search.best_params_)
print(random_search.best_score_)