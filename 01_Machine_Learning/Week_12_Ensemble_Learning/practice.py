from sklearn.ensemble import (VotingClassifier,BaggingClassifier,BaggingRegressor,RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor
,AdaBoostClassifier,AdaBoostRegressor,GradientBoostingClassifier,GradientBoostingRegressor,StackingClassifier,StackingRegressor)
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor


# Hard Voting
lr = LogisticRegression()

dt = DecisionTreeClassifier()

svm = SVC()

voting = VotingClassifier(

    estimators=[

        ("lr", lr),

        ("dt", dt),

        ("svm", svm)

    ],

    voting="hard"

)
# voting.fit(X_train, y_train)
# pred = voting.predict(X_test)

# Soft Voting
svm = SVC(
    probability=True
)
voting = VotingClassifier(

    estimators=[

        ("lr", lr),

        ("dt", dt),

        ("svm", svm)

    ],

    voting="soft"

)

# Bagging Classifier

tree = DecisionTreeClassifier(random_state=42)

bagging = BaggingClassifier(

    estimator=tree,

    oob_score=True,

    n_estimators=100,

    random_state=42

)
# bagging.fit(X_train, y_train)
# pred = bagging.predict(X_test)
# print(bagging.oob_score_)

# Bagging Regression


tree = DecisionTreeRegressor()

model = BaggingRegressor(

    estimator=tree,

    n_estimators=100,

    random_state=42

)

# model.fit(X_train, y_train)

# Random Forest

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42

)
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(accuracy_score(y_test,pred))

# Random Forest Regression
model = RandomForestRegressor(

    n_estimators=100,

    random_state=42

)

# model.fit(X_train,y_train)

# Extra Tree Classifier
model = ExtraTreesClassifier(

    n_estimators=100,

    random_state=42

)
# Extra Trees Regressor
model = ExtraTreesRegressor(

    n_estimators=100,

    random_state=42

)

# AdaBoost Classifier
model = ExtraTreesRegressor(

    n_estimators=100,

    random_state=42

)

# AdaBoost Regreesion
model = AdaBoostRegressor(

    n_estimators=100,

    random_state=42

)
# Gradient boosting Classifier
model = GradientBoostingClassifier(

    n_estimators=100,

    learning_rate=0.1,

    random_state=42

)
# Gradient Boosting Regression
model = GradientBoostingRegressor(

    n_estimators=100,

    learning_rate=0.1,

    random_state=42

)

# XGBoost Classifier
model = XGBClassifier(

    n_estimators=100,

    learning_rate=0.1,

    random_state=42

)

# XGBoost Regression
model = XGBRegressor(

    n_estimators=100,

    learning_rate=0.1,

    random_state=42

)

# LightGBM Classifier
model = LGBMClassifier(

    n_estimators=100,

    learning_rate=0.1,

    random_state=42

)

# LightGBM Regression
model=LGBMRegressor(

    n_estimators=100,

    learning_rate=0.1

)

# CatBoost Classifier
model = CatBoostClassifier(

    iterations=100,

    learning_rate=0.1,

    verbose=False

)

# CatBoost Regression
model=CatBoostRegressor(

iterations=100,

verbose=False

)

# Stacking Classifier
estimators = [

("rf", RandomForestClassifier()),

("svm", SVC(probability=True))

]
model = StackingClassifier(

estimators=estimators,

final_estimator=LogisticRegression()

)

# Stacking Regression
estimators = [

("rf", RandomForestRegressor())

]

model = StackingRegressor(

estimators=estimators,

final_estimator=LinearRegression()

)