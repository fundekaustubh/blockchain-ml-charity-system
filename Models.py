import pandas as pd
import time
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
import sklearn.metrics as sm
from numpy import absolute


class Models:
    def __init__(self, data):
        self.data = data
        X = data.loc[:, ~data.columns.isin(["name", "score"])]
        y = data.loc[:, ["score"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models_accuracy = dict()
        self.models_time = dict()
        self.models_summary = dict()
        self.ElasticNetRegression()
        self.LinearRegression()
        self.SupportVectorRegression()
        self.RandomForestRegression()
        self.PolynomialRegression()
        self.Summarize()

    def find_validation_df(self, y_test, y_pred):
        data = [[test, pred] for index, (test, pred) in enumerate(zip(y_test, y_pred))]
        validation_df = pd.DataFrame(data, columns=["Testing Value", "Predicted Value"])
        return validation_df

    def ElasticNetRegression(self):
        model_en = ElasticNet(alpha=1.0, l1_ratio=0.5)
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(
            model_en,
            self.X_train,
            self.y_train,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
        )
        scores = absolute(scores)
        start = time.time()
        model_en.fit(self.X_train, self.y_train)
        end = time.time()
        exec_time = end - start
        self.models_time["Elastic Net"] = exec_time * 10 ** 3
        y_pred = model_en.predict(self.X_test)
        acc = sm.r2_score(self.y_test, y_pred)
        self.models_accuracy["Elastic Net"] = (model_en, acc)

    def LinearRegression(self):
        model_lr = LinearRegression()
        start = time.time()
        model_lr.fit(self.X_train, self.y_train)
        end = time.time()
        exec_time = end - start
        self.models_time["Linear"] = exec_time * 10 ** 3
        y_pred = model_lr.predict(self.X_test).flatten()
        acc = sm.r2_score(self.y_test, y_pred)
        self.models_accuracy["Linear"] = (model_lr, acc)

    def SupportVectorRegression(self):
        model_svr = SVR(kernel="rbf")
        start = time.time()
        model_svr.fit(self.X_train, self.y_train)
        end = time.time()
        exec_time = end - start
        self.models_time["Support Vector"] = exec_time * 10 ** 3
        y_pred = model_svr.predict(self.X_test)
        acc = sm.r2_score(self.y_test, y_pred)
        self.models_accuracy["Support Vector"] = (model_svr, acc)

    def DecisionTreeRegression(self):
        model_dtr = DecisionTreeRegressor(random_state=0)
        start = time.time()
        model_dtr.fit(self.X_train, self.y_train)
        end = time.time()
        exec_time = end - start
        self.models_time["Decision Tree"] = exec_time * 10 ** 3
        y_pred = model_dtr.predict(self.X_test)
        acc = sm.r2_score(self.y_test, y_pred)
        self.models_accuracy["Decision Tree"] = (model_dtr, acc)

    def RandomForestRegression(self):
        model_rfr = RandomForestRegressor(n_estimators=100, random_state=0)
        start = time.time()
        model_rfr.fit(self.X_train, self.y_train)
        end = time.time()
        exec_time = end - start
        self.models_time["Random Forest"] = exec_time * 10 ** 3
        y_pred = model_rfr.predict(self.X_test)
        acc = sm.r2_score(self.y_test, y_pred)
        self.models_accuracy["Random Forest"] = (model_rfr, acc)

    def LassoRegresion(self):
        model_lasso = Lasso(alpha=1.0)
        start = time.time()
        model_lasso.fit(self.X_train, self.y_train)
        end = time.time()
        exec_time = end - start
        self.models_time["Lasso"] = exec_time * 10 ** 3
        y_pred = model_lasso.predict(self.X_test)
        acc = sm.r2_score(self.y_test, y_pred)
        self.models_accuracy["Lasso"] = (model_lasso, acc)

    def PolynomialRegression(self):
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(self.X_train)
        X_test_poly = poly.fit_transform(self.X_test)
        model_poly = LinearRegression()
        start = time.time()
        model_poly.fit(X_train_poly, self.y_train)
        end = time.time()
        exec_time = end - start
        self.models_time["Polynomial"] = exec_time * 10 ** 3
        y_pred = model_poly.predict(X_test_poly).flatten()
        acc = sm.r2_score(self.y_test, y_pred)
        self.models_accuracy["Polynomial"] = (model_poly, acc)

    def Summarize(self):
        models_summary = dict()
        for key in self.models_accuracy.keys():
            models_summary[key] = dict()
            models_summary[key][
                "Accuracy"
            ] = f"{round(self.models_accuracy[key][1]*100, 2)}%"
            models_summary[key]["Training Time"] = self.models_time[key]
            models_summary[key]["Model Factor"] = (
                self.models_accuracy[key][1] * 100 / self.models_time[key]
            )
        # models_summary.append(new_model)
        models_summary_df = pd.DataFrame.from_dict(models_summary).transpose()
        self.models_summary = models_summary_df