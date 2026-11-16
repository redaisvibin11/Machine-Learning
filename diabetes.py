from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import load_diabetes

data = load_diabetes()
x, y = data.data, data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model = {
    "linear regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(random_state=42),
    "random_forest": RandomForestRegressor(random_state=42),
}

for name, m in model.items():
    m.fit(x_train, y_train)
    preds = m.predict(x_test)
    print(
        f"{name} r2 score: {r2_score(y_test, preds):.3f} | MAE: {mean_absolute_error(y_test, preds):.2f}"
    )
