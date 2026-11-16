import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "Storage": [128, 256, 512, 1024, 2048],
    "Memory": [8, 16, 32, 64, 128],
    "Price": [100, 200, 400, 800, 1600],
}
df = pd.DataFrame(data)
x = df[["Storage", "Memory"]]
y = df["Price"]
model = LinearRegression()
model.fit(x, y)
predicted_price = model.predict(pd.DataFrame({"Storage": [512], "Memory": [128]}))
df["Predicted_Price"] = model.predict(x)
ax = df.plot(
    kind="line", x="Storage", y="Price", color="blue", label="Actual Price", linewidth=8
)
df.plot(
    kind="line",
    x="Storage",
    y="Predicted_Price",
    color="red",
    label="Predicted Price",
    ax=ax,
)
plt.title("storage and vs price")
plt.xlabel("Storage (GB)")
plt.ylabel("Price ($)")
plt.grid(True)
plt.show()
