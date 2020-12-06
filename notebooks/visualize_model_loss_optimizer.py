# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Software vs Machine Learning
#
# ![](img/software_vs_ml.png)
#
# ![](img/ml_debt.jpg)

# %% [markdown]
# # Widget to visualize linear regression, error, and loss

# %%
import numpy as np
import altair as alt
import pandas as pd
import ipywidgets


# %%
def f(x, slope: float, bias: float):
    """A simple linear model."""
    return x * slope + bias


# %%
def err2(pred, true):
    return (true - pred) ** 2

def mse(pred, true):
    return np.mean(err2(pred, true))


# %%
n = 20
std = 4

x = np.linspace(-10, 10, 20)
noise = np.random.normal(0, 2, size=n)

y = f(x, slope=1.3, bias=5) + noise

data = pd.DataFrame({"x": x, "y": y})

# %%
slope_dom = np.linspace(-2, 4.5, 66)
slope_losses = {
    _slope: mse(f(x, _slope, bias=5), y)
    for _slope in slope_dom
}
df_slope_losses = pd.DataFrame({
    "slope": slope_losses.keys(),
    "loss": slope_losses.values(),
})

# %%
alt.renderers.enable('altair_viewer')


# %%
def show_lin_reg(
    slope: float,
    bias: float,
    show_pred=True,
    show_err=True,
    show_err2=False,
    show_loss_landscape=False,
):
    
    pred = x * slope + bias
    
    data["pred"] = pred
    data["err"] = y - pred
    data["err2"] = (y - pred) ** 2
    data["x2"] = x - data["err"]

    mse = np.mean(data['err2'])
    mae = np.mean(np.abs(data['err']))
    
    chart = (
        alt.Chart(data)
        .mark_point()
        .encode(x="x", y="y")
        .properties(title=f"Lin Reg | MSE: {mse:5.01f} | MAE: {mae:5.02f}")
    )
    if show_pred:
        chart += (
            alt.Chart(data)
            .mark_line()
            .encode(x="x", y="pred")
        )
    if show_err:
        chart += (
            alt.Chart(data)
            .mark_line()
            .encode(x="x", y="y", y2="pred")

        )
    if show_err2:
        chart += (
            alt.Chart(data)
            .mark_rect(fill="none", stroke="red")
            .encode(x="x", y="y", x2="x2", y2="pred")

        )
    

    if not show_loss_landscape:
        return chart
    
    _chart_loss = (
        alt.Chart(df_slope_losses)
        .mark_line()
        .encode(x="slope", y="loss")
        .properties(title="Loss Landscape (slope)")
    )
    _chart_loss_hl = (
        alt.Chart(pd.DataFrame({"x": [slope], "y": [0], "y2": [400]}))
        .mark_line()
        .encode(x="x", y="y", y2="y2")
    )
    return chart | (_chart_loss + _chart_loss_hl)

# %%
# show_lin_reg(
#     slope=.3,
#     bias=8,
#     show_pred=True,
#     show_err=True,
#     show_err2=False,
# )   

# %%
ipywidgets.interact(
    show_lin_reg,
    slope=(-2.0, 2.0),
    bias=(-8.0, 8.0),
    show_pred=True,
    show_err=False,
)

# %% [markdown]
# ## Linear regression - more formally
#
# 0. Data
#
#
# 1. Model:
#   - $f(X) = X \beta = \hat y$
#
#
# 2. Loss / criterion:
#   - $ err_i = y_i - f(X_i)$
#   - $MSE = \frac{1}{n} \sum_{i=1}^{N} err_i^2$
#
#
# 3. Optimize:
#   - minimize the MSE yields the optimal $\hat\beta$ (after doing some math)
#   - $\hat\beta = (X^TX)^{-1}X^Ty$
#   - (or, more generally, use gradient descent to optimize the parameters)
