from typing import Optional

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

def plot_one_sample(df: pd.DataFrame, sample_id: str, predictions: Optional[pd.Series] = None):
    """Plot one sample from the dataset.
    
    Args:
        df (pd.DataFrame): The dataset.
        sample_id (int): The id of the example to plot.
        type (str): The type of the plot. Options are 'trend', 'seasonality', 'cyclicality'.
        predictions (Optional[pd.Series], optional): The predictions of the model. Defaults to None.
    """

    df = df.set_index('store_family').loc[sample_id].reset_index()
    target = df.set_index('store_family').loc[sample_id].reset_index()['sales']

    # Create the figure
    fig = px.line(
        title=f"Store_family ID: {sample_id}",
        x=pd.to_datetime(df["date"]),
        y=target,
        labels={"x": "Time", "y": "Sales"},
    )

    # Add the prediction if available
    if predictions is not None:
        predictions_ = predictions.iloc[sample_id]
        fig.add_scatter(
            x=pd.to_datetime(df["date"])[-1:],
            y=[predictions_],
            mode="markers",
            marker_size=10,
            name="Prediction",
            line=dict(color="green", width=2),
        )

    # Show the figure
    return fig


def plot_trend(df: pd.DataFrame, sample_id: str):
    '''Plot the trend of a sample from the dataset'''

    df = df.set_index('store_family').loc[sample_id].reset_index()
    target = df.set_index('store_family').loc[sample_id].reset_index()['sales']

    moving_average = df['sales'].rolling(window=365, center=True, min_periods=183).mean()

    # Create the figure
    fig = px.line(
        title=f"Store_family ID: {sample_id}",
        x=pd.to_datetime(df["date"]),
        y=moving_average,
        labels={"x": "Time", "y": "Sales"},
    )
    fig.update_traces(line_color='red')

    fig.add_scatter(
    x=pd.to_datetime(df["date"]),
    y=target,
    mode="lines",
    line=dict(color="slateblue", width=3)
    )

    # Show the figure
    return fig


def plot_seasonality(df: pd.DataFrame, sample_id: str, type: str, number_of_periods: int):
    '''Plot the seasonality of a sample from the dataset, based on type and number of periods.
    Args:
        df (pd.DataFrame): The dataset.
        sample_id (str): The id of the example to plot.
        type (str): The type of the plot. Options are 'weekly', 'monthly'
        number_of_periods (int): The number of periods to plot.
    '''

    # randomly select number_of_periods from the year_week column

    
    df = df.set_index('store_family').loc[sample_id].reset_index()

    df['year'] = pd.to_datetime(df['date']).dt.year
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month_of_year'] = pd.to_datetime(df['date']).dt.month
    df['week_of_year'] = pd.to_datetime(df['date']).dt.weekofyear
    df = df[['year', 'month_of_year', 'week_of_year', 'day_of_week', 'sales']]

    if type == 'weekly':

        df['year_week'] = df['year'].astype(str) + '_' + df['week_of_year'].astype(str)
        sample_year_week = df['year_week'].sample(number_of_periods).unique()

        # Create the figure
        fig = px.line(
            title=f"{type}_seasonality - {number_of_periods} random timeseries",
            x=df[df['year_week'] == '2013_1']['day_of_week'],
            y=df[df['year_week'] == '2013_1']['sales'],
            labels={"x": "Time", "y": "Sales"},
        )

        for i in sample_year_week:

            fig.add_scatter(
            x=df[df['year_week'] == i]['day_of_week'],
            y=df[df['year_week'] == i]['sales'],
            mode="lines"
            )

    elif type == 'monthly':
        df = df.groupby(['year', 'month_of_year'])['sales'].sum().reset_index()

        # Create the figure
        fig = px.line(
            title=f"{type}_seasonality - {number_of_periods} random timeseries",
            x=df[df['year'] == 2013]['month_of_year'],
            y=df[df['year'] == 2013]['sales'],
            labels={"x": "Time", "y": "Sales"},
        )

        for i in df['year'].unique():

            fig.add_scatter(
            x=df[df['year'] == i]['month_of_year'],
            y=df[df['year'] == i]['sales'],
            mode="lines"
            )

        for i in df['year'].unique():

            fig.add_scatter(
            x=df[df['year'] == i],
            y=df[df['year'] == i]['sales'],
            mode="lines"
            )

    else:
        raise ValueError('type must be weekly or monthly')

    fig.update_layout(showlegend=False)
    fig.show()


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds))
        preds.plot(ax=ax)
    return ax


def plot_predictions(df_preds: pd.DataFrame, df_actual: pd.DataFrame, store_nbr: int, family: str):
    '''Takes in predictions and actual values and plots them'''

    # select store_nbr and family
    df_preds = df_preds[(df_preds['store_nbr'] == store_nbr) & (df_preds['family'] == family)]
    df_preds.set_index('date', inplace=True)
    df_preds.drop(columns=['store_nbr', 'family'], inplace=True)

    df_actual = df_actual[(df_actual['store_nbr'] == store_nbr) & (df_actual['family'] == family)]
    df_actual.set_index('date', inplace=True)
    df_actual.drop(columns=['store_nbr', 'family'], inplace=True)

    # plot predictions
    plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25")

    fig, (ax1) = plt.subplots(1, 1, figsize=(20, 6))
    plt.title(f'Actual vs Predicted Sales for Store {store_nbr}, Family {family}')

    # label x-axis
    ax1.set_xlabel('Date')

    # label y-axis
    ax1.set_ylabel('Sales')

    palette = dict(palette='husl', n_colors=64)

    ax1 = df_actual.multi_step_1.plot(**plot_params, ax=ax1)
    ax1 = plot_multistep(df_preds, ax=ax1, palette_kwargs=palette)
    _ = ax1.legend(['Actual Sale', 'Predicted Sale'])

    plt.show()
