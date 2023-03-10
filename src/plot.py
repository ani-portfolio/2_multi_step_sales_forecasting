from typing import Optional

import pandas as pd
import numpy as np
import plotly.express as px

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