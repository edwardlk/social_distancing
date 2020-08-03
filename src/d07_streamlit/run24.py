#!/usr/bin/env python3
'''
'''
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from pickle import load
from pytz import timezone


def past_24_graph():
    # Load our data
    df = pd.read_pickle("../d01_data/03_SQL_data_for_frontend_ee.pkl")

    # Import model & add predictions
    # Import our model
    rf = load(open("../d03_modeling/rfc_HW_23.pkl",'rb'))

    # Define input to model
    X = df.drop('time_bin', axis=1)

    # Add predictions & predicted probability of UNSAFE to df
    df['prediction'] = rf.predict(X)
    probs = pd.DataFrame(rf.predict_proba(X), columns=rf.classes_)
    probs.columns = ['prob_safe', 'prob_unsafe']
    df['prob_unsafe'] = probs.prob_unsafe

    # Graph the safety estimates over the past 24 hours
    # Downsample to hourly for easy interpretability
    df_hourly = df[['time_bin', 'prob_unsafe']]
    df_hourly = df_hourly.set_index('time_bin')
    df_hourly = df_hourly.resample('h').mean()
    df_hourly = df_hourly.reset_index()
    df_hourly['hour'] = df_hourly.time_bin.apply(lambda x: x.hour)

    # Restrict to relevant hours & past 24 hrs
    yesterday = (
        dt.datetime.now(tz=timezone('US/Eastern')) - dt.timedelta(days=1))
    df_today = df_hourly[df_hourly.time_bin >= yesterday]

    # print(df_today.time_bin.strftime("%a, %B %d, %I:00 %p"))

    xlabels = df_today.time_bin.apply(lambda x: x.strftime("%a, %-I %p"))
    plt.style.use('ggplot')
    bar = plt.bar(x=df_today.time_bin, height=df_today.prob_unsafe, width=0.01)
    plt.xlabel("Day and Hour")
    plt.ylabel("Probability that it\'s UNSAFE")
    plt.xticks(ticks=df_today.time_bin, labels=xlabels, rotation=45, ha='right', fontsize=10)
    plt.title("Risk Level over the Past 24 Hours")
    plt.tight_layout()
    plt.savefig("../d06_visuals/risk_24hrs.png")


if __name__ == '__main__':
    past_24_graph()
