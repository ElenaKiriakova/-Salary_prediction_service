import pandas as pd
import dill as pickle
import plotly.express as px
import streamlit as st

st.set_page_config(
    layout='wide'
)

def dash0():

    st.title('')

    with open(r'models/dash0', 'rb') as file:
        dash0 = pickle.load(file)

    fig = px.bar(dash0['most_common_skills'])
    fig.update_layout(
        width = 1000,
        height = 500,
        title="Most frequent skills",
        xaxis_title="Skills",
        yaxis_title="Count",
        legend_title="Category",
        title_x = 0.5
    )

    st.write(fig)

    with open(r'models/dash1', 'rb') as file:
        dash1 = pickle.load(file)

    nvac = pd.DataFrame.from_dict(dash1['most_common_vac'], orient='index')
    labels = nvac.index
    values = nvac.values.flatten()

    fig = px.pie(labels=labels, values=values, names=labels,
                 title="Distribution of vacancy fields")
    fig.update_layout(title_x=0.7,
                      width = 1000,
                      height = 700,
                      margin = dict(l=300))
    st.write(fig)

    with open(r'models/dash2', 'rb') as file:
        dash2 = pickle.load(file)

    fig = px.box(dash2['salary'], x="Category", y="Salary_median")
    fig.update_layout(
        title="Salary according to vacancy",
        xaxis_title="Vacancy",
        yaxis_title="Salary",
        title_x=0.5,
        width=1000,
        height=700,
        # margin=dict(l=300)
    )
    st.write(fig)

    with open(r'models/dash3', 'rb') as file:
        dash3 = pickle.load(file)

    fig = px.line(dash3['experience'])
    fig.update_layout(
        title="Salary according to experience",
        xaxis_title="Years of experience",
        yaxis_title="Salary",
        legend_title="Vacancy",
        title_x = 0.5,
        width = 1000,
        height = 700,
    )
    st.write(fig)

    with open(r'models/dash4', 'rb') as file:
        dash4 = pickle.load(file)

    fig = px.line(dash4['level'])
    fig.update_layout(
        title="Salary according to position level",
        xaxis_title="Position level",
        yaxis_title="Salary",
        legend_title="Vacancy",
        title_x=0.5,
        width=1000,
        height=700,
        # margin=dict(l=300)
    )
    st.write(fig)


