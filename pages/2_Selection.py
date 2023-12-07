import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import models

#Variables
anal_ops = ['Regression analysis', 'Classification analysis', 'Description analysis', 'Clustering analysis']

reg_ops = ['Linear Regression', 'Polynomial Regression', 'Random Forests Regressor', 'Support Vector Machines Regression', 
        'K-Nearest-Neighbors Regressor','Artificial Neural Networks Regressor']

class_ops = ['Decision Tree Classifier', 'Random Forests Classifier','Support Vector Machine Classifier',
            'K-Nearest-Neighbors Classifier', 'Artificial Neural Networks Classifier']

desc_ops = ['Principal Component Analysis', 'Kernel Principal Component Analysis', 'Autoencoder']

clust_ops = ['K-means Clustering']

#Functions
if 'metrics' not in st.session_state:
    st.session_state.metrics = 0

#header styling
with open('./ressources/components/navbar.css', 'r') as f:
    style = f.read()
    st.markdown(
        f"""<style>{style}</style>
        """, unsafe_allow_html=True)

header = st.container()

with header:
    with open('./ressources/components/navbar.html', 'r') as html:
        st.markdown(html.read(), unsafe_allow_html=True)

def call_reg_model(algo, targ, hyper):
    st.session_state.metrics = models.train_reg_model(algo, targ, hyper)
    return

def call_class_model(algo, targ, hyper):
    st.session_state.metrics = 0
    st.session_state.metrics = models.train_class_model(algo, targ, hyper)
    return

def call_desc_model(algo, not_apply, hyper):
    st.session_state.metrics = 0
    st.session_state.metrics = models.train_desc_model(algo, hyper, not_apply) 

def call_clust_model(algo, hyper):
    st.session_state.metrics = 0
    st.session_state.metrics = models.train_clust_model(algo, hyper)

def zeroMet():
    st.session_state.metrics = 0

def display_metrics():
    if st.session_state.metrics != 0 and vis:
            for i in range(len(st.session_state.metrics)):
                st.write(st.session_state.metrics[i])

def display_hyps(algo):
    vals = []
    for i in range(len(hyp_dict)):
        if hyp_dict[i]['Algorithm'] != algo:
            continue
        
        if hyp_dict[i]['Type'] == 'slider':
            max_val = float(hyp_dict[i]['max_value'])
            if hyp_dict[i]['Algorithm'] in desc_ops:
                max_val = df.shape[1]
            val = st.slider(hyp_dict[i]['Name'],
                    min_value=float(hyp_dict[i]['min_value']), 
                    max_value=float(max_val),
                    step=float(hyp_dict[i]['step']), 
                    value=float(hyp_dict[i]['value']))
        else:
            val = st.select_slider(hyp_dict[i]['Name'], 
                    options=hyp_dict[i]['options'].replace(' ', '').split(','))
        vals.append(val)
    return vals

#hyperparameters options
hyp_dict = pd.read_excel('./hyperparams.xlsx').to_dict('records')

#Frontend
if os.path.exists('data/processed_dataset.csv') or os.path.exists('data/dataset.csv'):
    df = pd.read_csv('data/processed_dataset.csv') if os.path.exists('data/processed_dataset.csv') else pd.read_csv('data/dataset.csv')

    analysis_type = st.selectbox('Choose what type of analysis you would like to perform on your data', options=anal_ops, on_change=zeroMet)

    if analysis_type == anal_ops[0]:
        algorithm = st.selectbox('Please select the algorithm to work on the data', options=reg_ops, on_change=zeroMet)

        hyper_parms = display_hyps(algorithm)

        target = st.selectbox('Please choose the collums with the target values', options=[col for col in df.columns])

        st.button('Run!', on_click=call_reg_model, args=[algorithm, target, hyper_parms])
        vis = st.checkbox('Visualize result metrics')

        display_metrics()        

    elif analysis_type == anal_ops[1]:
        algorithm = st.selectbox('Please select the algorithm to work on the data', options=class_ops, on_change=zeroMet)

        hyper_parms = display_hyps(algorithm)

        target = st.selectbox('Please choose the collums with the target values', options=[col for col in df.columns])

        st.button('Run!', on_click=call_class_model, args=[algorithm, target, hyper_parms])
        vis = st.checkbox('Visualize result metrics')

        display_metrics()

    elif analysis_type == anal_ops[2]:
        algorithm = st.selectbox('Please select the algorithm to work on the data', options=desc_ops, on_change=zeroMet)
        
        hyper_parms = display_hyps(algorithm)

        not_apply = st.multiselect('Select the columns to not apply the algorithm', options=df.columns)
        st.button('Run!', on_click=call_desc_model, args=[algorithm, not_apply, hyper_parms])
        vis = st.checkbox('Visualize result metrics')

        display_metrics()

    elif analysis_type == anal_ops[3]:
        algorithm = st.selectbox('Please select the algorithm to work on the data', options=clust_ops, on_change=zeroMet)

        hyper_parms = display_hyps(algorithm)

        st.button('Run!', on_click=call_clust_model, args=[algorithm, hyper_parms])
        vis = st.checkbox('Visualize result metrics')

        display_metrics()

else:
        st.write('Please upload your data before continuing')