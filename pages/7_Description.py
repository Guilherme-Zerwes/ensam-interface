import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import models

with Image.open('./ressources/imgs/favicon.png') as img:
    st.set_page_config(
        page_title='Description Analysis',
        page_icon=img
    )

#Variables

desc_ops = ['Principal Component Analysis', 'Kernel Principal Component Analysis', 'Autoencoder']

#Functions
if 'metrics' not in st.session_state:
    st.session_state.metrics = 0

#header styling
with open('./ressources/components/navbar.css', 'r') as f:
    style = f.read()
    st.markdown(
        f"""<style>{style}</style>
        """, unsafe_allow_html=True)
     
#page styling
with open('./pages/7_Description.css', 'r') as f:
    style = f.read()
    st.markdown(
        f"""<style>{style}</style>
        """, unsafe_allow_html=True)

header = st.container()

with header:
    with open('./ressources/components/navbar.html', 'r') as html:
        st.markdown(html.read(), unsafe_allow_html=True)

def call_desc_model(algo, not_apply, hyper):
    st.session_state.metrics = 0
    st.session_state.metrics = models.train_desc_model(algo, hyper, not_apply)

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

    st.markdown('<h1>Description Analysis</h1>', unsafe_allow_html=True)

    algorithm = st.selectbox('Please select the algorithm to work on the data', options=desc_ops, on_change=zeroMet)
    
    hyper_parms = display_hyps(algorithm)

    not_apply = st.multiselect('Select the columns to not apply the algorithm', options=df.columns)

    left, right = st.columns(2)

    left.button('Run!', on_click=call_desc_model, args=[algorithm, not_apply, hyper_parms])
    vis = right.checkbox('Visualize result metrics')

    display_metrics()

else:
        st.write('Please upload your data before continuing')