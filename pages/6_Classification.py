import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import models
import reduced
import inspection

with Image.open('./ressources/imgs/favicon.png') as img:
    st.set_page_config(
        page_title='Classification Analysis',
        page_icon=img
    )

#Variables
class_ops = ['Decision Tree Classifier', 'Random Forests Classifier','Support Vector Machine Classifier',
            'K-Nearest-Neighbors Classifier', 'Artificial Neural Networks Classifier']

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
with open('./pages/6_Classification.css', 'r') as f:
    style = f.read()
    st.markdown(
        f"""<style>{style}</style>
        """, unsafe_allow_html=True)

header = st.container()

with header:
    with open('./ressources/components/navbar.html', 'r') as html:
        st.markdown(html.read(), unsafe_allow_html=True)

def call_class_model(algo, targ, hyper):
    df = pd.read_csv('data/dataset.csv')
    if not inspection.verify_str(df.drop(columns=targ)):
        st.warning('Your dataset contains non floats or integers values')
        return
    if not inspection.very_NaN(df):
        st.warning('Your dataset has missing values')
        return
    if not inspection.verify_targ(df[targ]):
        st.warning('The target value should be categorical, not continuous')
        return
    if reduce:
        df = reduced.reduce_order(targ, df)
    st.session_state.metrics = 0
    st.session_state.metrics = models.train_class_model(algo, targ, hyper, df)
    st.success("Dataset analysed sucessfully!")
    return

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
if os.path.exists('data/dataset.csv'):
    df = pd.read_csv('data/dataset.csv')     

    st.markdown('<h1>Classification Analysis</h1>', unsafe_allow_html=True)

    algorithm = st.selectbox('Please select the algorithm to work on the data', options=class_ops, on_change=zeroMet)

    hyper_parms = display_hyps(algorithm)

    target = st.selectbox('Please choose the collums with the target values', options=[col for col in df.columns])

    left, right = st.columns(2)

    left.button('Run!', on_click=call_class_model, args=[algorithm, target, hyper_parms])
    reduce = right.checkbox('Reduced order', help="Applies Proper Orthogonal Decomposition to combine the properties and keeps only the most relevant ones, which enables training a model with a smaller overall dimension.")
    vis = right.checkbox('Visualize result metrics')

    display_metrics()
else:
        st.write('Please upload your data before continuing')