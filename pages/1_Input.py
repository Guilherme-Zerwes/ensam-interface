import streamlit as st
import pandas as pd
from streamlit_extras.switch_page_button import switch_page
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
import process

with Image.open('./ressources/imgs/favicon.png') as img:
    st.set_page_config(
        page_title='Upload your data',
        page_icon=img
    )

if 'profile' not in st.session_state:
    st.session_state.profile = 0

def report_data(pandasDf):
    if st.session_state.profile == 0:
        return ydata_profiling.ProfileReport(pandasDf, title="Your data's profile report")
    else:
        return st.session_state.profile

#header styling
with open('./ressources/components/navbar.css', 'r') as f:
    style = f.read()
    st.markdown(
        f"""<style>{style}</style>
        """, unsafe_allow_html=True)
    
with open('./pages/1_Input.css', 'r') as f:
    style = f.read()
    st.markdown(
        f"""<style>{style}</style>
        """, unsafe_allow_html=True)

header = st.container()
body = st.container()

with header:
    with open('./ressources/components/navbar.html', 'r') as html:
        st.markdown(html.read(), unsafe_allow_html=True)

with body:
    st.markdown('<h1>Data Input</h1>', unsafe_allow_html=True)
    left,right = st.columns([1,1])
    file = left.file_uploader('Please upload the desired dataset', type=['csv'], key="batata")
    delimiter = right.selectbox('Please select the delimiter', options=[',', ';'])

    if file:
        df = pd.read_csv(file, delimiter=delimiter)
        df.to_csv('data/dataset.csv', index=False)
        st.write("Data preview")
        st.dataframe(df.head())
        
        make_profile = st.checkbox('Generate profile report?', help="Generates a exploratory data analysis report")
        preprocess = st.checkbox('Preprocess your data?', help="Pre-processes the dataset. Removes numerical missing values by inputing with the median value along said column. Also aplies a standart scalar that removes the average and turns the standart deviation to 1. At last, it aplies OneHotEncoding to the categorical data.")

        if preprocess:
            left,right = st.columns([1,1])
            not_apply = left.multiselect('Select the columns to not apply the pre-processing', options=df.columns)
            right.write(" ")
            right.write(" ")
            right.write(" ")
            right.write(" ")
            run = right.button('Run')

            if run:
                df_process = df.drop(not_apply, axis=1)
                df_process = process.process_data(df_process)
                df_process[not_apply] = df[not_apply]
                df_process.to_csv('data/dataset.csv', index=False)
                st.write('Your preprocessing was successful!')
                switch_page("Analysis")
        
        change_page = st.button('Choose analysis >>', key=1)

        if make_profile:
            st.session_state.profile = report_data(df)
            st_profile_report(st.session_state.profile)
            st.download_button('Download report', st.session_state.profile.to_html(), 'data.html',  key=2)

        if change_page:
            switch_page("Analysis")