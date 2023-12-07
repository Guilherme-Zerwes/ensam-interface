import streamlit as st
from PIL import Image

with Image.open('./ressources/imgs/favicon.png') as img:
    st.set_page_config(
        page_title='Choose analysis type',
        page_icon=img
    )

#header styling
with open('./ressources/components/navbar.css', 'r') as f:
    style = f.read()
    st.markdown(
        f"""<style>{style}</style>
        """, unsafe_allow_html=True)
    
with open('./ressources/components/analysis.css', 'r') as f:
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
    with open('./ressources/components/analysis.html', 'r') as html:
        st.markdown(html.read(), unsafe_allow_html=True)
