import streamlit as st
from PIL import Image
from streamlit_extras.switch_page_button import switch_page

#page config
with Image.open('./ressources/imgs/favicon.png') as img:
    st.set_page_config(
        page_title='Learn Machine Learning',
        page_icon=img
    )

#header styling
with open('./ressources/components/navbar.css', 'r') as f:
    style = f.read()
    st.markdown(
        f"""<style>{style}</style>
        """, unsafe_allow_html=True)
    
#page styling
with open('./ressources/components/home.css', 'r') as f:
    style = f.read()
    st.markdown(
        f"""<style>{style}</style>
        """, unsafe_allow_html=True)

#page body and content
header = st.container()
body = st.container()

with header:
    with open('./ressources/components/navbar.html', 'r') as html:
        st.markdown(html.read(), unsafe_allow_html=True)

with body:
    with open('./ressources/components/home.html', 'r') as html:
        st.markdown(html.read(), unsafe_allow_html=True)

    # change_page = st.button('Start Learning >>')
    # if change_page:
    #     switch_page("Input")
