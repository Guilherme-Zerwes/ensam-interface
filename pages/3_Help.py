import streamlit as st
import os
import base64
from PIL import Image

with Image.open('./ressources/imgs/favicon.png') as img:
    st.set_page_config(
        page_title='Help',
        page_icon=img
    )

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

with open('./pages/3_Help.css', 'r') as f:
    style = f.read()
    st.markdown(
        f"""<style>{style}</style>
        """, unsafe_allow_html=True)

def open_pdfs(file):
    with open(os.path.join('./ressources/files', file), 'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

st.markdown('<h1>Help Files</h1>', unsafe_allow_html=True)

file_names = os.listdir('./ressources/files')
counter = 0
for i in range(len(file_names)):
    if file_names[i].startswith('_'):
        nice_file_names = file_names[i].removeprefix(f'_{counter + 1}_').replace('_',' ').removesuffix('.pdf')
        st.button(f'{i+1} - {nice_file_names}', on_click=open_pdfs, args=[file_names[i]])
        counter += 1