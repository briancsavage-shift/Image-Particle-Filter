import os, cv2
import streamlit as st
import pandas as pd

from simulation import Simulate

def main():
    

    st.set_page_config(page_title="Scraping Launcher",
                       page_icon="🚀",
                       layout="wide",
                       initial_sidebar_state="expanded")

    if "Simulation" not in st.session_state:
        st.session_state["Simulation"] = Simulate()

    with st.sidebar:
        st.sidebar.title("Settings")
        mapOptions = os.listdir(os.path.join(os.getcwd(), "maps"))
        mapSelected = st.sidebar.selectbox("Map of Environment", mapOptions)
        
        l, r = st.columns(2)
        l.button("Timestep", on_click=st.session_state["Simulation"].timestep)
        r.button("Reset", on_click=st.session_state["Simulation"].reset)
        #numTimestep = st.sidebar.number_input("Number of steps", min_value=1, max_value=100, value=10)
        #exportImage = st.sidebar.checkbox("Export data", value=False)
    
    st.session_state["Simulation"].loadMap(os.path.join(os.getcwd(), "maps", mapSelected))
    (ref, env) = st.session_state["Simulation"].export()

    left, right = st.columns(2)
    left.image(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    right.image(cv2.cvtColor(env, cv2.COLOR_BGR2RGB))

    left, right = st.columns(2)
    left.write(pd.DataFrame(st.session_state["Simulation"].pos, columns=["X", "Y"]))
    
    df = pd.DataFrame(st.session_state["Simulation"].dXY, columns=["dX", "dY"])
    right.write(df.assign(constraint=df.dX ** 2 + df.dY ** 2))






if __name__ == "__main__":
    main()