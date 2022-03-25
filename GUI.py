import os, cv2
import streamlit as st
import pandas as pd

from simulation import Simulate
from filter import ParticleFilter

def main():
    st.set_page_config(page_title="Scraping Launcher",
                       page_icon="🚀",
                       layout="wide",
                       initial_sidebar_state="expanded")

    if "Simulation" not in st.session_state:
        st.session_state["Simulation"] = Simulate()
    

    with st.sidebar:
        st.markdown("## Drone View")
        left, right = st.columns(2)

        st.markdown("-----------")
        st.markdown("## Settings")
        mapOptions = os.listdir(os.path.join(os.getcwd(), "maps"))
        mapSelected = st.sidebar.selectbox("Map of Environment", mapOptions)
        
        size = st.slider("Size of Observered", 25, 100, 25)
        st.session_state["Simulation"].loadMap(os.path.join(os.getcwd(), "maps", mapSelected))
        st.session_state["Simulation"].setViewSize(size)


        st.button("Timestep +1", on_click=st.session_state["Simulation"].timestep)
        st.button("Reset Simulation", on_click=st.session_state["Simulation"].reset)
        st.button("Export Images", on_click=st.session_state["Simulation"].export)
        posX = st.session_state["Simulation"].X
        posY = st.session_state["Simulation"].Y
        
        left.image(cv2.cvtColor(st.session_state["Simulation"].getDroneRef(), cv2.COLOR_BGR2RGB))
        right.image(cv2.cvtColor(st.session_state["Simulation"].getDroneEnv(), cv2.COLOR_BGR2RGB))




    flt = ParticleFilter(simulation=st.session_state["Simulation"])
    flt.generatePoints()
    sim = flt.similarityHeuristic(reference=st.session_state["Simulation"].getDroneRef(), 
                                  expected=st.session_state["Simulation"].getDroneEnv())  
    print('Similarity', sim)

    (ref, env) = st.session_state["Simulation"].export()

    st.write(":eight_spoked_asterisk: *Sum of Movement Vectors*")
    st.write(":red_circle: *Actual Current Position*")

    left, right = st.columns(2)
    left.image(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    right.image(cv2.cvtColor(env, cv2.COLOR_BGR2RGB))

    left, right = st.columns(2)
    df = pd.DataFrame(st.session_state["Simulation"].dXY, columns=["dX", "dY"])
    right.write(df.assign(constraint=df.dX ** 2 + df.dY ** 2))
    left.write(pd.DataFrame(st.session_state["Simulation"].pos, columns=["X", "Y"]))
    





if __name__ == "__main__":
    main()