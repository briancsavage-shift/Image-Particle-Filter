import os, cv2
import streamlit as st
import pandas as pd

from simulation import Simulate
from filter import ParticleFilter

def main():
    st.set_page_config(page_title="Particle Filter Simulation",
                       page_icon="🚀",
                       layout="wide",
                       initial_sidebar_state="expanded")

    if "Simulation" not in st.session_state:
        st.session_state["Simulation"] = Simulate()
    

    with st.sidebar:
        st.markdown("## Drone View")
        left, right = st.columns(2)
        simHR = st.empty()
        simML = st.empty()

        st.markdown("-----------")
        st.markdown("## Settings")
        mapOptions = os.listdir(os.path.join(os.getcwd(), "maps"))
        mapSelected = st.sidebar.selectbox("Map of Environment", mapOptions)
        
        size = st.slider("Size of Observered", 25, 100, 75)
        st.session_state["Simulation"].loadMap(os.path.join(os.getcwd(), "maps", mapSelected))
        st.session_state["Simulation"].setViewSize(size)


        st.button("Timestep +1", on_click=st.session_state["Simulation"].timestep)
        st.button("Reset Simulation", on_click=st.session_state["Simulation"].reset)
        st.button("Export Images", on_click=st.session_state["Simulation"].export)
        posX = st.session_state["Simulation"].X
        posY = st.session_state["Simulation"].Y
        
        ref = st.session_state["Simulation"].getDroneRef()
        ref = cv2.applyColorMap(ref, cv2.COLORMAP_JET)
        left.image(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
        right.image(cv2.cvtColor(st.session_state["Simulation"].getDroneEnv(), cv2.COLOR_BGR2RGB))


    flt = ParticleFilter(simulation=st.session_state["Simulation"])
    flt.generatePoints()
    sH = flt.similarityHeuristic(ref=st.session_state["Simulation"].getDroneRef(), 
                                 exp=st.session_state["Simulation"].getDroneEnv())  
    cH = flt.similarityHeuristic(ref=st.session_state["Simulation"].getDroneRef(), 
                                 exp=st.session_state["Simulation"].getDroneRef())  


    simHR.markdown(f"### Similarity Heuristic: `{sH}`")
    simML.markdown(f"### Similarity Check: `{cH}`")

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