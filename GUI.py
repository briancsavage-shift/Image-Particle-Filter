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
        mapSelected = st.sidebar.selectbox("Map of Environment", sorted(mapOptions))
        
        size = st.slider("Size of Observered Region", 25, 100, 75)
        particleCount = st.slider("Number of Particle for Filter", 100, 500, 250)
        maxNextGen = st.slider("Max Particle Count", 100, 500, 250)
        rounds = st.slider("Number of Rounds for Filter", 1, 10, 3)
        
        st.session_state["Simulation"].loadMap(os.path.join(os.getcwd(), "maps", mapSelected))
        st.session_state["Simulation"].setViewSize(size)

        st.button("Timestep +1", on_click=st.session_state["Simulation"].timestep)
        st.button("Reset Simulation", on_click=st.session_state["Simulation"].reset)
        st.button("Export Images", on_click=st.session_state["Simulation"].export)
        
        left.markdown("`Expected View`")
        right.markdown("`Actual View`")
        left.image(cv2.cvtColor(st.session_state["Simulation"].estimatedView(), cv2.COLOR_BGR2RGB))
        right.image(cv2.cvtColor(st.session_state["Simulation"].trueView(), cv2.COLOR_BGR2RGB))


    filter = ParticleFilter(sampleSize=particleCount, 
                            maxNextGen=maxNextGen, 
                            simulation=st.session_state["Simulation"])
    
    # (eX, eY) = st.session_state["Simulation"].estimatedPosition()
    images = filter.sense(rounds=rounds)

    # simHR.markdown(f"`-------Similarity-Heuristic\n {round(sH, 4)}`")
    # simML.markdown(f"`Similarity-Machine-Learning\n {cH}`")

    (ref, env) = st.session_state["Simulation"].export()


    st.title("Drone Simulation")
    st.write(":eight_spoked_asterisk: *Estimated Position*")
    st.write(":red_circle: *Actual Positions*")

    left, right = st.columns(2)
    left.markdown("`Drone Position`")
    right.markdown("`Map Overlayed`")
    left.image(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    right.image(cv2.cvtColor(env, cv2.COLOR_BGR2RGB))

    left, right = st.columns(2)
    left.markdown("`Actual Position Coordinates`")
    right.markdown("`Only Movement Vectors`")
    df = pd.DataFrame(st.session_state["Simulation"].dXY, columns=["dX", "dY"])
    right.write(df.assign(constraint=df.dX ** 2 + df.dY ** 2))
    left.write(pd.DataFrame(st.session_state["Simulation"].pos, columns=["X", "Y"]))
    
    st.markdown("-------")
    st.title("Particle Filter")
    
    for i, imgSet in enumerate(images):
        st.markdown(f"Round #{i + 1}")
        l, m, r = st.columns(3)
        l.markdown("`Post Weighting`")
        m.markdown("`Resampling based on Weight`")
        r.markdown("`Moved Resampled`")
        imageR1, imageR2, imageM2 = imgSet
        l.image(cv2.cvtColor(imageR1, cv2.COLOR_BGR2RGB))
        m.image(cv2.cvtColor(imageR2, cv2.COLOR_BGR2RGB))
        r.image(cv2.cvtColor(imageM2, cv2.COLOR_BGR2RGB))
        st.markdown("-------")



if __name__ == "__main__":
    main()