import os, cv2
import streamlit as st
import pandas as pd
import numpy as np

from typing import List, Tuple
from matplotlib import pyplot as plt

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

        st.markdown("-----------")
        st.markdown("## Settings")
        mapOptions = os.listdir(os.path.join(os.getcwd(), "maps"))
        mapSelected = st.sidebar.selectbox("Map of Environment",
                                           sorted(mapOptions))
        
        size = st.slider("Size of Observered Region", 25, 100, 75)
        maxNextGen = st.slider("Max Particle Count", 100, 500, 250)
        rounds = st.slider("Number of Rounds for Filter", 1, 10, 3)
        
        st.session_state["Simulation"].loadMap(os.path.join(os.getcwd(),
                                                            "maps",
                                                            mapSelected))
        st.session_state["Simulation"].setViewSize(size)

        st.button("Timestep +1",
                  on_click=st.session_state["Simulation"].timestep)

        st.button("Reset Simulation",
                  on_click=st.session_state["Simulation"].reset)

        st.button("Export Images",
                  on_click=st.session_state["Simulation"].export)
        
        left.markdown("`Expected View`")
        right.markdown("`Actual View`")
        left.image(cv2.cvtColor(st.session_state["Simulation"].estimatedView(),
                                cv2.COLOR_BGR2RGB))
        right.image(cv2.cvtColor(st.session_state["Simulation"].trueView(),
                                 cv2.COLOR_BGR2RGB))

    pfilter = ParticleFilter(maxGenSize=maxNextGen,
                            simulation=st.session_state["Simulation"])

    (bpML, scrML, imgsML) = pfilter.senseWithLearning(rounds=rounds)
    (bpHU, scrHU, imgsHU) = pfilter.senseWithHeuristic(rounds=rounds)

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

    df = pd.DataFrame(st.session_state["Simulation"].dXY,
                      columns=["dX", "dY"])

    right.write(df.assign(constraint=df.dX ** 2 + df.dY ** 2))
    left.write(pd.DataFrame(st.session_state["Simulation"].pos,
                            columns=["X", "Y"]))
    
    st.markdown("-------")
    st.title("Particle Filter")

    movedHRImage = imgsML[-1][-1]
    movedMLImage = imgsHU[-1][-1]
    left, middle, right = st.columns(3)

    left.pyplot(makePlot((scrHU, scrML)))
    middle.image(cv2.cvtColor(movedHRImage, cv2.COLOR_BGR2RGB))
    right.image(cv2.cvtColor(movedMLImage, cv2.COLOR_BGR2RGB))


    for i, (bP, scr, imgSet, bP2, scr2, imgSet2) in enumerate(zip(bpHU,
                                                                  scrHU,
                                                                  imgsHU,
                                                                  bpML,
                                                                  scrML,
                                                                  imgsML)):
        st.markdown(f"### Round {i + 1}")
        st.markdown(f"***Heuristic***")
        l, m, r = st.columns(3)
        l.markdown("`Post Weighting`")
        m.markdown("`Resampling based on Weight`")
        r.markdown("`Normalized Distance from True Position`")
        imageR1, imageR2, imageM2 = imgSet
        
        overlay = lambda sImg : \
            cv2.addWeighted(sImg, 0.5,
                            st.session_state["Simulation"].environment, 0.5,
                            0)
        
        eX, eY = bP
        eX, eY = st.session_state["Simulation"].convertCoordinates(eX, eY)
        imgR1 = cv2.cvtColor(overlay(imageR1), cv2.COLOR_BGR2RGB)
        imgR2 = cv2.cvtColor(overlay(imageR2), cv2.COLOR_BGR2RGB)
        imgM2 = cv2.cvtColor(overlay(imageR2), cv2.COLOR_BGR2RGB)

        imgR1 = cv2.circle(imgR1, (eX, eY), 5, (0, 0, 255), -1)
        imgR2 = cv2.circle(imgR2, (eX, eY), 5, (0, 0, 255), -1)

        l.image(imgR1)
        m.image(imgR2)
        r.metric("Heuristic Score", round(scr, 4))

        st.markdown(f"***Machine Learning***")
        l, m, r = st.columns(3)
        l.markdown("`Post Weighting`")
        m.markdown("`Resampling based on Weight`")
        r.markdown("`Normalized Distance from True Position`")
        imageR1, imageR2, imageM2 = imgSet2

        eX, eY = bP2
        eX, eY = st.session_state["Simulation"].convertCoordinates(eX, eY)
        imgR1 = cv2.cvtColor(overlay(imageR1), cv2.COLOR_BGR2RGB)
        imgR2 = cv2.cvtColor(overlay(imageR2), cv2.COLOR_BGR2RGB)

        imgR1 = cv2.circle(imgR1, (eX, eY), 5, (0, 0, 255), -1)
        imgR2 = cv2.circle(imgR2, (eX, eY), 5, (0, 0, 255), -1)

        l.image(imgR1)
        m.image(imgR2)
        r.metric("Learning Based Score", round(scr2, 4))

        st.markdown("-------")


def makePlot(ys: Tuple[List[float], List[float]]) -> plt.figure:
    fig = plt.figure(figsize=(10, 8))
    for y in ys:
        x = np.arange(len(y))
        plt.plot(x, y, linewidth=2.0)
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Distance from true position")
    plt.title("Particle Filter Accuracies By Method")
    plt.legend(["Heuristic", "Learning Based"])
    return fig


if __name__ == "__main__":
    main()