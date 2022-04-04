# Image-Particle-Filter

[![Build Status](https://travis-ci.org/briancsavage/image-particle-filter.svg?branch=master)](https://travis-ci.org/briancsavage/image-particle-filter)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://share.streamlit.io/briancsavage/image-particle-filter/GUI.py#drone-simulation)

* ***Summary*** - Image based Particle Filter for Drone Localization
* [**Live Demo**](https://share.streamlit.io/briancsavage/image-particle-filter/GUI.py#drone-simulation)


# Usage & Installation
* [**Step 1**] - Clone repository locally via `git clone https://github.com/briancsavage/Image-Particle-Filter.git`
* [**Step 2**] - Navigate to repository locally via `cd /path/to/clone`
* [**Step 3**] - Install dependencies via `pip install -r requirements.txt`
* [**Step 4**] - Run Streamlit application locally via `streamlit run GUI.py`


# Implementation & Method

| Code | Explaination |
|:----:|:-------------|
| <img src="https://user-images.githubusercontent.com/47962267/161453030-69b3040a-175a-4d56-8db5-810d0f37ac44.png"> | <b>Heuristical Estimator for Position</b> <br/><br/> <ul> First, calculates the histogram for the reference and expected perspective. This returns a dictionary where the keys are values between 0-255 and the keys represent the frequency counts of the BGR values. </ul><ul> Using these frequency counts, we calculate the mean squared error between the reference and expected color histograms. Then, we return 1 divided by the mean squared error of the reference image.  </ul> |
| <img src="https://user-images.githubusercontent.com/47962267/161453132-533e876d-238e-491a-8d18-cd67104f92a9.png"> | <b>Hog Transformer</b> <br/><br/> <ul> Instead of using the color histogram, we use a `HOG Transformer` to compute the histogram of oriented gradients in 2x2 patches across the image. This returns a flattened feature vector for the image that expresses the directionality of the color change across the patches of the image. </ul> <ul> The final pre-processing step we perform is applying SKLearn's `StandardScaler()` to scale the feature vectors to a zero mean and unit variance. </ul><ul> We write a seperate lambda function for the hog transformation so that the pre-processing step could be easily parallelized, since the major computation of the function is in this hog feature extraction step.  |
| <img src="https://user-images.githubusercontent.com/47962267/161453137-c529feef-248b-4ed3-8fd6-71354592d8a1.png"> | <b>Learning Based Estimator for Position</b> <br/><br/> <ul> In the constructor for the `PerspectiveSimularity` class, we first initialize the hog feature extractor from above. Within the training step, we use a `SGDRegressor` as the model to train. The justification behind using a `SGD` algorithm over `LBFGS` or `ADAM` is since we have a minimal amount of data relative to the preformance benefits of `ADAM`, and `SGD` is more likely to converge to a global minimum than `LBFGS` at the expense of training time. </ul> <ul> At inference time, we fit the hog transformer and standard scaler to the images first. Then, we use the trained regressor to predict the `p(z\|x)` values for the provided images. Using these estimates, we apply a softmax function on the array to find the corresponding confidence levels for each of the tested for positions. |

<br><br>

# Simulation Interface

![Web capture_3-4-2022_192946_localhost](https://user-images.githubusercontent.com/47962267/161453642-2b407749-16b5-4851-a491-1d9a83bee303.jpeg)
![Web capture_3-4-2022_193014_localhost](https://user-images.githubusercontent.com/47962267/161453644-49d95e0d-896e-4e30-b8b0-eaed702e2544.jpeg)
![Web capture_3-4-2022_193034_localhost](https://user-images.githubusercontent.com/47962267/161453647-0e4b25cf-6b3d-41ef-8db3-b2692b968aad.jpeg)
![Web capture_3-4-2022_19312_localhost](https://user-images.githubusercontent.com/47962267/161453653-121bd4d8-3d96-439e-8cd1-8c5ec73cb162.jpeg)







