# Interface-property-predictions
## Information
Function: Predictions of traction-separation(T-S) laws using force-displacement(F-D) data from double cantilever beam tests  
Authors: Sanjida Ferdousi and Dr. Yijie Jiang, University of North Texas  
Version: 1.2  
Date: March 23rd, 2021.  
## Requirements for setting up the environment:
Python 3.7+ (recommended: Python 3.8)  
Required Python packages: scikit-learn, pickle, pandas, xgboost  
Required files: “predict.py” and “multioutputregressor.pkl”.  
## Steps:
Steps to evaluate the trained model on user’s data:  
Step 1: Data preparation. To predict T-S relations, normalized Force data should be used as input. Normalized force is F ̅=Fa^2/(Eh^3 t) where a is the distance of loading location to initial crack tip, E is the young’s modulus of the beam, h is the beam thickness, and t is the beam width of user’s DCB model. User also need to input the maximum value of the normalized displacement (d ̅=d/a).  Input data must be stored in csv format. While prediction, user’s force-displacement (F-D) data will be automatically interpolated uniformly into 1000 segments with the range from 0 to 0.4.   
Step 2: Keep the trained model “multioutputregressor.pkl”, evaluation code “predict.py” and input data in the same folder.   
Step 3: Run evaluation code “predict.py”. You will receive the messages that will guide you for further process.  
Step 4: After completing the prediction, the output will be stored in “normalized_prediction.csv” file at the current directory. Each row in this csv file represents the predicted normalized T-S data for the corresponding normalized F-D data. Please note that each predicted T-S data is resulted in 500 segments from 0 to 0.024 separation. The predicted T-S laws can be visualized if the user runs the “predict.py” file using any UI (e.g., Spyder, PyCharm).  
Step 5: In addition to T-S relation prediction, users can also evaluate the model if they provide a normalized T-S laws dataset along with the maximum value of normalized separation. Note that normalized traction is  T ̅=T/E where E is young modulus of the beam and normalized separation ( S ̅=S/a , where a is the initial crack length). There is no restriction for number of segments in traction data.   
Example files of input force data ‘normalized_Force.csv’ and its prediction ‘normalized_traction.csv’ are provided.  
# Citation
Please cite our paper if the paper (and code) is useful to you. paper: "Characterize traction–separation relation and interfacial imperfections by data-driven machine learning models".  
Ferdousi, S., Chen, Q., Soltani, M. et al. Characterize traction–separation relation and interfacial imperfections by data-driven machine learning models. Sci Rep 11, 14330 (2021). https://doi.org/10.1038/s41598-021-93852-y  
