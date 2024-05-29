# Eye_fixations_project
The goal of this project is to analyze eye fixations data of multiple participants to establish the statistics of eye fixations and the proportion between the fixations on the side mirror and the ones in front of the car.

For the analysis this repository contains 3 python files. The main data file "Q1 - Eye_data.csv" is not here but can be provided privately if requested.

The python files:

### Eye_data_visualisation_trial_1.py 
- Reads eye-tracking data from a CSV file, extracts specific columns related to participant number, trial number, and eye coordinates, then creates a scatter plot of the X and Y coordinates of the left eye for a specific participant and trial. The scatter plot visualizes the points of focus during the trial.

### Eye_velocity_visualisation_trial_1.py
- Reads eye-tracking data from a CSV file, calculates velocities and identifies fixations based on predefined parameters. It then plots the smoothed eyes velocity over time for a specific trial. Based on this it is possible to visually detect fixations and saccades.

### Main_programme.py
- Reads eye-tracking data from a CSV file and extracts relevant columns. After averaging over both eyes and handling NaN values, it calculates velocities and identifies fixations for all participants and trials. The fixations are determined based on velocity thresholds, and statistics such as average, standard deviation, and median fixation duration are calculated fo the whole dataset. Additionally, it creates a histogram showing the proportion of fixations on a mirror surface compared to overall fixations over time. Finally, it optionally saves the fixation data array for future use.

### Graphs folder
Additionally the repository includes produced graphs for analysis in the Graphs folder.
