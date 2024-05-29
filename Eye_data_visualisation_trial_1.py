import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File path
file_path = 'Q1 - Eye_data.csv'

# Reading the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Specifying index 
col_1_participant_number = 0
col_2_trial_number = 1
col_3_x_cord_pix_left_eye = 2
col_4_x_cord_pix_right_eye = 3
col_5_y_cord_pix_left_eye = 4
col_6_y_cord_pix_right_eye = 5
col_7_frame_number = 6
col_8_key_pressed = 7
col_9_key_pressed_time = 8

# Extracting the specified columns into pandas Series
# Trial and person numbers
participant_number = df.iloc[:, col_1_participant_number].values
trial_number = df.iloc[:, col_2_trial_number].values

# Coordinates
x_cord_pix_left_eye = df.iloc[:, col_3_x_cord_pix_left_eye].values
y_cord_pix_left_eye = df.iloc[:, col_5_y_cord_pix_left_eye].values



#drawing visualisation
x_cord = np.empty(0)
y_cord = np.empty(0)
for i in range(0,len(participant_number)):

    if participant_number[i] == 1 and trial_number[i] == 1:
        x_cord = np.concatenate((x_cord, np.array([x_cord_pix_left_eye[i]])), axis=0) 
        y_cord = np.concatenate((y_cord, np.array([y_cord_pix_left_eye[i]])), axis=0)

    else:
        break

# Plot the data
plt.scatter(x_cord, y_cord, label='Points of focus')
plt.xlabel('X Coordinate (Pixels)')
plt.ylabel('Y Coordinate (Pixels)')
plt.title('Scatter Plot of X and Y Coordinates')
plt.legend()

# Set the x and y axis limits
plt.xlim(0, 1960)
plt.ylim(1080, 0)

plt.show()

