import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# IMPORTANT PARAMETERS
pix_per_deg = 57  # velocity of pixels per second corresponding to one degree per second in the experiment env.
dt = 1/2000  # timestep between two csv datapoints
minimal_fix_time = 30 #in ms
max_vel_fix_treshold = 50 #deg/s

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

# Coordinates left eye
x_cord_pix_left_eye = df.iloc[:, col_3_x_cord_pix_left_eye].values
y_cord_pix_left_eye = df.iloc[:, col_5_y_cord_pix_left_eye].values

# Coordinates right eye
x_cord_pix_right_eye = df.iloc[:, col_4_x_cord_pix_right_eye].values
y_cord_pix_right_eye = df.iloc[:, col_6_y_cord_pix_right_eye].values




# Averaging over two eyes and dealing with NaN data
x_cord_pix_both_eyes = np.zeros([len(x_cord_pix_right_eye)])
y_cord_pix_both_eyes = np.zeros([len(y_cord_pix_right_eye)])

for i in range(0,len(x_cord_pix_right_eye)):

    #if every eye has valid data
    if not np.isnan(x_cord_pix_left_eye[i]) and not np.isnan(x_cord_pix_right_eye[i]):
        x_cord_pix_both_eyes[i] = (x_cord_pix_left_eye[i] + x_cord_pix_right_eye[i]) / 2
        y_cord_pix_both_eyes[i] = (y_cord_pix_left_eye[i] + y_cord_pix_right_eye[i]) / 2

    else:
        x_cord_pix_both_eyes[i] = 0
        y_cord_pix_both_eyes[i] = 0



# Initiating velocity lists with velocity zero at starting point
x_vel_lst = np.zeros(1)
y_vel_lst = np.zeros(1)
xy_vel_lst = np.zeros(1)
dt_lst = [0]

# Creating velocity lists from coordinates
j=0 #time
for i in range(1, len(x_cord_pix_both_eyes)):

    if participant_number[i] == 1 and trial_number[i] == 1:
        # print(2)
        # Calculating velocities for every coordinate
        if x_cord_pix_both_eyes[i] != 0 and x_cord_pix_both_eyes[i-1] != 0:
            x_vel = (x_cord_pix_both_eyes[i] - x_cord_pix_both_eyes[i-1]) / dt
            y_vel = (y_cord_pix_both_eyes[i] - y_cord_pix_both_eyes[i-1]) / dt          
        else:
            x_vel = 0
            y_vel = 0   



        # Calculating a common velocity vector for x and y
        xy_vel_pix = np.sqrt(x_vel**2 + y_vel**2)
        # Changing velocity from pixels/sec for deg/sec
        xy_vel_deg = xy_vel_pix / pix_per_deg

        # Appending the velocity lists
        x_vel_lst = np.concatenate((x_vel_lst, np.array([x_vel])), axis=0)
        y_vel_lst = np.concatenate((y_vel_lst, np.array([y_vel])), axis=0)
        xy_vel_lst = np.concatenate((xy_vel_lst, np.array([xy_vel_deg])), axis=0)
        
        #next datapoint
        j += 1
        # Appending time lists in miliseconds
        dt_lst.append(dt * j * 1000)

    elif trial_number[i] == 2:
        break

# Apply moving average to velocity data
window_size = 9  # Adjust as needed
xy_vel_lst_smooth = np.convolve(xy_vel_lst, np.ones(window_size)/window_size, mode='valid')


#--------identifying fixations--------------------

# Boolean array indicating where velocities are below the threshold
below_threshold = (xy_vel_lst_smooth < max_vel_fix_treshold) & (xy_vel_lst_smooth != 0)

# Find the indices where fixations start and end
start_indices = np.where(np.diff(below_threshold.astype(int)) == 1)[0] + 1
end_indices = np.where(np.diff(below_threshold.astype(int)) == -1)[0] + 1

# If the first or last value is True, add the corresponding index
if below_threshold[0]:
    start_indices = np.insert(start_indices, 0, 0)
if below_threshold[-1]:
    end_indices = np.append(end_indices, len(xy_vel_lst_smooth))

# Create an array to store fixation information
fixation_array = []

# Iterate over fixations and populate the fixation_array
for start, end in zip(start_indices, end_indices):
    fixation_length = end - start
    middle_time_of_fix = (start + end) // 2
    x_cor_end = x_cord_pix_both_eyes[end]  # Assuming x coordinates are in xy_vel_lst_smooth
    y_cor_end = y_cord_pix_both_eyes[end]  # Assuming y coordinates are in xy_vel_lst_smooth

    fixation_info = [start*dt*1000, end*dt*1000, fixation_length*dt*1000, middle_time_of_fix*dt*1000, x_cor_end, y_cor_end]
    fixation_array.append(fixation_info)

# Convert fixation_array to a NumPy array
fixation_array = np.array(fixation_array)
fixation_array = fixation_array[fixation_array[:,2]>30*dt*1000]

print(fixation_array)
#---------------------------------------EXP------------------------------------------------


# [start_of_fix, end_of_fix, length, middle_time_of_fix, x_cor_end, y_cor_end]

fixations_array = np.empty((0,0))


#plotting
plt.plot(dt_lst[:len(xy_vel_lst_smooth)], xy_vel_lst_smooth, label='Smoothed Data', color='orange')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Smoothed Velocity over Time')
plt.legend()
plt.show()