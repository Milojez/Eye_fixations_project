import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------IMPORTANT PARAMETERS-----------------------------
pix_per_deg = 57  # velocity of pixels per second corresponding to one degree per second (eye angular mov.) in the experiment env.
minimal_fix_time = 30 #in ms , minimal length of fixations to be a valid fixation
max_vel_fix_treshold = 30 #deg/s, velocity treshold in a fixation sequence, above = saccade

dt = 1/2000  # timestep between two csv datapoints in seconds



#----------------LOADING DATA------------------------------------
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

#----------------EXTRACTING COLUMNS-------------------------
# Trial and person numbers
participant_number = df.iloc[:, col_1_participant_number].values
trial_number = df.iloc[:, col_2_trial_number].values

# Coordinates left eye
x_cord_pix_left_eye = df.iloc[:, col_3_x_cord_pix_left_eye].values
y_cord_pix_left_eye = df.iloc[:, col_5_y_cord_pix_left_eye].values

# Coordinates right eye
x_cord_pix_right_eye = df.iloc[:, col_4_x_cord_pix_right_eye].values
y_cord_pix_right_eye = df.iloc[:, col_6_y_cord_pix_right_eye].values


#--------------------AVERAGING DATA-------------------------------------
# Averaging over two eyes and dealing with NaN data
x_cord_pix_both_eyes = np.zeros([len(x_cord_pix_right_eye)])
y_cord_pix_both_eyes = np.zeros([len(y_cord_pix_right_eye)])

for i in range(0,len(x_cord_pix_right_eye)):

    # If every eye has valid data (no serious disturbances)
    if not np.isnan(x_cord_pix_left_eye[i]) and not np.isnan(x_cord_pix_right_eye[i]):
        x_cord_pix_both_eyes[i] = (x_cord_pix_left_eye[i] + x_cord_pix_right_eye[i]) / 2
        y_cord_pix_both_eyes[i] = (y_cord_pix_left_eye[i] + y_cord_pix_right_eye[i]) / 2
    
    # Blinks etc.
    else:
        x_cord_pix_both_eyes[i] = 0
        y_cord_pix_both_eyes[i] = 0


#-----------------------------CALCULATING FIXATIONS PARTICIPANT->TRIAL-------------------------
# Initialize cumulative fixation array
cumulative_fixation_array = np.empty((0, 6))

# Iterate over unique participants
unique_participants = np.unique(participant_number)

#LOOP_1 - participant
for participant in unique_participants:

    # Filter data for the current participant, boolean mask
    participant_mask = participant_number == participant
    current_participant_trial_numbers = trial_number[participant_mask]

    # Iterate over unique trials for the current participant
    unique_trials = np.unique(current_participant_trial_numbers)

#LOOP_2 - trial
    for trial in unique_trials:
        print("person: ",participant, "trail: ",trial )

        # Filter data for the current trial
        trial_mask = current_participant_trial_numbers == trial
        current_x_cord = x_cord_pix_both_eyes[participant_mask][trial_mask]
        current_y_cord = y_cord_pix_both_eyes[participant_mask][trial_mask]
        

        # Initiating a velocity array with velocity zero at starting point
        xy_vel_lst = np.zeros(1)

        # Calculating the velocities for the trial
        for i in range(1, len(current_x_cord)):
            # Calculating velocities for every coordinate (which are not blinks etc.)
            if current_x_cord[i] != 0 and current_x_cord[i-1] != 0:
                x_vel = (current_x_cord[i] - current_x_cord[i-1]) / dt
                y_vel = (current_y_cord[i] - current_y_cord[i-1]) / dt
            # Blinks etc.          
            else:
                x_vel = 0
                y_vel = 0

            # Calculating a common velocity vector for x and y
            xy_vel_pix = np.sqrt(x_vel**2 + y_vel**2)

            # Changing velocity from pixels/sec for deg/sec
            xy_vel_deg = xy_vel_pix / pix_per_deg

            # Appending the velocity list
            xy_vel_lst = np.concatenate((xy_vel_lst, np.array([xy_vel_deg])), axis=0)
            

        # --------Filtering/Smoothening the array----------------
        window_size = 9  # Adjust as needed
        # Convolution to counteract the noise 
        xy_vel_lst_smooth = np.convolve(xy_vel_lst, np.ones(window_size)/window_size, mode='valid')


        #--------identifying fixations--------------------
        # fixations row -> [start_of_fix, end_of_fix, length, middle_time_of_fix, x_cor_end, y_cor_end]

        # Boolean array indicating where velocities are below the threshold
        below_threshold = (xy_vel_lst_smooth < max_vel_fix_treshold) & (xy_vel_lst_smooth != 0)

        # Find the indices where fixations start and end and store it in arrays
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
            x_cor_end = current_x_cord[int(middle_time_of_fix)]  # Assuming x coordinates are in xy_vel_lst_smooth
            y_cor_end = current_y_cord[int(middle_time_of_fix)]  # Assuming y coordinates are in xy_vel_lst_smooth

            fixation_info = [start*dt*1000, end*dt*1000, fixation_length*dt*1000, middle_time_of_fix*dt*1000, x_cor_end, y_cor_end]
            fixation_array.append(fixation_info)

        # Convert fixation_array to a numpy array
        fixation_array = np.array(fixation_array)
        # Making sure the minimal fixation length is kept
        fixation_array = fixation_array[fixation_array[:,2]>minimal_fix_time]

        # Appending cumulative fixation array with trial by trial
        cumulative_fixation_array = np.vstack((cumulative_fixation_array,fixation_array))
        # print(cumulative_fixation_array)

#-------------STATISTICS----------
# Calculate statistics
average = np.mean(cumulative_fixation_array[2])
std_deviation = np.std(cumulative_fixation_array[2])
median = np.median(cumulative_fixation_array[2])

print("fixations average: ", average)
print("fixations std deviation: ", std_deviation)
print("fixations median: ", median)



#--------------CREATING HISTOGRAPH------------------------
# Extract middle times
middle_times = cumulative_fixation_array[:, 3]

# Define bin width and range
bin_width = 200  # 200ms in seconds
bins_range = np.arange(0, 6000 + bin_width, bin_width)

# Compute histogram of middle times
hist, bin_edges = np.histogram(middle_times, bins=bins_range)

# Initialize an array to store proportions
bin_proportions = []

# Iterate through the bins to find proportion per bin
for i in range(len(bin_edges) - 1):
    # Find all fixations within a bin
    fixations_within_bin = cumulative_fixation_array[(cumulative_fixation_array[:, 3] >= bin_edges[i]) & (cumulative_fixation_array[:, 3] < bin_edges[i + 1])]
    # Define the conditions for x and y coordinates
    fixations_on_the_mirror = fixations_within_bin[(fixations_within_bin[:, 4] < 500) & (fixations_within_bin[:, 5] > 750)]
    
    # Calculate the proportion of rows meeting the condition
    if len(fixations_within_bin) == 0:
        proportion = 0
    else:
        proportion = len(fixations_on_the_mirror) / len(fixations_within_bin)
    
    # Append the proportion to the array
    bin_proportions.append(proportion)
    
# Add labels to the axes
plt.xlabel('Time in miliseconds (each bin 200ms)')
plt.ylabel("Mirror fixations / All fixations")

# Add a title to the plot
plt.title('Proportion of mirror to overall fixations in 200ms timesteps')

# Plot the proportions
plt.bar(bin_edges[:-1], bin_proportions, width=bin_width, edgecolor='black')
plt.show()


# #--------------SAVING FIXATIONS ARRAY-------------------------
# #saving the final array for later use
# file_path_save = 'Cumulative_fixations_array.txt'
# np.savetxt(file_path_save, cumulative_fixation_array)

