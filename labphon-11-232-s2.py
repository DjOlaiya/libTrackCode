#Use: when pointed into the directory containing all the OpenFace results files
#for a participant in the metronomic speech task (produced by the prior script),
#this script constructs a consolidated results sheet for the participant,
#which contains vertical lip aperture trajectories for each trial.
#Four rows of VLA trajectories are created for each trial: two for the far camera distance
#(one in mm and one in pixels), and two again for the close camera distance.

import os
import math
from scipy import signal


directory_of_OF_results = 'xxx' #The directory that includes the .csv ouput files for each of this participant's trials

target_path = os.mkdir(os.path.join(directory_of_OF_results + 'consolidated/'))
outfile = open(os.path.join(directory_of_OF_results + 'consolidated/trajectories.csv'), 'w')

directory = os.fsencode(directory_of_OF_results)

#We first initialize our key variables:
participant_id = None
cam_id = None
run_num = None
syllname = None
token_num = None
initial_seg = None
vowel = None

#Creating the header for the final .csv file that will contain the participant's consolidated trajectory data
outfile.write('participant_id' + ',' + 'camera_id' + ',' + 'run_number' + ',' + 'syllable' + ',' + 'token_number' + ',' + 'initial_segment' + ',' + 
              'vowel' + ',' + 'min_conf' + ',' + 'max_pitch' + ',' + 'max_yaw' + ',' + 'mean_z_trans' + ',' + 'units' + ',' + 'VLA_frame_1' + ',' + 'VLA_frame_2' + 
              ',' + 'VLA_frame_3' + ',' + 'VLA_frame_4' + ',' + 'VLA_frame_5' + ',' + 'VLA_frame_6' + ',' + 'VLA_frame_7' + ',' + 'VLA_frame_8' + ',' + 'VLA_frame_9' + ',' + 'VLA_frame_10' + 
              ',' + 'VLA_frame_11' + ',' + 'VLA_frame_12' + ',' + 'VLA_frame_13' + ',' + 'VLA_frame_14' + ',' + 'VLA_frame_15' + ',' + 'VLA_frame_16' + ',' + 'VLA_frame_17' + ',' + 
              'VLA_frame_18' + ',' + 'VLA_frame_19' + ',' + 'VLA_frame_20' + ',' + 'VLA_frame_21' + ',' + 'VLA_frame_22' + ',' + 'VLA_frame_23' + ',' + 'VLA_frame_24' + ',' + 'VLA_frame_25' + 
              ',' + 'VLA_frame_26' + ',' + 'VLA_frame_27' + ',' + 'VLA_frame_28' + ',' + 'VLA_frame_29' + ',' + 'VLA_frame_30')

#Iterate over all the participant's OF output files (one for each trial)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith('.csv'):
        trial_data = open(os.path.join(directory_of_OF_results + filename), 'r')
        trajectory = [] #We create a set of empty lists to serve as buffers
        trajectory_2 = []
        trajectory_3 = []
        trajectory_4 = []
        confidences = []
        pitches = []
        yaws = []
        displacements = []
        row_counter = 0
        divided_filename = filename.split('.')
        chunks = divided_filename[0].split('_')
        participant_id = str(chunks[0]) #Grab important variable values for the trial from the savenames generated in the previous script
        cam_id = str(chunks[1])
        run_num = str(chunks[2])
        syllname = str(chunks[3])
        token_num = str(chunks[4])
        initial_seg = str(chunks[3][0])
        vowel = str(chunks[3][1])
        outfile.write('\n')
        for aline in trial_data:
            row_counter += 1
            if row_counter == 1:
                outfile.write(participant_id + ',' + cam_id + ',' + run_num + ',' + syllname + ',' + token_num + ',' + initial_seg + ',' + vowel + ',')
            else:
                values = aline.split(',') #Each row of the OpenFace output is turned into a list, and we can refer to specific column values by their indices
                #The operations below look to the columns containing the locations, in the (x,y) plane, of OF landmarks 51 and 57 (in both mm and pix)
                #and compute the relevant Euclidean distances
                vertical_lip_aperture = math.sqrt(((float(values[198]) - float(values[204])) ** 2) + ((float(values[266]) - float(values[272])) ** 2))
                vertical_lip_aperture_pix = math.sqrt(((float(values[62]) - float(values[68])) ** 2) + ((float(values[130]) - float(values[136])) ** 2))
                #We then store each vertical lip aperture trajectory as a list (initialized above)
                trajectory.append(vertical_lip_aperture)
                trajectory_2.append(vertical_lip_aperture_pix)
                confidence = float(values[3])
                abs_pitch = abs(float(values[8])) #We record confidence, absolute pitch and yaw, and displacement, to use as possible trial exclusion criteria
                abs_yaw = abs(float(values[9]))
                displace = float(values[7])
                confidences.append(confidence)
                pitches.append(abs_pitch)
                yaws.append(abs_yaw)
                displacements.append(displace)
        min_conf = min(confidences)
        max_pitch = max(pitches) #We compute the maximum absolute pitch and yaw over the trial, as well as the mean displacement over the trial. Again, if we need to, we can drop trials where these parameters exceed certain thresholds
        max_yaw = max(yaws)
        mean_displacement = (sum(displacements) / len(displacements))
        outfile.write(str(min_conf) + ',')
        outfile.write(str(max_pitch) + ',')
        outfile.write(str(max_yaw) + ',')
        outfile.write(str(mean_displacement) + ',' + 'mm' + ',')
        smoothed = signal.savgol_filter(trajectory, 7, 3, mode='nearest') #Here we apply a Savitzky-Golay filter to the trajectory, using 7-sample windows fit to 3rd-degree polynomials.
        smoothed2 = signal.savgol_filter(trajectory_2, 7, 3, mode='nearest')
        smoothed3 = signal.savgol_filter(trajectory_3, 7, 3, mode='nearest')
        smoothed4 = signal.savgol_filter(trajectory_4, 7, 3, mode='nearest')
        for i in smoothed:
            outfile.write(str(i) + ',')
        outfile.write('\n')
        outfile.write(participant_id + ',' + cam_id + ',' + run_num + ',' + syllname + ',' + token_num + ',' + initial_seg + ',' + vowel + ',')
        outfile.write(str(min_conf) + ',')
        outfile.write(str(max_pitch) + ',')
        outfile.write(str(max_yaw) + ',')
        outfile.write(str(mean_displacement) + ',' + 'pix' + ',')
        for i in smoothed2:
            outfile.write(str(i) + ',')
       
outfile.close()
            
print('\n' + '*** Trajectory processing complete ***')  
            