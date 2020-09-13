#Use: Processes videos for each of a participant's sprints into a different video
#segment for each syllable.  These segments are fed through OpenFace using a Windows batch script
#and the resulting OpenFace output files are saved using names that capture relevant variable information.
#The script requires that the user point it at each of the participant's sprints for indivdual segmentation
#and tracking.

import os.path
import subprocess
from moviepy.video.io.VideoFileClip import VideoFileClip

directory = 'C:\\Users\\Lab\\Documents\\Validation\\' #location of this file
ppt = 's_01' #participant code, assumes data is inside a subfolder with this name
distance = 'far' #which camera footage is from, set to 'far' or 'close'
sprint = '01' #which of the five sprints is being coded
filextension = '.mp4' #file extension for input and output video
vid = ppt + '\\' + ppt + '_' + distance + '_' + sprint + filextension #input video file
numclips = 5 #number of clips to be iterated over per camera
numsegs = 20 #number of syllables per clip
beatlength = .48387 #length of time between each beat in seconds, calculated from metronome BPM-- 60/124 = .48387
includeaudio = False #'true' saves video clips out with audio track, 'false' removes the audio track for faster processing
openfacelocation = 'C:\\Users\\Lab\\Documents\\OpenFace_2.2.0' #file path to OpenFace's executables

#calibration parameters for close camera
if distance == 'close':
    fx_value = '0'
    fy_value = '0'
    cx_value = '0'
    cy_value = '0'

#calibration parameters for far camera
if distance == 'far':
    fx_value = '0'
    fy_value = '0'
    cx_value = '0'
    cy_value = '0'

#create folder for segmented videos if it doesn't already exist
try:
    segdirname = '\\segmented_videos' + '_' + ppt + '_' + distance + '_' + sprint + '\\'
    os.makedirs(ppt + segdirname)
except:
    pass

#create the counter that counts off syllables
counter = 1

#set trial start and end times for first trial
trial_start_time = 0
trial_end_time = trial_start_time + beatlength

#starting at the beginning of each clip, cuts it into NUMSEGS new clips of BEATLENGTH seconds each
#after that number of segments have been cut out, the rest of each clip is discarded
for i in range(0, numsegs):
    #count off beats in measure to determine syllable
    if counter == 1:
        syllable = 'nu'
    if counter == 2:
        syllable = 'mu'
    if counter == 3:
        syllable = 'ni'
    if counter == 4:
        syllable = 'mi'
    #adds one to iteration integer to make it more intuitively readable when saved out as part of the file name below
    totalcount = i + 1
    #for numbers under 10, add a leading zero 
    if totalcount < 10:
        totalcount = str(totalcount)
        totalcount = '0' + totalcount
    #save out each clip
    with VideoFileClip(vid) as video:
      new = video.subclip(float(trial_start_time), float(trial_end_time))
      new.write_videofile(ppt + segdirname + ppt + '_' + distance + '_' + sprint + '_' + str(syllable) + '_' + str(totalcount) + filextension, audio=includeaudio)
    #increase counter and prepare for next iteration
    trial_start_time = trial_end_time
    trial_end_time = trial_end_time + beatlength
    if counter < 4:
        counter +=  1
    elif counter == 4:
        counter = 1

print('\n' + 'Video cutting complete, creating batch script' + '\n' + '\n')

outfile = open(ppt + '\\OpenFace_Script.bat', 'w') #create batch file
outfile.write('cd ' + str(openfacelocation) + '\n' + '\n') #change directory to where OpenFace executables are
outfile.write('FeatureExtraction.exe') #run the feature extraction

#create a list of all the new segmented videos created in the previous loop
#by looking in the newly created 'segmented_videos' folder for any file with that file extension (e.g. MP4)
cliplist = []
for root, dirs, files in os.walk(ppt + segdirname):
    for file in files:
        if file.endswith(filextension):
            cliplist.append(file)

#for each clip in the cliplist, adds a line to the batch script with the file location
for clip in cliplist:
    video_label = clip
    outfile.write(' -f ' + directory + ppt + segdirname + video_label)

#add camera calibration information to the batch script and close the file
outfile.write(' -out_dir ' + str(directory + '/' + ppt + '\\all_Openface_results') + ' -2Dfp -3Dfp -tracked -pose -fx ' + fx_value + ' -fy ' + fy_value + ' -cx ' + cx_value + ' -cy ' + cy_value)
outfile.close()

print('\n' + 'Executing OpenFace batch script, this may take a few minutes...' + '\n' + '\n')

#execute the batch script
cmd = (ppt + '\\OpenFace_Script.bat')
subprocess.call(cmd, shell=True)
print('\n' + '*** Data coding complete ***' + '\n' + '\n')
