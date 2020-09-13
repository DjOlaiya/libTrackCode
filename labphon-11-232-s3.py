"""
Use: Automatically trim trials containing extreme head orientations from a master data sheet.
Before running this script, ensure that you have created a single master data sheet from the summary sheets created for each participant.  There should be 
a single set of column headers at the top of this summary datasheet.
"""
infile = open('master data sheet location', 'r') #point to the your master data sheet
outfile = open('trimmed data sheet you are creating', 'w') #give the name of the trimmed datasheet you wish to create

#Initialize some lists and dictionaries we'll need later

max_pitches = []
max_yaw = []
part_IDs = {}
close_mm = []
close_pix = []
far_mm = []
far_pix = []

#Iterate through the raw master data sheet, and store the redundant data for the different
#combinations of cameras and units to different lists. We can iterate through these lists later.

line_count = 0
for aline in infile:
    line_count += 1
    pieces = aline.split(',')
    if line_count == 1:
        header = pieces
    elif pieces[1] == 'close' and pieces[11] == 'mm':
        close_mm.append(pieces)
    elif pieces[1] == 'close' and pieces[11] == 'pix':
        close_pix.append(pieces)
    elif pieces[1] == 'far' and pieces[11] == 'mm':
        far_mm.append(pieces)
    elif pieces[1] == 'far' and pieces[11] == 'pix':
        far_pix.append(pieces)

#Grab the full set of each maximum pitch and maximum yaw (without grabbing the redundant
# copies repeated for other cameras/units) so we can compute means and SDs.
#Note that we can't trust values from trials with bad tracking (which we'll drop later on anyway),
#so we exclude that data from being compiled.
#Also, grab partipant IDs to add to our dictionary, so we can create within-participant
# means and SDs for mean Z-translation later on.

for i in close_mm:
    if float(i[7]) >= 0.85:
        max_pitches.append(i[8])
        max_yaw.append(i[9])
    if (i[0] in part_IDs): 
        continue
    else: 
        part_IDs.update({i[0]:[0, 0]})

#The next several sections create means and SDs for maximum pitch and maximum yaw.
#We'll use these for computing z-scores to use to exclude extreme trials later on.
        
sum_max_pitches = 0
sum_max_yaw = 0

for i in max_pitches:
    sum_max_pitches += float(i)

mean_max_pitch = (sum_max_pitches / len(max_pitches))
sq_residual_pitches = []

for i in max_pitches:
    sq_resid = (float(i) - mean_max_pitch)**2
    sq_residual_pitches.append(sq_resid)
    
ss_pitches = 0

for i in sq_residual_pitches:
    ss_pitches += i
    
stdev_pitches = (ss_pitches / len(max_pitches))**0.5

for i in max_yaw:
    sum_max_yaw += float(i)
    
mean_max_yaw = (sum_max_yaw / len(max_pitches))
sq_residual_yaw = []

for i in max_yaw:
    sq_resid = (float(i) - mean_max_yaw)**2
    sq_residual_yaw.append(sq_resid)
    
ss_yaw = 0

for i in sq_residual_yaw:
    ss_yaw += i
    
stdev_yaw = (ss_yaw / len(max_pitches))**0.5

#We will use our by-participant dictionary to create within-participant means and 
#SDs for Z-translation.  We'll use these to generate within-participant Z-scores
#to serve as the basis for excluding extreme trials a little later.

for key in part_IDs:
    z_trans = []
    for i in close_mm:
        if i[0] == key and float(i[7]) >= 0.85: 
            z_trans.append(i[10])
    sum_z_trans = 0
    for i in z_trans:
        sum_z_trans += float(i)
    mean_z_trans = (sum_z_trans / len(z_trans))
    sq_residual_trans = []
    for i in z_trans:
        sq_resid = (float(i) - mean_z_trans)**2
        sq_residual_trans.append(sq_resid)
    ss_trans = 0
    for i in sq_residual_trans:
        ss_trans += i
    stdev_trans = (ss_trans / len(z_trans))
    temp_dict = {key: [mean_z_trans, stdev_trans]}
    part_IDs.update(temp_dict)
    

#The trick now is keeping the data from the two cameras (not to mention the different units)
# properly reconciled.  We want to mark trials to be dropped if EITHER camera got bad tracking,
#so we start by marking badly tracked trials from the far camera, then update the close camera's 
#trials to be dropped so that they match.  Then we start determining from the close camera which trials to 
#drop due to extreme head orientation.  Then, using the close camera (mm) as our master copy,
#we'll update the trials to be dropped in each of the other three combinations of camera and units.

for i in far_mm:
    if float(i[7]) < 0.85:
        i.append('drop')
    else:
        i.append('keep')
        
for i in close_mm:
    i.append('keep')
    
for i in close_pix:
    i.append('keep')
    
for i in far_pix:
    i.append('keep')
    
for i in close_mm:
    for t in far_mm:
        if t[0] == i[0] and t[2] == i[2] and t[4] == i[4]:
            if t[84] == 'drop':
                i[84] = 'drop'
    if float(i[7]) < .85:
        i[84] = 'drop'
    if ((float(i[8]) - mean_max_pitch) / stdev_pitches) > 2.5:
        i[84] = 'drop'
    if ((float(i[9]) - mean_max_yaw) / stdev_yaw) > 2.5:
        i[84] = 'drop'
    for key in part_IDs:
        if i[0] == key:
            if ((float(i[10]) - part_IDs[key][0]) / part_IDs[key][1]) > 2.5 or ((float(i[10]) - part_IDs[key][0]) / part_IDs[key][1]) < -2.5:
                i[84] = 'drop'
        
for i in far_mm:
    for t in close_mm:
        if t[0] == i[0] and t[2] == i[2] and t[4] == i[4]:
            if t[84] == 'drop':
                i[84] = 'drop'

for i in close_pix:
    for t in close_mm:
        if t[0] == i[0] and t[2] == i[2] and t[4] == i[4]:
            if t[84] == 'drop':
                i[84] = 'drop'

for i in far_pix:
    for t in close_mm:
        if t[0] == i[0] and t[2] == i[2] and t[4] == i[4]:
            if t[84] == 'drop':
                i[84] = 'drop'

#The remaining logic writes trials to be kept to our final trimmed data sheet.

header_count = 0

for i in header:
    header_count += 1
    if header_count < 83:
        outfile.write(i + ',')
    elif header_count == 84:
        outfile.write(i)

for i in close_mm:
    if i[84] == 'keep':
        count = 0
        for t in i:
            count += 1
            if count < 83:
                outfile.write(t + ',')
            elif count == 84:
                outfile.write(t)
                
for i in far_mm:
    if i[84] == 'keep':
        count = 0
        for t in i:
            count += 1
            if count < 83:
                outfile.write(t + ',')
            elif count == 84:
                outfile.write(t)

for i in close_pix:
    if i[84] == 'keep':
        count = 0
        for t in i:
            count += 1
            if count < 83:
                outfile.write(t + ',')
            elif count == 84:
                outfile.write(t)

for i in far_pix:
    if i[84] == 'keep':
        count = 0
        for t in i:
            count += 1
            if count < 83:
                outfile.write(t + ',')
            elif count == 84:
                outfile.write(t)

infile.close()
outfile.close()