import pickle
import numpy as np

def cut_array(array, target_length, reverse=False):
    
    if reverse:
        array = array[::-1]

    if len(array) >= target_length:
        array = array[:target_length] 
        
    if reverse:
        array = array[::-1]

    return array

 
max_len = 64 
file_path = f'/home/asugu/work/event_data_tt_pad{max_len}_reversed.pkl'

with open(file_path, 'rb') as file:
    event_data = pickle.load(file)

 

for event in event_data:
    if len(event['track_E']) > max_len:
        event['track_E'] = cut_array(event['track_E'], max_len, reverse=True)
        event['track_pt'] = cut_array(event['track_pt'], max_len, reverse=True)
        event['track_pid'] = cut_array(event['track_pid'], max_len, reverse=True)
        event['track_charge'] = cut_array(event['track_charge'], max_len, reverse=True)
        event['track_d0'] = cut_array(event['track_d0'], max_len, reverse=True)
        event['track_dz'] = cut_array(event['track_dz'], max_len, reverse=True)
        event['track_d0_sig'] = cut_array(event['track_d0_sig'], max_len, reverse=True)
        event['track_dz_sig'] = cut_array(event['track_dz_sig'], max_len, reverse=True)


for event in event_data:
    if len(event['track_E']) > max_len:
        print(len(event['track_E']))


file_path = f'/home/asugu/work/event_data_tt_pad{max_len}_reversed_modi.pkl'

with open(file_path, 'wb') as file:
    pickle.dump(event_data, file)


# Initialize a counter for counting 1's in the 'flav' parameter
"""count_ones = 0

# Iterate through the list of dictionaries
for event_dict in event_data:
    flav_value = event_dict['flav']
    count_ones += np.sum(flav_value == 1)

# Print the total count of 1's in the 'flav' parameter
print("Total number of 1's in 'flav':", count_ones)
print(len(event_data))

print(count_ones/len(event_data))
"""