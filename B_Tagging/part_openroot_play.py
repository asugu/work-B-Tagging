import uproot3 as uproot
import numpy as np
import awkward as ak
import gc

import pickle


def pad_array(array, target_length, default_value=0, reverse=False):
    
    if len(array) >= target_length:
        array = array[:target_length] 
        
    else:    
        array = np.pad(array, (0, target_length - len(array)), constant_values=default_value)
    
    if reverse:
        array = array[::-1]

    return array


root_file_path = "/home/asugu/work/part/outfile_ttbar.root"
root_file = uproot.open(root_file_path)

tree_names = root_file.keys()
print(tree_names)

tree = root_file['outtree;7']

branch_iEvent = tree['br_iEvent']
branch_btag = tree["btag"]
branch_flav = tree["flav"]

branch_track_E = tree["part_E"]
branch_track_pt = tree["track_pt"]
branch_track_px = tree["part_px"]
branch_track_py = tree["part_py"]
branch_track_pz = tree["part_pz"]

branch_track_pid = tree["track_pid"]
branch_track_charge = tree["track_charge"]
branch_track_d0 = tree["track_d0"]
branch_track_dz = tree["track_dz"]
branch_track_d0_sig = tree["track_d0_sig"]
branch_track_dz_sig = tree["track_dz_sig"]

iEvent = ak.Array(branch_iEvent.array())
jet_btag = ak.Array(branch_btag.array())
flav = ak.Array(branch_flav.array())

unsorted_track_E = ak.Array(branch_track_E.array())
unsorted_track_pt = ak.Array(branch_track_pt.array())
unsorted_track_px = ak.Array(branch_track_px.array())
unsorted_track_py = ak.Array(branch_track_py.array())
unsorted_track_pz = ak.Array(branch_track_pz.array())
unsorted_track_pid = ak.Array(branch_track_pid.array())
unsorted_track_charge = ak.Array(branch_track_charge.array())
unsorted_track_d0 = ak.Array(branch_track_d0.array())
unsorted_track_dz = ak.Array(branch_track_dz.array())
unsorted_track_d0_sig = ak.Array(branch_track_d0_sig.array())
unsorted_track_dz_sig = ak.Array(branch_track_dz_sig.array())

                          
print("hey hey")

#jet_btag_true = [1 if value == 5 or value == -5 else 0 for value in flav]

jet_btag_true = [1 if value == 5 else 0 for value in flav]

#flavor = [5 if value == 5 or value == -5 else 4 if value == 4 or value == -4 else 0 for value in flav]
flavor = [5 if value == 5 else 4 if value == 4  else 0 for value in flav]


no_ctag = 0
for i in range(len(flavor)):
    if flavor[i] == 4:
        no_ctag += 1

print(no_ctag,"number of c_tags")

print(len(jet_btag))

no_flav = 0
for i in range(len(jet_btag_true)):
    no_flav += jet_btag_true[i] 

print(no_flav,"number of b_tag_true")

sorted_indices = ak.argsort(unsorted_track_pt, axis=1, ascending=False,stable=True)


if all(len(unsorted_track_E[i]) == len(unsorted_track_pt[i]) == len(unsorted_track_pid[i]) == len(unsorted_track_charge[i]) == len(unsorted_track_d0[i]) == len(unsorted_track_d0_sig[i]) == len(unsorted_track_dz[i]) == len(unsorted_track_dz_sig[i])  for i in range(len(unsorted_track_E))):
    print("All lengths match ")
else:
    print("Lengths do not match" )

    

track_E = ak.Array([
    [unsorted_track_E[i][sorted_indices[i][j]] for j in range(len(unsorted_track_E[i]))]
    for i in range(len(unsorted_track_E))
])

track_pt = ak.Array([
    [unsorted_track_pt[i][sorted_indices[i][j]] for j in range(len(unsorted_track_pt[i]))]
    for i in range(len(unsorted_track_pt))
])

track_px = ak.Array([
    [unsorted_track_px[i][sorted_indices[i][j]] for j in range(len(unsorted_track_px[i]))]
    for i in range(len(unsorted_track_px))
])

track_py = ak.Array([
    [unsorted_track_py[i][sorted_indices[i][j]] for j in range(len(unsorted_track_py[i]))]
    for i in range(len(unsorted_track_py))
])

track_pz = ak.Array([
    [unsorted_track_pz[i][sorted_indices[i][j]] for j in range(len(unsorted_track_pz[i]))]
    for i in range(len(unsorted_track_pz))
])

track_pid = ak.Array([
    [unsorted_track_pid[i][sorted_indices[i][j]] for j in range(len(unsorted_track_pid[i]))]
    for i in range(len(unsorted_track_pid))
])

track_charge = ak.Array([
    [unsorted_track_charge[i][sorted_indices[i][j]] for j in range(len(unsorted_track_charge[i]))]
    for i in range(len(unsorted_track_charge))
])

track_d0 = ak.Array([
    [unsorted_track_d0[i][sorted_indices[i][j]] for j in range(len(unsorted_track_d0[i]))]
    for i in range(len(unsorted_track_d0))
])

track_dz = ak.Array([
    [unsorted_track_dz[i][sorted_indices[i][j]] for j in range(len(unsorted_track_dz[i]))]
    for i in range(len(unsorted_track_dz))
])

track_d0_sig = ak.Array([
    [unsorted_track_d0_sig[i][sorted_indices[i][j]] for j in range(len(unsorted_track_d0_sig[i]))]
    for i in range(len(unsorted_track_d0_sig))
])

track_dz_sig = ak.Array([
    [unsorted_track_dz_sig[i][sorted_indices[i][j]] for j in range(len(unsorted_track_dz_sig[i]))]
    for i in range(len(unsorted_track_dz_sig))
])


del unsorted_track_E, unsorted_track_pt, unsorted_track_px, unsorted_track_py, unsorted_track_pz, unsorted_track_pid, unsorted_track_charge, unsorted_track_d0, unsorted_track_dz, unsorted_track_d0_sig, unsorted_track_dz_sig
gc.collect()


jet_count = []
for i in range(len(track_E)):
    jet_count.append(len(track_E[i]))


non_empty_track_indices = [i for i, track in enumerate(track_E) if len(track) > 0]

max_track_length = max(len(track) for track in track_E)

print("max length is : ", max_track_length)

# calculation for padded to 16

max_track_length = 16

track_E = [pad_array(track, max_track_length) for track in track_E]
track_pt = [pad_array(track, max_track_length) for track in track_pt]
track_px = [pad_array(track, max_track_length) for track in track_px]
track_py = [pad_array(track, max_track_length) for track in track_py]
track_pz = [pad_array(track, max_track_length) for track in track_pz]
track_pid = [pad_array(track, max_track_length) for track in track_pid]
track_charge = [pad_array(track, max_track_length) for track in track_charge]
track_d0 = [pad_array(track, max_track_length) for track in track_d0]
track_dz = [pad_array(track, max_track_length) for track in track_dz]
track_d0_sig = [pad_array(track, max_track_length) for track in track_d0_sig]
track_dz_sig = [pad_array(track, max_track_length) for track in track_dz_sig]



event_data = []
for i in non_empty_track_indices:
    event_dict = {
        'iEvent': iEvent[i],
        'btag': jet_btag_true[i],
        'flav': flavor[i],
        'jet_count': jet_count[i],

        'track_E': np.array(track_E[i], dtype=np.float32),
        'track_pt': np.array(track_pt[i], dtype=np.float32),
        'track_px': np.array(track_px[i], dtype=np.float32),
        'track_py': np.array(track_py[i], dtype=np.float32),
        'track_pz': np.array(track_pz[i], dtype=np.float32),
        'track_pid': np.array(track_pid[i], dtype=np.float32),
        'track_charge': np.array(track_charge[i], dtype=np.float32),
        'track_d0': np.array(track_d0[i], dtype=np.float32),
        'track_dz': np.array(track_dz[i], dtype=np.float32),
        'track_d0_sig': np.array(track_d0_sig[i], dtype=np.float32),
        'track_dz_sig': np.array(track_dz_sig[i], dtype=np.float32),

    }
    event_data.append(event_dict)

file_path = 'event_data_tt_pad16_7_4mom.pkl'

with open(file_path, 'wb') as file:
    pickle.dump(event_data, file)
