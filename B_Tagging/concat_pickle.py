import pickle

pad = 64
rev = ''#'_reversed'

with open(f'/home/asugu/work/event_data_tt_pad{pad}_6_4mom{rev}.pkl', 'rb') as f1:
    data1 = pickle.load(f1)

with open(f'/home/asugu/work/event_data_tt_pad{pad}_7_4mom{rev}.pkl', 'rb') as f2:
    data2 = pickle.load(f2)


concatenated_data = data1 + data2

with open(f'event_data_tt_pad{pad}_4mom{rev}.pkl', 'wb') as f:
    pickle.dump(concatenated_data, f)
