
import math
import numpy as np

def inv_mass_indv(entry_data, type_entry):

    # we assume E >> m
    # so we define the invariant mass as followed :
    

    pt1entry, pt2entry = type_entry
    pt1 = entry_data[pt1entry]
    pt2 = entry_data[pt2entry]
    if pt1 != 0 and pt2 != 0:
        a = np.cosh(entry_data[pt1entry + 1] * 0.0435 - entry_data[pt2entry + 1] * 0.0435)
        b = np.cos(entry_data[pt1entry + 2] * 2 * math.pi / 144 - entry_data[pt2entry + 2] * 2 * math.pi / 144)
        invariant_mass = np.sqrt(2 * pt1 * 0.5 * pt2 * 0.5 * (a - b))
        return invariant_mass
    else:
        return None

# ET: multiply by 0.5
# eta: multiply by 0.0435
# phi: multiply by 2 * pi / 144

# due to it being L1 data!
    
def invariant_mass(dataset, type):
    result_list = []
    if type == 'jet':
        type_entry = (63, 66)
    elif type == 'muon':
        type_entry = (39, 42)
    elif type == 'egamma':
        type_entry = (3, 6)
    else:
        raise ValueError("indicate type: 'jet', 'muon' or 'egamma'")

    for entry_data in dataset[:-1]:  # assuming the last entry is not needed
        inv_mass = inv_mass_indv(entry_data, type_entry)
        if inv_mass is None or math.isinf(inv_mass):
            continue
        else:
            result_list.append(inv_mass)
    return result_list