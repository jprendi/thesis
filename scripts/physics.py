
import math
import numpy as np

def inv_mass_indv(entry_data, type_entry):
    """
    Calculate the invariant mass for individual particles.

    Parameters:
        entry_data (numpy.ndarray): Array containing the data of the particle.
        type_entry (tuple): Tuple containing the indices of the PT values for two particles.

    Returns:
        float or None: The invariant mass if both PT values are non-zero, otherwise None.
    """

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
    
def invariant_mass(dataset, type, return_indices = False):
    """
    Calculate the invariant mass for particles of a specific type in the dataset.

    Parameters:
        dataset (list of numpy.ndarray): List containing arrays of particle data.
        type (str): The type of particles ('jet', 'muon', or 'egamma').

    Returns:
        return_indeces= False:
        list: List of invariant mass values for the specified type of particles.

        return_indeces= True:
        tuple: tuple[0] is the list as stated above but tuple[1] now contains a list with the indices that we used to calculate tuple[0] for
    """
    result_list = []

    if return_indices:
        indices_list = []

    if type == 'jet':
        type_entry = (63, 66)
    elif type == 'muon':
        type_entry = (39, 42)
    elif type == 'egamma':
        type_entry = (3, 6)
    else:
        raise ValueError("indicate type: 'jet', 'muon' or 'egamma'")

    for index, entry_data in enumerate(dataset[:-1]):  # assuming the last entry is not needed
        inv_mass = inv_mass_indv(entry_data, type_entry)
        if inv_mass is None or math.isinf(inv_mass):
            continue
        else:
            if return_indices:
                result_list.append(inv_mass)
                indices_list.append(index)
            else:
                result_list.append(inv_mass)
    
    if return_indices:
        return result_list, indices_list
    else:
        return result_list
