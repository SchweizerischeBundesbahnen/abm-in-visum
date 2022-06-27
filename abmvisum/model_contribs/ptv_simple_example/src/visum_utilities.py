import os
import logging

from operator import itemgetter

import numpy as np

# ----------------------------------------------------------

def _get_count_chunks(count_data, chunk_size):
    return int(np.ceil(count_data / chunk_size)) if chunk_size else 1

# ----------------------------------------------------------

def SetMulti(container, attribute, values, active_only=False, chunk_size=None):
    """Set the values of an attribute for all objects in a given network object collection

    container - COM object - the network object collection (e.g. Nodes, Links)
    attribute - string - attribute ID
    values - list - new values, as many as there are objects in the collection
    activeOnly - bool - True ==> set for active objects only
    return - none

    SetMulti(Visum.Net.Nodes, "ADDVAL1", [-1, -2, -3])
    """
    if active_only:
        raw = container.GetMultiAttValues(attribute, active_only) # a bit expensive, but the only way to get the indices
        indices = map(itemgetter(0), raw)
    else:
        indices = range(1, len(values) + 1)

    if not chunk_size or chunk_size >= len(indices):
        # logging.info(f"DEBUG: Setting attribute {attribute} in single pass")
        container.SetMultiAttValues(attribute, list(zip(indices, values)))
    else:
        # logging.info(f"DEBUG: Setting attribute {attribute} in chunks")
        new_data = list(zip(indices, values))
        count_chunks = _get_count_chunks(len(new_data), chunk_size)
        for i in range(count_chunks):
            container.SetMultiAttValues(attribute, new_data[i * chunk_size:(i + 1) * chunk_size])

# ----------------------------------------------------------

def get_UDA_IDs(container):
    all_attributes = container.Attributes.GetAll
    return [attr.ID for attr in all_attributes if attr.Category.startswith("User") and attr.Editable]

def insert_UDA_if_missing(visum_container, uda_name, value_type = None):
    # starts from end (UDAs are listed last)
    is_uda_missing = True
    for current_attr in reversed(visum_container.Attributes.GetAll):
        if str(current_attr.ID).upper() == uda_name.upper():
            is_uda_missing = False
            break

    if is_uda_missing:
        visum_container.AddUserDefinedAttribute(uda_name, uda_name, uda_name, value_type if value_type is not None else 1)

    return is_uda_missing

def get_attribute_type(visum_container, attr_name):
    # starts from end (UDAs are listed last)
    for current_attr in reversed(visum_container.Attributes.GetAll):
        if str(current_attr.ID).upper() == attr_name.upper():
            return current_attr.ValueType

    raise Exception(f'Attribute {attr_name} does not exist.')

# ----------------------------------------------------------

def GetSubContainers(visum_container, chunk_size=None, reindex=None, is_uda_missing=True):
    count_net_objects = visum_container.Count

    uda_name = "ObjectIndex"

    if is_uda_missing:
        if insert_UDA_if_missing(visum_container, uda_name):
            reindex = True

    if reindex:
        index_values = range(1, count_net_objects + 1)
        SetMulti(visum_container, uda_name, index_values, chunk_size=chunk_size)  # write indices in chunks as well

    # collect sub_containers
    sub_containers = []
    count_chunks = _get_count_chunks(count_net_objects, chunk_size)
    for i in range(count_chunks):
        sub_container = visum_container.GetFilteredSet(f"[{uda_name}]>{i * chunk_size} & [{uda_name}]<={(i + 1) * chunk_size}")
        sub_containers.append(sub_container)
    return sub_containers

# ----------------------------------------------------------

def GetMulti(container, attribute, active_only=False, chunk_size=None, reindex=False, uda_missing=True):
    """Get the values of an attribute for all objects in a given network object collection

    container - COM object - the network object collection (e.g. Nodes, Links)
    attribute - string - attribute ID
    activeOnly - bool - True ==> get for active objects only
    chunks - int - None ==> chunksize
    reindex - bool - False ==> True when resetting the dummy index
    # TODO: Case when activeOnly = True and chunks = True
    return - list - new values, as many as there are objects in the collection

    values = GetMulti(Visum.Net.Nodes, "ADDVAL1")
    """
    # this is a hack, because Blocks and BlockItems have GetMultiAttValues without activeonly!
    count_net_object = container.Count
    if not chunk_size or chunk_size >= count_net_object:
        # logging.info(f"DEBUG: Reading attribute {attribute} in single pass")
        try:
            raw = container.GetMultiAttValues(attribute, active_only)
        except:
            raw = container.GetMultiAttValues(attribute)
        values = list(map(itemgetter(1), raw))
    else:
        assert not active_only, "Not implemented for active_only == True"
        values = []
        # logging.info(f"DEBUG: Reading attribute {attribute} in chunks")
        sub_containers = GetSubContainers(container, chunk_size, reindex, uda_missing)
        for sub_container in sub_containers:
            chunk_raw = sub_container.GetMultiAttValues(attribute, active_only)
            chunk_values = list(map(itemgetter(1), chunk_raw))
            values.extend(chunk_values)
    return values

# ----------------------------------------------------------

def GetMultiByFormula(container, formula, active_only=False, chunk_size=None, reindex=None, uda_missing=True):
    """Get the values of a formula evaluated on all objects in a given network object collection

    container - COM object - the network object collection (e.g. Nodes, Links)
    formula - string - formula to be evaluated (e.g. "10*[NO] + [TYPENO]")
    activeOnly - bool - True ==> get for active objects only

    return - list - new values, as many as there are objects in the collection

    values = GetMultiByFormula(Visum.Net.Nodes, "10*[NO] + [TYPENO]")
    """
    count_net_objects = container.Count
    if chunk_size is None or chunk_size >= count_net_objects:
        # logging.info(f"DEBUG: Calculating formula {formula} in single pass")
        raw = container.GetMultiByFormula(formula, active_only)
        values = list(map(itemgetter(1), raw))
    else:
        # logging.info(f"DEBUG: Calculating formula {formula} in chunks")
        values = []
        for sub_container in GetSubContainers(container, chunk_size, reindex, uda_missing):
            chunk_raw = sub_container.GetMultiByFormula(formula, active_only)
            chunk_values = list(map(itemgetter(1), chunk_raw))
            values.extend(chunk_values)
    return values

def write_net_and_demand(Visum, model_dir, project_name, step_name, only_demand):
    net_layout_file_name = os.path.join(model_dir, "Layout.net")
    demand_layout_file_name = os.path.join(model_dir, "Layout_withoutMatrixEntries.dmd")
    net_file_name = os.path.join(model_dir, r"{}_{}.net".format(project_name, step_name))
    demand_file_name = os.path.join(model_dir, r"{}_{}.dmd".format(project_name, step_name)) 
    try:
        if not only_demand:
            Visum.IO.SaveNet(net_file_name, net_layout_file_name)
        Visum.IO.SaveDemandFile(demand_file_name, False, demand_layout_file_name)
    except Exception as error:
        logging.info("Test output could not be generated: %s" % error)