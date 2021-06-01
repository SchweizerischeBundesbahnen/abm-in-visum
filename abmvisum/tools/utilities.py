import logging
import tempfile
from operator import itemgetter

import numpy as np
import pandas as pd

try:
    import VisumPy.helpers as VPH
except ModuleNotFoundError as e:
    pass


def SetMulti(container, attribute, values, activeOnly=False, chunks=None):
    """Set the values of an attribute for all objects in a given network object collection

    container - COM object - the network object collection (e.g. Nodes, Links)
    attribute - string - attribute ID
    values - list - new values, as many as there are objects in the collection
    activeOnly - bool - True ==> set for active objects only
    return - none

    SetMulti(Visum.Net.Nodes, "ADDVAL1", [-1, -2, -3])
    """
    if activeOnly:
        raw = container.GetMultiAttValues(attribute, activeOnly)  # a bit expensive, but the only way to get the indices
        indices = map(itemgetter(0), raw)
    else:
        indices = range(1, len(values) + 1)

    count_netobject = container.Count
    if chunks is None or chunks >= count_netobject:
        # logging.info(f"DEBUG: Setting attribute {attribute} in single pass")
        container.SetMultiAttValues(attribute, list(zip(indices, values)))
    else:
        # logging.info(f"DEBUG: Setting attribute {attribute} in chunks")
        # chunks is the number of objects for which the attribute value should be set in a single call to SetMultiAttValues
        newdata = list(zip(indices, values))
        for i in range(0, len(values), chunks):
            container.SetMultiAttValues(attribute, newdata[i:min(chunks + i, len(values))])


# ----------------------------------------------------------

def GetSubContainers(container, chunks=None, reindex=None, uda_missing=True):
    count_netobject = container.Count

    uda_name = "ObjectIndex"
    # Modified from VPH, starts from the end
    for anAttr in reversed(container.Attributes.GetAll):
        if str(anAttr.ID).upper() == uda_name.upper():
            uda_missing = False
            break

    if uda_missing:
        container.AddUserDefinedAttribute(uda_name, uda_name, uda_name, 1)
        reindex = True

    if reindex:
        index_values = range(1, count_netobject + 1)
        SetMulti(container, uda_name, index_values, chunks=chunks)  # write indices in chunks as well

    # collect subcontainers
    subcontainers = []
    for i in range(int(np.ceil(count_netobject / chunks))):
        subcontainer = container.GetFilteredSet(f"[{uda_name}]>{i * chunks} & [{uda_name}]<={(i + 1) * chunks}")
        subcontainers.append(subcontainer)
    return subcontainers


# ----------------------------------------------------------

def GetMulti(container, attribute, activeOnly=False, chunks=None, reindex=False, uda_missing=True):
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
    count_netobject = container.Count
    if chunks is None or chunks >= count_netobject:
        # logging.info(f"DEBUG: Reading attribute {attribute} in single pass")
        try:
            raw = container.GetMultiAttValues(attribute, activeOnly)
        except:
            raw = container.GetMultiAttValues(attribute)
        values = list(map(itemgetter(1), raw))
    else:
        values = []
        # logging.info(f"DEBUG: Reading attribute {attribute} in chunks")
        for subcontainer in GetSubContainers(container, chunks, reindex, uda_missing):
            chunk_raw = subcontainer.GetMultiAttValues(attribute, activeOnly)
            chunk_values = list(map(itemgetter(1), chunk_raw))
            values.extend(chunk_values)
    return values  # todo: integer conversion


# ----------------------------------------------------------

def GetMultiByFormula(container, formula, activeOnly=False, chunks=None, reindex=None, uda_missing=True):
    """Get the values of a formula evaluated on all objects in a given network object collection

    container - COM object - the network object collection (e.g. Nodes, Links)
    formula - string - formula to be evaluated (e.g. "10*[NO] + [TYPENO]")
    activeOnly - bool - True ==> get for active objects only

    return - list - new values, as many as there are objects in the collection

    values = GetMultiByFormula(Visum.Net.Nodes, "10*[NO] + [TYPENO]")
    """
    count_netobject = container.Count
    if chunks is None or chunks >= count_netobject:
        # logging.info(f"DEBUG: Calculating formula {formula} in single pass")
        raw = container.GetMultiByFormula(formula, activeOnly)
        values = list(map(itemgetter(1), raw))
    else:
        # logging.info(f"DEBUG: Calculating formula {formula} in chunks")
        values = []
        for subcontainer in GetSubContainers(container, chunks, reindex, uda_missing):
            chunk_raw = subcontainer.GetMultiByFormula(formula, activeOnly)
            chunk_values = list(map(itemgetter(1), chunk_raw))
            values.extend(chunk_values)
    return values


# ----------------------------------------------------------

def start_logging():
    logger = logging.getLogger()
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# ----------------------------------------------------------

# evaluate attribute expressions
def eval_attrexpr(container, attrexpr, np_dtype=np.float):
    '''
    container = a Visum net object container
    attrexpr = a list of expressions, containing attributes of container as variables. Attribute IDs must be enclosed in []

    returns a np.array of shape (# of objects in container, # of expressions in attrexpr) filled with the results of the expressions

    Example:
    attrexpr = ['[NO]', '[NO]+[empl_pct]']
    container = Visum.Net.POICategories.ItemByKey(6).POIs
    res = eval_attrexpr(container, attrexpr)

    res[:10] =
    array([[5.56068163e+00, 2.60000000e+02],
           [5.56452041e+00, 2.61000000e+02],
           [5.92958914e+00, 3.76000000e+02],
           [7.59186171e+00, 1.98200000e+03],
           [7.59337419e+00, 1.98500000e+03],
           [7.60489448e+00, 2.00800000e+03],
           [7.70796153e+00, 2.22600000e+03],
           [7.72797554e+00, 2.27100000e+03],
           [7.73761628e+00, 2.29300000e+03],
           [7.78821156e+00, 2.41200000e+03]])
    '''
    num_exprs = len(attrexpr)
    num_objects = container.Count

    if np.all(attrexpr == "1"):
        return np.ones((num_objects, num_exprs), dtype=np_dtype)

    results = np.empty((num_objects, num_exprs), dtype=np_dtype)

    for i, expr in enumerate(attrexpr):
        if expr == "1":
            results[:, i] = 1
        else:
            results[:, i] = VPH.GetMultiByFormula(container, expr)
    return results


# evaluate matrix expressions
def eval_matexprs(Visum, matrix_expressions):
    nZones = Visum.Net.Zones.Count
    result = np.empty((len(matrix_expressions), nZones, nZones), dtype=np.float)

    helper_matrix_no = 99999  # temporary result matrix for matrix formulas

    helper_matrix_exists = helper_matrix_no in VPH.GetMulti(Visum.Net.Matrices, "NO")
    if helper_matrix_exists:
        helper_matrix = Visum.Net.Matrices.ItemByKey(helper_matrix_no)
    else:
        helper_matrix = Visum.Net.AddMatrix(helper_matrix_no)

    for i in range(len(matrix_expressions)):
        success = helper_matrix.SetValuesToResultOfFormula(matrix_expressions[i])
        assert success, "matrix expression invalid: " + matrix_expressions[i]
        result[i] = VPH.GetMatrixRaw(Visum, helper_matrix_no)

    return result


# evaluate matrix expressions
def eval_matexprs_cached_skims(Visum, matrix_expressions, skims):
    nZones = Visum.Net.Zones.Count
    result = np.empty((1, nZones, nZones), dtype=np.float)

    for mode, expr in matrix_expressions:
        if expr != '-9999':
            result[0, :, :] += calc_utils_for_matr_expr(skims, expr)
    return result


def calc_utils_for_matr_expr(skims, expr):
    _result = eval('lambda matrix_11, matrix_12, matrix_21, matrix_22, matrix_23, matrix_24, matrix_25, '
                   'matrix_26, from_acc_car, to_acc_car, from_acc_ride, to_acc_ride, to_pc_car, to_pc_ride: '
                   + expr,
                   {'min': np.minimum, 'max': np.maximum})
    result = np.exp(_result(skims.get_skim('car_travel_times_sym'),
                            skims.get_skim('car_net_distance_sym'),
                            skims.get_skim('pt_travel_times_train_sym'),
                            skims.get_skim('pt_travel_times_bus_sym'),
                            skims.get_skim('pt_access_times_sym'),
                            skims.get_skim('pt_egress_times_sym'),
                            skims.get_skim('pt_transfers_sym'),
                            skims.get_skim('pt_frequencies_sym'),
                            skims.get_skim('at_car').T,
                            skims.get_skim('at_car'),
                            skims.get_skim('at_ride').T,
                            skims.get_skim('at_ride'),
                            skims.get_skim('pc_car'),
                            skims.get_skim('pc_ride')))
    return result


# ----------------------------------------------------------


def get_filtered_subjects(subjects, filterExpr):
    return subjects.GetFilteredSet(filterExpr) if filterExpr != "" else subjects


def build_activity_dict(Visum):
    return dict(Visum.Net.Activities.GetMultipleAttributes(["Id", "Code"]))


def build_secondary_act_dict(Visum):
    act_array = np.array(
        Visum.Net.Activities.GetMultipleAttributes(["Id", "Code", "IsHomeActivity", "lt_loc_distance_attr"]))
    sec_act_dict = {}
    for act in act_array:
        if (float(act[2]) == 0.0) & (len(act[3]) == 0):
            sec_act_dict[float(act[0])] = act[1]
    return sec_act_dict


def build_activityLocation_dict(Visum):
    activityLocation_dict = dict()
    for zone_no, act_code, activityLocation_key in Visum.Net.ActivityLocations.GetMultipleAttributes(
            ["Location\\ZoneNo", "ActivityCode", "Key"]):
        activityLocation_dict[(zone_no, act_code)] = activityLocation_key
    return activityLocation_dict


def build_zoneNoActCode_to_locationNo_dict(Visum):
    activityLocation_dict = dict()
    for zoneNo, act_code, location_no in Visum.Net.ActivityLocations.GetMultipleAttributes(
            ["Location\\ZoneNo", "ActivityCode", "LocationNo"]):
        activityLocation_dict[(zoneNo, act_code)] = int(location_no)
    return activityLocation_dict


def build_activityID_to_attractionAttr_dict(Visum):
    return dict(Visum.Net.Activities.GetFilteredSet("[Id]>0").GetMultipleAttributes(['Id', 'attraction_attribute']))


def zoneNos_to_zoneIndices(zone_nos, zoneNo_to_zoneInd):
    assert np.all(zone_nos > 0)  # zone nos must be > 0
    # convert zoneNo -> index
    zone_indices = np.empty(len(zone_nos), dtype=int)
    for i in range(len(zone_nos)):
        zone_indices[i] = zoneNo_to_zoneInd[zone_nos[i]]
    return zone_indices


def get_zone_indices(subjects, zone_no_attr_id, zoneNo_to_zoneInd, chunks=None):
    zone_nos = np.array(GetMulti(subjects, zone_no_attr_id, chunks=chunks), dtype=int)
    return zoneNos_to_zoneIndices(zone_nos, zoneNo_to_zoneInd)


# returns list containing the index of each value in discrete_values
# i.e. returns value_indices s.t. discrete_values_list[value_indices[i]] == values[i]
def get_value_indices(discrete_values_list, values):
    discrete_value_to_index = dict()
    for i in range(len(discrete_values_list)):
        discrete_value_to_index[discrete_values_list[i]] = i

    value_indices = [discrete_value_to_index[val] for val in values]
    return value_indices


def od_to_demand_mat(Visum, mat_no, mat_name, visum_obj, origin_col, dest_col):
    zones = np.array(Visum.Net.Zones.GetMultiAttValues('No'), dtype=int)[:, 1]
    matrices_visum = [i.AttValue('No') for i in Visum.Net.Matrices.GetAll]

    if mat_no in matrices_visum:
        print('Matrix already exists. Overwriting values...')
    else:
        mat_visum = Visum.Net.AddMatrix(mat_no, 2, 3)
        mat_visum.SetAttValue("Name", mat_name)
        mat_visum.SetAttValue("Code", mat_name)

    o_col = "o_col"
    d_col = "d_col"
    df = pd.DataFrame(visum_obj.GetMultipleAttributes([origin_col, dest_col]), columns=[o_col, d_col])
    df = df.loc[(df[o_col] != 0) & (df[d_col] != 0)]

    zoneNo_to_zoneInd = dict(zip(zones, range(len(zones))))
    df.replace({o_col: zoneNo_to_zoneInd}, inplace=True)
    df.replace({d_col: zoneNo_to_zoneInd}, inplace=True)

    od_mat = np.zeros((len(zones), len(zones)))
    for entry in df[[o_col, d_col]].values:
        od_mat[int(entry[0]), int(entry[1])] = od_mat[int(entry[0]), int(entry[1])] + 1

    VPH.SetMatrixRaw(Visum, mat_no, od_mat)


# todo I hope we can get rid of that soon...
def init_schedules(visum, filtered_persons, logging):
    if filtered_persons.Count == 0:
        return

    logging.info('initializing %d schedules' % filtered_persons.Count)

    tra_file_header = """$VISION
* Schweizerische Bundesbahnen SBB Personenverkehr Bern
* 03/06/20
* From: x.ver
* To:   y.ver
* 
* Table: Version block
* 
$VERSION:VERSNR;FILETYPE;LANGUAGE
11;Trans;ENG

* 
* Table: Schedules (deleted)
* 
$-SCHEDULE:PERSONNO;NO
"""
    keys = np.array(filtered_persons.GetMultipleAttributes(['No', 'No']))

    f = tempfile.NamedTemporaryFile(mode='w', suffix=".tra", delete=False, newline='')
    f.write(tra_file_header)
    keys__ = np.copy(keys)
    keys__[:, 1] = 1
    keys__ = keys__.astype(int)
    pd.DataFrame(keys__).to_csv(f, sep=";", header=False, index=False, encoding='utf-8')
    f.close()

    visum.ApplyModelTransferFile(f.name)
    keys[:, 0] = 1
    visum.Net.AddMultiSchedules(keys)


def set_out_of_home_time(persons):
    if persons.Count == 0:
        return

    tot_perf_time = np.array(GetMulti(persons, r"Sum:Schedules\Sum:Tours\Sum:Trips\ToActivityExecution\Duration"))
    tot_perf_time[tot_perf_time == None] = 0.0
    tot_perf_time = tot_perf_time.astype(float)

    tot_travel_time = np.array(GetMulti(persons, r"Sum:Schedules\Sum:Tours\Sum:Trips\Duration"))
    tot_travel_time[tot_travel_time == None] = 0.0
    tot_travel_time = tot_travel_time.astype(float)

    out_of_home_time = (tot_perf_time + tot_travel_time) / 3600.0
    SetMulti(persons, "out_of_home_time", out_of_home_time)
