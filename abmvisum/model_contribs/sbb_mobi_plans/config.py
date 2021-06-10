import collections

import VisumPy.helpers as VPH
import numpy as np

import abmvisum.tools.utilities as utilities
from .matrix_cache import MatrixCache


def get_UDA_IDs(container):
    all_attributes = container.Attributes.GetAll
    return [attr.ID for attr in all_attributes if attr.category.startswith("User")]


class Config(object):
    """
        Central storage of all necessary parameters. The parameters come from Visum-internal table structures (at
        the moment, parameters are defined as POI).
        Also, it has the optional possibility to add the skim matrices into a Python cache (as defined in the
        Visum Matrix structur).

        Parameters:
            Visum: Instance of the PTV Visum software.
            logging: logger as initialized in the simulator
            add_skims_to_cache: Possibility to load all skim matrices as defined in the config into the Python cache.

        Attributes:
            Visum: Instance of the PTV Visum software.
            choice_models: dictionary of all the choice models as defined in the ABM procedure
            impedance_expr = impedance expression (depending on activity type)
            logging = possibility to log Visum messages
            skim_matrices: dictionary cached skim matrices
    """
    def __init__(self, Visum, logging, add_skims_to_cache=True):
        self.Visum = Visum
        self.choice_models = collections.defaultdict(list)
        self.impedance_expr = {}
        self.logging = logging
        self.init_choice_models()

        self.skim_matrices = MatrixCache(logging)
        if add_skims_to_cache:
            self.add_skims_to_cache()

    def add_skims_to_cache(self):
        """
            Loads the skims which are needed to run SBBs' model into the cache
        """
        self.logging.info('loading all skim matrices to cache')
        self.skim_matrices.add_skim_to_cache("car_travel_times_sym", VPH.GetMatrixRaw(self.Visum, 11))
        self.skim_matrices.add_skim_to_cache("car_net_distance_sym", VPH.GetMatrixRaw(self.Visum, 12))
        self.skim_matrices.add_skim_to_cache("pt_travel_times_train_sym", VPH.GetMatrixRaw(self.Visum, 21))
        self.skim_matrices.add_skim_to_cache("pt_travel_times_bus_sym", VPH.GetMatrixRaw(self.Visum, 22))
        self.skim_matrices.add_skim_to_cache("pt_access_times_sym", VPH.GetMatrixRaw(self.Visum, 23))
        self.skim_matrices.add_skim_to_cache("pt_egress_times_sym", VPH.GetMatrixRaw(self.Visum, 24))
        self.skim_matrices.add_skim_to_cache("pt_transfers_sym", VPH.GetMatrixRaw(self.Visum, 25))
        self.skim_matrices.add_skim_to_cache("pt_frequencies_sym", VPH.GetMatrixRaw(self.Visum, 26))

        self.skim_matrices.add_skim_to_cache("at_car", np.array(VPH.GetMulti(self.Visum.Net.Zones,
                                                                             "at_car"))[np.newaxis, :])
        self.skim_matrices.add_skim_to_cache("at_ride", np.array(VPH.GetMulti(self.Visum.Net.Zones,
                                                                              "at_ride"))[np.newaxis, :])
        self.skim_matrices.add_skim_to_cache("pc_ride", np.array(VPH.GetMulti(self.Visum.Net.Zones,
                                                                              "pc_ride"))[np.newaxis, :])
        self.skim_matrices.add_skim_to_cache("pc_car", np.array(VPH.GetMulti(self.Visum.Net.Zones,
                                                                             "pc_car"))[np.newaxis, :])

    def init_choice_models(self):
        """
            1. It goes through the Visum ABM procedure table and stores every segment in self.choice_models. Every
                choice model can have multiple segments (e.g., age classes). The segments can be freely
                customized in the procedure table
            2. It saves optional impedance expressions as defined in the POI table. The expression can be functions
                of all skims available in the cache.
        """
        # get definitions of choice models with segments
        poicat_lookup = dict(self.Visum.Net.POICategories.GetMultipleAttributes(['Code', 'No']))
        attributes = ['ChoiceModel', 'Specification', 'ID', 'ResAttr', 'Filter', 'AddData', 'Active', 'MaxPlusOne',
                      'Comment']
        dstrat_poicat = poicat_lookup['DStrata']
        dstrat_pois = self.Visum.Net.POICategories.ItemByKey(dstrat_poicat).POIs
        dstrata_attrs = dstrat_pois.GetMultipleAttributes(attributes)
        # store choice model definitions
        for dstrat_attrs in dstrata_attrs:
            segment_data = dict(zip(attributes[1:], dstrat_attrs[1:]))
            segment_data['ResAttr'] = segment_data['ResAttr'][1:-1]  # [ResultAttr] -> ResultAttr
            choice_model_name = dstrat_attrs[0]
            specification_name = dstrat_attrs[1]
            segment_data['SpecificationName'] = specification_name
            try:
                segment_data['SpecificationPoiCatNo'] = poicat_lookup[specification_name]
                self.choice_models[choice_model_name].append(segment_data)
            except:
                print("No POIcat for parameters of %s" % specification_name)
        try:
            impedance_poicat = poicat_lookup['ImpedanceParams']
            impedance_pois = self.Visum.Net.POICategories.ItemByKey(impedance_poicat).POIs
            attributes = ['ActId', 'DStrat', 'FilterExpr', 'bike', 'car', 'pt', 'ride', 'walk']
            impedance_attrs = impedance_pois.GetMultipleAttributes(attributes)
            for att in impedance_attrs:
                key = (att[0], att[1])
                mode_params = []
                for i in range(3, 8):
                    mode_params.append((attributes[i], att[i]))
                if att[2] == '':
                    imped_filter_expr = None
                else:
                    imped_filter_expr = att[2]
                self.impedance_expr[key] = (imped_filter_expr, mode_params)
        except:
            self.logging.info("No impedance params defined or they are corrupt")

    def load_choice_para(self, choice_model):
        """
            Loads choice parameters for specific model types. The standard case is a discrete choice model with
            Betas that depend on a certain attribute.
        """
        model_data = self.choice_models[choice_model]

        is_destZone_choice = choice_model == "PrimLoc" or choice_model == "DestMajor" or choice_model == "SecDest"
        has_fixed_activity_per_segment = choice_model == "PrimLoc"
        is_mode_choice = (choice_model == "ModeMajor") or (choice_model == "ModeMinor")

        para = []
        for segment_data in model_data:
            para_POI = self.Visum.Net.POICategories.ItemByKey(segment_data['SpecificationPoiCatNo']).POIs
            UDA_IDs = get_UDA_IDs(para_POI)
            if len(UDA_IDs) == 0 or not "ATTREXPR" in UDA_IDs or is_destZone_choice and not "MATRIXEXPR" in UDA_IDs:
                self.logging.info("Parameters for %s corrupt and skipped." % segment_data['Specification'])
                continue
            try:
                UDA_IDs.remove("ATTREXPR")
                if 'MATRIXEXPR' in UDA_IDs:
                    UDA_IDs.remove('MATRIXEXPR')

                attr_expr = VPH.GetMulti(para_POI, 'ATTREXPR')
                comments = VPH.GetMulti(para_POI, 'Comment')

                if is_destZone_choice:
                    segment_data["AttrExpr"] = attr_expr
                    segment_data["MatrixExpr"] = VPH.GetMulti(para_POI, 'MATRIXEXPR')
                    segment_data["Beta"] = VPH.GetMulti(para_POI, 'Parameter')
                    segment_data["ODKaliFExpr"] = VPH.GetMulti(para_POI, 'ODKaliFExpr')
                    if "IPFITERATIONS" in UDA_IDs:
                        segment_data["IPFIterations"] = VPH.GetMulti(para_POI, "IPFIterations")[0]
                    else:
                        segment_data["IPFIterations"] = 0

                    segment_data['ImpedanceParams'] = None
                    if (segment_data['AddData'], segment_data['SpecificationName']) in self.impedance_expr.keys():
                        segment_data['ImpedanceParams'] = self.impedance_expr[(segment_data['AddData'],
                                                                               segment_data['SpecificationName'])][1]
                    if choice_model == "SecDest":
                        segment_data['ImpedanceParams'] = {}
                        for sec_act in utilities.build_secondary_act_dict(self.Visum).keys():
                            if (sec_act, segment_data['SpecificationName']) in self.impedance_expr.keys():
                                segment_data['ImpedanceParams'][sec_act] = self.impedance_expr[(sec_act,
                                                                                                segment_data[
                                                                                                    'SpecificationName'])][
                                    1]
                    if has_fixed_activity_per_segment:
                        # activityID = segment_data['AddData']
                        # segment_data["Attraction"] = self.activityID_to_attractionAttr[activityID]
                        segment_data["Attraction"] = VPH.GetMulti(para_POI, 'AttractionAttribute')[0]
                elif is_mode_choice:
                    beta = para_POI.GetMultipleAttributes(UDA_IDs)
                    choices = list(map(int, beta[0]))
                    beta = np.array(beta[1:])  # ignore first row that is used to define the value of each choice
                    segment_data["AttrExpr"] = attr_expr[1:]
                    segment_data["MatrixExpr"] = VPH.GetMulti(para_POI, 'MATRIXEXPR')[1:]
                    segment_data["Choices"] = choices
                    segment_data["Beta"] = beta
                    assert len(segment_data["AttrExpr"]) == len(segment_data["Beta"]) == len(segment_data["MatrixExpr"])
                else:
                    beta = para_POI.GetMultipleAttributes(UDA_IDs)
                    choices = list(map(int, beta[0]))
                    beta = np.array(beta[1:])  # ignore first row that is used to define the value of each choice
                    segment_data["AttrExpr"] = attr_expr[1:]
                    segment_data["Choices"] = choices
                    segment_data["Beta"] = beta
                    segment_data['Comments']= comments[1:]
                    assert len(segment_data["AttrExpr"]) == len(segment_data["Beta"])
                para.append(segment_data)
            except:
                self.logging.info("Parameters for %s corrupt and skipped." % segment_data['Specification'])
        return para

    def load_act_dur_para(self, choice_model):
        """
            Loads activity duration choice parameters
        """
        model_data = self.choice_models[choice_model]

        para = []
        for segment_data in model_data:
            paraPOI = self.Visum.Net.POICategories.ItemByKey(segment_data['SpecificationPoiCatNo']).POIs
            UDA_IDs = get_UDA_IDs(paraPOI)
            if len(UDA_IDs) == 0 or not "ATTREXPR" in UDA_IDs:
                print("Parameters for %s corrupt and skipped." % segment_data['Specification'])
                continue
            try:
                UDA_IDs.remove("ATTREXPR")
                segment_data["AttrExpr"] = VPH.GetMulti(paraPOI, 'ATTREXPR')
                segment_data["Subgroup_Comment"] = VPH.GetMulti(paraPOI, 'Comment')
                UDA_attrs = paraPOI.GetMultipleAttributes(UDA_IDs)
                segment_data["distribution_data"] = [dict(zip(UDA_IDs, UDA_attrs_subsegment)) for UDA_attrs_subsegment
                                                     in UDA_attrs]
                assert (len(segment_data["AttrExpr"]) == len(segment_data["distribution_data"]) == len(
                    segment_data["Subgroup_Comment"]))
                para.append(segment_data)
            except:
                print("Parameters for %s corrupt and skipped." % segment_data['Specification'])
        return para

    def load_start_times_para(self, choice_model):
        """
            Loads activity start time choice parameters
        """
        model_data = self.choice_models[choice_model]

        para = []
        for segment_data in model_data:
            para_POI = self.Visum.Net.POICategories.ItemByKey(segment_data['SpecificationPoiCatNo']).POIs
            UDA_IDs = get_UDA_IDs(para_POI)
            if len(UDA_IDs) == 0 or not "ATTREXPR" in UDA_IDs or not "TIMESERIESNO" in UDA_IDs:
                print("Parameters for %s corrupt and skipped." % segment_data['Specification'])
                continue
            try:
                data = np.array(para_POI.GetMultipleAttributes(['ATTREXPR', 'TIMESERIESNO', 'EXPBETA', 'Comment']))
                segment_data["AttrExpr"] = data[:, 0]
                segment_data["TimeSeriesNo"] = data[:, 1]
                segment_data["ExpBeta"] = data[:, 2]
                segment_data["Subgroup_Comment"] = data[:, 3]
                assert (len(segment_data["AttrExpr"]) == len(segment_data["TimeSeriesNo"]) == len(
                    segment_data["Subgroup_Comment"]))
                para.append(segment_data)
            except:
                print("Parameters for %s corrupt and skipped." % segment_data['Specification'])
        return para
