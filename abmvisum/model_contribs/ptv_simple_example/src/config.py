import re
import logging
import collections
from enum import Enum

import numpy as np

import VisumPy.helpers as VPH

from src import visum_utilities

def _remove_row_indices(container, indices_to_remove):
    for index in indices_to_remove:
        del container[index]

def _remove_col_indices(container, indices_to_remove):
    for row_index, container_row in enumerate(container):
        container[row_index] = list(container_row)
        for col_index in indices_to_remove:
            del container[row_index][col_index]

class ModeImpedanceType(Enum):
    PrTShortestPathImpedance = 'PrTShortestPathImpedance'
    PuTShortestPathImpedance = 'PuTShortestPathImpedance'
    DirectDistance = 'DirectDistance'
    NoModeImpedanceType = ''


class Config(object):
    def __init__(self, Visum):
        self.Visum = Visum
        self.init_choice_models()

    def init_choice_models(self):
        # get definitions of choice models with segments
        table_names = set(name for (no,name) in self.Visum.Net.TableDefinitions.GetMultiAttValues("Name"))
        assert 'Model Steps' in table_names
        attributes = ['ChoiceModel', 'Specification', 'Specification2', 'Filter', 'AddData', 'Comment']
        msteps_entries = self.Visum.Net.TableDefinitions.ItemByKey('Model Steps').TableEntries.GetMultipleAttributes(attributes)

        # store choice model definitions
        self.choice_models = collections.defaultdict(list)
        for msteps_entry in msteps_entries:
            choice_model_name = msteps_entry[0]
            specification_name = msteps_entry[1]
            segment_data = dict(zip(attributes[1:], msteps_entry[1:]))
            if specification_name in table_names:
                self.choice_models[choice_model_name].append(segment_data)
            else:
                logging.warning(f"No table for parameters of {specification_name}")


    def load_choice_para(self, choice_model, specification_column = 'Specification'):
        model_data = self.choice_models[choice_model]

        is_destZone_choice = choice_model == "PrimLoc" or choice_model == "DestMajor" or (
            choice_model == "DestAndModeMinor" and specification_column == "Specification")
        uses_mode_choice_table = choice_model == "DestMajor" or (
            choice_model == "DestAndModeMinor" and specification_column == "Specification")
        is_mode_choice = choice_model == "ModeMajor" or (
            choice_model == "DestAndModeMinor" and specification_column == "Specification2")

        para = []
        for segment_data in model_data:
            table_name = segment_data[specification_column]
            local_segment_data = segment_data.copy()
            paras = self.Visum.Net.TableDefinitions.ItemByKey(table_name).TableEntries
            UDA_IDs = visum_utilities.get_UDA_IDs(paras)
            if len(UDA_IDs) == 0 or not "ATTREXPR" in UDA_IDs or is_destZone_choice and not "MATRIXEXPR" in UDA_IDs:
                logging.error("Parameters for %s corrupt and skipped: ATTREXPR or MATRIXEXPR missing" % table_name)
                continue
            try:
                UDA_IDs.remove("ATTREXPR")
                ids_to_remove = ['MATRIXEXPR', 'COMMENT', 'DESTINATIONIMPEDANCE', 'MODEIMPEDANCE']
                for id_to_remove in ids_to_remove:
                    if id_to_remove in UDA_IDs:
                        UDA_IDs.remove(id_to_remove)
                attr_expr = VPH.GetMulti(paras, 'ATTREXPR')

                if is_destZone_choice:
                    local_segment_data["AttrExpr"] = attr_expr
                    local_segment_data["MatrixExpr"] = VPH.GetMulti(paras, 'MATRIXEXPR')
                    if uses_mode_choice_table:
                        local_segment_data["ModeChoiceTable"] = VPH.GetMulti(paras, 'MODECHOICETABLE')
                elif is_mode_choice:
                    self.load_mode_choice_table_common(paras, table_name, UDA_IDs, attr_expr, local_segment_data)
                else:
                    choices_row_no, beta, choices = self.get_clean_choices(paras, UDA_IDs, attr_expr)

                    # ignore row choices_row_no (it is used to define the choice values)
                    del attr_expr[choices_row_no]

                    local_segment_data["AttrExpr"] = attr_expr
                    local_segment_data["Choices"] = choices
                    local_segment_data["Beta"] = np.array(beta)
                    assert len(local_segment_data["AttrExpr"]) == len(local_segment_data["Beta"])
                para.append(local_segment_data)
            except Exception as error:
                logging.error("Parameters for %s corrupt and skipped: %s" % (table_name, error))
                raise
        return para


    def parse_mode_impedances(self, table_name, mode_impedances):
        mode_impedance_types = []
        mode_impedance_definitions = []
        for mode_impedance in mode_impedances:
            # prt impedance is specified as PrTShortestPathImpedance(...)
            # put impedance is specified as PuTShortestPathImpedance(...)
            # direct distance is specified as DirectDistance()
            mode_impedance = mode_impedance.strip()
            if m := re.fullmatch(r'PrTShortestPathImpedance\s*\((.*)\)', mode_impedance, re.IGNORECASE):
                # the value inside the parentheses in PrTShortestPathImpedance(...) is the prt impedance definition
                mode_impedance_types.append(ModeImpedanceType.PrTShortestPathImpedance)
                mode_impedance_definitions.append(m.group(1).strip())
            elif m := re.fullmatch(r'PuTShortestPathImpedance\s*\((.*)\)', mode_impedance, re.IGNORECASE):
                # the value inside the parentheses in PuTShortestPathImpedance(...) is the put demand segment
                mode_impedance_types.append(ModeImpedanceType.PuTShortestPathImpedance)
                mode_impedance_definitions.append(m.group(1).strip())
            elif re.fullmatch(r'DirectDistance\s*\(\s*\)', mode_impedance, re.IGNORECASE):
                # direct distance
                mode_impedance_types.append(ModeImpedanceType.DirectDistance)
                mode_impedance_definitions.append('')
            else:
                # empty
                if mode_impedance:
                    logging.warning(f"Parameter '{mode_impedance}' for mode impedance definition in table {table_name} is corrupt.")
                mode_impedance_types.append(ModeImpedanceType.NoModeImpedanceType)
                mode_impedance_definitions.append('')

        return mode_impedance_types, mode_impedance_definitions

    def get_mode_impedance_prtsyss(self, table_name, mode_ids, betas, mode_impedance_types):
        prtsyss = [None] * len(mode_impedance_types)

        mode_id_to_tsys = dict(self.Visum.Net.Modes.GetMultipleAttributes(["ID", "First:TSyss\\Code"]))
        for row_index, (betas_cur_term, mode_impedance_type) in enumerate(zip(betas, mode_impedance_types)):
            if mode_impedance_type != ModeImpedanceType.PrTShortestPathImpedance:
                continue

            # for prt impedance there must be exactly one mode with coefficient < 0
            has_errors = False
            if len(betas_cur_term[betas_cur_term != 0]) != 1:
                logging.warning(f"Table {table_name}: There must be exactly one mode for which the coefficient is != 0.")
                has_errors = True
            elif len(betas_cur_term[betas_cur_term < 0]) != 1:
                has_errors = True
                logging.warning(f"Table {table_name}: The mode coefficient must be < 0.")

            if not has_errors:
                # mode with coefficient < 0 determines tsys for prt shortest path search
                prt_mode_ind = np.argmin(betas_cur_term)
                prt_mode_id = mode_ids[prt_mode_ind]
                prtsyss[row_index] = self.Visum.Net.TSystems.ItemByKey(mode_id_to_tsys[prt_mode_id])
            else:
                # error => do not use any impedance for this term
                mode_impedance_types[row_index] = ModeImpedanceType.NoModeImpedanceType

        return prtsyss


    def load_mode_choice_table_common(self, table_entries, table_name, UDA_IDs, attr_expr, segment_data = None):
        if segment_data is None:
            segment_data = collections.defaultdict(list)

        choices_row_index, beta, choices = self.get_clean_choices(table_entries, UDA_IDs, attr_expr)

        segment_data["MatrixExpr"] = VPH.GetMulti(table_entries, 'MATRIXEXPR')
        segment_data["DestinationImpedance"] = VPH.GetMulti(table_entries, 'DESTINATIONIMPEDANCE')
        mode_impedances = VPH.GetMulti(table_entries, 'MODEIMPEDANCE')

        # ignore choices row
        for cur_list in [attr_expr, segment_data["MatrixExpr"], segment_data["DestinationImpedance"], mode_impedances]:
            del cur_list[choices_row_index]

        segment_data["AttrExpr"] = attr_expr
        segment_data["Choices"] = choices
        segment_data["Beta"] = np.array(beta) 
        assert len(segment_data["AttrExpr"]) == len(segment_data["Beta"]) == len(segment_data["MatrixExpr"])    

        mode_impedance_types, mode_impedance_definitions = self.parse_mode_impedances(table_name, mode_impedances)
        segment_data["ModeImpedanceTypes"] = mode_impedance_types
        segment_data["ModeImpedanceDefinitions"] = mode_impedance_definitions

        prtsyss = self.get_mode_impedance_prtsyss(table_name, choices, segment_data["Beta"], mode_impedance_types)
        segment_data["ModeImpedancePrTSys"] = prtsyss

        return segment_data


    def get_clean_choices(self, table_entries, UDA_IDs, attr_expr):
        """
        Retrieve only relevant columns and rows.
        - choices row is identified by attr_expr="ChoiceID"
        - choices row is removed from rows
        - user columns are removed as well (columns with empty value in choices row)
        """
        choices_row_index = attr_expr.index("ChoiceID")
        beta = list(table_entries.GetMultipleAttributes(UDA_IDs))
        choices_row = beta[choices_row_index]
        del beta[choices_row_index]

        # find and remove user columns from UDA_IDs, which are not part of the model definition
        # (user columns are identified by not having any value in the choices row)
        user_UDA_columns = []
        for column_index, choice_value in enumerate(choices_row):
            if  choice_value is None:
                user_UDA_columns.append(column_index)

        choices = list(choices_row)
        if user_UDA_columns:
            _remove_row_indices(choices, user_UDA_columns)
            _remove_col_indices(beta, user_UDA_columns)
        choices = list(map(int,choices))

        return choices_row_index, beta, choices


    def load_act_dur_para(self, choice_model):
        model_data = self.choice_models[choice_model]

        para = []
        for segment_data in model_data:
            paras = self.Visum.Net.TableDefinitions.ItemByKey(segment_data['Specification']).TableEntries 
            UDA_IDs = visum_utilities.get_UDA_IDs(paras)
            if len(UDA_IDs) == 0 or not "ATTREXPR" in UDA_IDs:
                logging.error("Parameters for %s corrupt and skipped: ATTREXPR missing" % segment_data['Specification'])
                continue
            try:
                UDA_IDs.remove("ATTREXPR")
                segment_data["AttrExpr"] = VPH.GetMulti(paras, 'ATTREXPR')
                segment_data["Subgroup_Comment"] = VPH.GetMulti(paras, 'Comment') 
                UDA_attrs = paras.GetMultipleAttributes(UDA_IDs)
                segment_data["distribution_data"] = [dict(zip(UDA_IDs, UDA_attrs_subsegment)) for UDA_attrs_subsegment in UDA_attrs]
                assert (len(segment_data["AttrExpr"]) == len(segment_data["distribution_data"]) == len(segment_data["Subgroup_Comment"]))
                para.append(segment_data)
            except Exception as error:
                logging.error("Parameters for %s corrupt and skipped: %s" %(segment_data['Specification'], error))
        return para


    def load_start_times_para(self, choice_model):
        model_data = self.choice_models[choice_model]

        para = []
        for segment_data in model_data:
            paras = self.Visum.Net.TableDefinitions.ItemByKey(segment_data['Specification']).TableEntries 
            UDA_IDs = visum_utilities.get_UDA_IDs(paras)
            if len(UDA_IDs) == 0 or not "ATTREXPR" in UDA_IDs or not "TIMESERIESNO" in UDA_IDs:
                logging.error("Parameters for %s corrupt and skipped: ATTREXPR or TIMESERIESNO missing" % segment_data['Specification'])
                continue
            try:
                segment_data["AttrExpr"] = VPH.GetMulti(paras, 'ATTREXPR')
                segment_data["TimeSeriesNo"] = VPH.GetMulti(paras, 'TIMESERIESNO')
                segment_data["Subgroup_Comment"] = VPH.GetMulti(paras, 'Comment')
                assert (len(segment_data["AttrExpr"]) == len(segment_data["TimeSeriesNo"]) == len(segment_data["Subgroup_Comment"]))
                para.append(segment_data)
            except Exception as error:
                logging.error("Parameters for %s corrupt and skipped: %s" % (segment_data['Specification'], error))
                raise
        return para

    def load_mode_choice_table(self, table_name):
        paras = self.Visum.Net.TableDefinitions.ItemByKey(table_name).TableEntries 
        UDA_IDs = visum_utilities.get_UDA_IDs(paras)
        if len(UDA_IDs) == 0 or not "ATTREXPR" in UDA_IDs or not "MATRIXEXPR" in UDA_IDs or not "DESTINATIONIMPEDANCE" in UDA_IDs or not "MODEIMPEDANCE" in UDA_IDs:
            logging.error("Parameters for %s corrupt and skipped: ATTREXPR, MATRIXEXPR, DESTINATIONIMPEDANCE or MODEIMPEDANCE missing" % table_name)
        try:
            # these attributes will be parsed separately in load_mode_choice_table_common()
            UDA_IDs.remove("ATTREXPR")
            UDA_IDs.remove('MATRIXEXPR')
            UDA_IDs.remove('DESTINATIONIMPEDANCE')
            UDA_IDs.remove('MODEIMPEDANCE')

            if 'COMMENT' in UDA_IDs:
                UDA_IDs.remove('COMMENT')

        except Exception as error:
            logging.error("Parameters for %s corrupt and skipped: %s" %(table_name, error))
            raise

        attr_expr = VPH.GetMulti(paras, 'ATTREXPR')
        return self.load_mode_choice_table_common(paras, table_name, UDA_IDs, attr_expr)
