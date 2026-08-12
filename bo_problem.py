from simulator import *
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import torch

from policy_decoding import BOROUGHS, CATEGORIES, decode_policy_vector

from torch import Tensor

from typing import List, Union

import multiprocessing
from functools import partial

def simulation_loop(X,
                    obj = 'delay_75',
                    drop_cost = 100,
                    drop_by_age = False,
                    equity = "max",
                    fcfs_violation = 0.99,
                    drop_age = 100,
    ):
    decoded = decode_policy_vector(X)
    city_budget = decoded["city_budget"]
    byborough_policy = decoded["byborough_policy"]
    borough_budgets = decoded["borough_budgets"]
    inspection_policies = decoded["inspection_policies"]
    service_fractions = decoded["service_fractions"]
    drop_by_age = decoded["drop_by_age"]
    
    # run the simulation
    sim = simulator(borough_budgets = borough_budgets,
                    inspection_policies = inspection_policies,
                    city_budget= city_budget,
                    byborough_policy= byborough_policy,
                    recursive = 3,
                    date_start = '2019-01-01',
                    date_end = '2019-12-31',
                    fcfs_violation=fcfs_violation,
                    service_fractions=service_fractions,
                    drop_age=drop_age,
                    drop_using_age=drop_by_age,
                    save_logs = False,
                    sla_objective = obj,
                    drop_cost = drop_cost,
                    equity = equity,
                    save_label= 'bo')
    sim.simulate()
    if equity == "max":
        return -sim.efficiency_objectives/1000000, -sim.equity_objectives/1000000
    else:
        return -sim.efficiency_objectives/1000000, -sim.equity_objectives


class parks_simulation:
    _num_objectives = 2

    def __init__(self,
                city_budget = False,
                byborough_policy = True,
                drop_by_age = False,
                obj = 'delay_75',
                drop_cost = 100,
                drop_age = 100,
                equity = "max",
                fcfs_violation = 0.99,
    ) -> None:
        if not city_budget and not byborough_policy:
            raise ValueError('Borough-budget policies require byborough_policy=True')
        if city_budget and drop_by_age:
            raise ValueError('City-budget policies do not support drop_by_age=True')
        self.city_budget = city_budget
        self.borough_policy = byborough_policy
        self.drop_by_age = drop_by_age
        self.drop_age = drop_age
        self.equity = equity
        self.fcfs_violation = fcfs_violation
        if not city_budget:
            if drop_by_age:
                self.dim = 35
            else:
                self.dim = 65
            
            self.ref_point = [-7000000, -3000000]
            self.ref_point = torch.tensor(self.ref_point)
            self.max_hv = -10000000.0
        else:
            if byborough_policy:
                self.dim = 60
            else:
                self.dim = 12
            self.ref_point = [-7000000, -3000000]
            self.ref_point = torch.tensor(self.ref_point)
            self.max_hv = -10000000.0
        
        self.bounds = torch.tensor([[0.0]*self.dim, [1.0]*self.dim])
        if self.city_budget and self.borough_policy:
            self.bounds[1, :30] *= 0.1
        print("initialized problem with dimension ", self.dim)
        self.obj = obj
        self.drop_cost = drop_cost
        print("city budget:", self.city_budget)
        print("byborough_policy:", self.borough_policy)
        print("drop_by_age:", self.drop_by_age)
        print("drop_age:", self.drop_age)
        print("obj:", self.obj)
        print("drop_cost:", self.drop_cost)
        print("equity_objective:", self.equity)

    @staticmethod

    def __call__(self, X: Tensor) -> Tensor:
        r"""
        Evaluate the objectives on a set of points.
        """
        # we need to run the simulation in parrallel
        # we need to return the two objectives
        # segment the input into parts of dim
        X = X.numpy().flatten()
        print(X.shape)
        partial_simulation_loop = partial(simulation_loop, 
                                          obj = self.obj, 
                                          drop_cost = self.drop_cost, 
                                          drop_by_age = self.drop_by_age, 
                                          drop_age = self.drop_age,
                                          equity = self.equity,
                                          fcfs_violation = self.fcfs_violation)
        X = np.array_split(X, len(X)/self.dim)
        with multiprocessing.Pool() as pool:
            results = pool.map(partial_simulation_loop, X)
        return torch.tensor(results)

        
def simulation_loop_eval(X,
                    obj = 'delay_75',
                    drop_cost = 100,
                    drop_by_age = False,
                    date_start = '2019-01-01',
                    date_end = '2019-12-31',
                    fcfs_violation=0.99,
                    dropping_frequency=[28, 80, 26, 27, 36],
                    equity="max",
                    drop_age=100):
    decoded = decode_policy_vector(X)
    city_budget = decoded["city_budget"]
    byborough_policy = decoded["byborough_policy"]
    borough_budgets = decoded["borough_budgets"]
    inspection_policies = decoded["inspection_policies"]
    service_fractions = decoded["service_fractions"]
    drop_by_age = decoded["drop_by_age"]
    
    # run the simulation
    sim = simulator(borough_budgets = borough_budgets,
                    inspection_policies = inspection_policies,
                    city_budget= city_budget,
                    byborough_policy= byborough_policy,
                    recursive = 3,
                    date_start = date_start,
                    date_end = date_end,
                    fcfs_violation=fcfs_violation,
                    dropping_frequency=dropping_frequency,
                    service_fractions=service_fractions,
                    drop_age=drop_age,
                    drop_using_age=drop_by_age,
                    save_logs = True,
                    sla_objective = obj,
                    drop_cost = drop_cost,
                    save_label= 'bo',
                    equity = equity,)
    sim.simulate()
    # sim.srs = None # clean the srs to save memory
    return sim


def format_x(X):
    decoded = decode_policy_vector(X)
    city_budget = decoded["city_budget"]
    byborough_policy = decoded["byborough_policy"]
    borough_budgets = decoded["borough_budgets"]
    inspection_policies = decoded["inspection_policies"]
    service_fractions = decoded["service_fractions"]
    drop_by_age = decoded["drop_by_age"]

    if not city_budget:
        borough_budgets = pd.DataFrame(borough_budgets.reshape(1, -1), columns=BOROUGHS)
        print('borough_budgets:\n', borough_budgets.round(3))
    
    
    if byborough_policy:
        inspection_policies = pd.DataFrame(
            inspection_policies.T, columns=BOROUGHS, index=CATEGORIES
        )
        service_fractions = pd.DataFrame(
            service_fractions.T, columns=BOROUGHS, index=CATEGORIES
        )
    else:
        inspection_policies = pd.DataFrame(
            inspection_policies, columns=["City"], index=CATEGORIES
        )
        service_fractions = pd.DataFrame(
            service_fractions, columns=["City"], index=CATEGORIES
        )
    print('inspection_policy:\n', inspection_policies.round(3))
    print('target_service_fractions:\n', service_fractions.round(3))

    
    return {'is_city_budget': city_budget, 
            'is_byborough_policy': byborough_policy, 
            'borough_budgets': borough_budgets, 
            'inspection_policies': inspection_policies, 
            'service_fractions': service_fractions,
            'is_dropbyage': drop_by_age}


def format_df_objectives(df_objectives, delay_column = 'delay_75'):
    df_objectives = df_objectives.query('SRCategory != "Plant Tree"')
    df_objectives = df_objectives.query('BoroughCode != "Inspected"')
    # constants
    boroughs=['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']
    categories=['Hazard', 'Illegal Tree Damage', 'Other', 'Prune', 'Remove Tree',
       'Root/Sewer/Sidewalk']
    # first get the delays
    delays = pd.DataFrame(df_objectives[delay_column].values.reshape(5, 6).T, columns=boroughs, index=categories)
    print(delay_column, '\n', delays)

    # then get the actual service fractions
    frac = pd.DataFrame(df_objectives['FracInspected'].values.reshape(5, 6).T, columns=boroughs, index=categories)
    print('\nfrac_inspected\n', frac.round(2))

    # total cost each borough
    total_cost = df_objectives.groupby('BoroughCode').total_cost.sum()
    print('\ntotal cost by borough\n', total_cost.round(2))
    print('total cost ', total_cost.values.sum())

    return {
        'delays': delays,
        'fracinspected': frac,
        'total_cost': total_cost
    }

def get_shape_for_map(
        sim,
        drop_cost = 100
):
    srid_to_ct = pd.read_csv('data_clean/srid_to_ct.csv')
    srid_to_ct = srid_to_ct.dropna()
    srid_to_ct.GEOID = srid_to_ct.GEOID.astype(int).astype(str)
    ins = sim.inspected_srs.merge(srid_to_ct, on='GlobalID', how='left')
    bac = sim.backlog_srs.merge(srid_to_ct, on='GlobalID', how='left')
    dro = sim.dropped_srs.merge(srid_to_ct, on='GlobalID', how='left')
    total = sim.srs.merge(srid_to_ct, on='GlobalID', how='left')
    counts = total.groupby(['BoroughCode', 'GEOID', 'SRCategory']).GlobalID.count().reset_index().rename(columns={'GlobalID':'total'})

    # calculating the costs ...
    # first the delays
    ins['delay'] = ins['InspectionDate'] - ins['CreatedDate']


    delaydf = ins.groupby(['BoroughCode', 'GEOID', 'SRCategory']).delay.apply(np.median).reset_index()

    # calculate the overall uninspected fraction
    uninspectedcount = bac.groupby(['BoroughCode', 'GEOID', 'SRCategory']).CreatedDate.count().reset_index().rename(columns={'CreatedDate':'uninspected'})
    inspectedcount = ins.groupby(['BoroughCode', 'GEOID', 'SRCategory']).CreatedDate.count().reset_index().rename(columns={'CreatedDate':'inspected'})
    droppedcount = dro.groupby(['BoroughCode', 'GEOID', 'SRCategory']).CreatedDate.count().reset_index().rename(columns={'CreatedDate':'dropped'})
    merged = pd.merge(uninspectedcount, inspectedcount, on=['GEOID', 'SRCategory'], how='outer')
    merged = pd.merge(merged, droppedcount, on=['GEOID', 'SRCategory'], how='outer')

    merged.dropped = merged.dropped.fillna(0.1)
    merged.inspected = merged.inspected.fillna(0.1)
    merged.uninspected = merged.uninspected.fillna(0.1)
    merged['percentage_dropped'] = (merged.dropped +merged.uninspected) / (merged.dropped + merged.inspected + merged.uninspected)

    # bring in the cost data
    risk_params_file = 'data_clean/mean_risk_rating.csv'
    risk_params_df = pd.read_csv(risk_params_file)

    delaydf = delaydf.merge(risk_params_df, on=['BoroughCode', 'SRCategory'], how='outer')
    delaydf = delaydf.merge(counts, on=['BoroughCode', 'GEOID', 'SRCategory'], how='outer')
    merged = merged.merge(risk_params_df, on=['BoroughCode', 'SRCategory'], how='outer')
    merged = merged.merge(counts, on=['BoroughCode', 'GEOID', 'SRCategory'], how='outer')
    delaydf['cost'] = delaydf.delay * delaydf.RiskRating * delaydf.total
    merged['cost'] = merged.percentage_dropped * drop_cost * merged.RiskRating * merged.total
    # calculate cost by GEOID
    cost_drop = merged.groupby(['GEOID']).cost.sum().reset_index().rename(columns={'cost':'cost_drop'})
    cost_delay = delaydf.groupby(['GEOID']).cost.sum().reset_index().rename(columns={'cost':'cost_delay'})

    total_cost = cost_drop.merge(cost_delay, on='GEOID', how='outer')
    total_cost['total_cost'] = total_cost.cost_drop + total_cost.cost_delay
    # now merge the shape file and plot this on a map
    # read in the shapefile shapefile/nyct2020.shp
    shapefile = 'shapefile/nyct2020.shp'
    shape = gpd.read_file(shapefile)
    shape.GEOID = shape.GEOID.astype(str).str[0:12]
    shape = shape.to_crs({'init': 'epsg:4326'}) ## convert to lat/long

    shape = shape[['GEOID', 'geometry']]
    shape = shape.merge(total_cost, on='GEOID', how='left')
    shape.total_cost = shape.total_cost/3 # divide by 3 since we ran the simulation 3 times
    return shape
