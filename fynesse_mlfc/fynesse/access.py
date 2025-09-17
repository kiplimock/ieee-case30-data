from .config import *
import osmnx as ox
import matplotlib.pyplot as plt
import math


import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def plot_city_map(place_name, latitude, longitude, box_size_km=2, poi_tags=None):
    """
    Plot a simple city map with area boundary, buildings, roads, nodes, and optional POIs.

    Parameters
    ----------
    place_name : str
        Name of the place (used for boundary + plot title).
    latitude, longitude : float
        Central coordinates.
    box_size_km : float
        Size of the bounding box in kilometers (default 2 km).
    poi_tags : dict, optional
        Tags dict for POIs (e.g. {"amenity": ["school", "restaurant"]}).
    """

    # Convert km to degrees
    lat_offset = (box_size_km / 2) / 111
    lon_offset = (box_size_km / 2) / (111 * math.cos(math.radians(latitude)))

    north = latitude + lat_offset
    south = latitude - lat_offset
    east = longitude + lon_offset
    west = longitude - lon_offset
    bbox = (west, south, east, north)

    # Area boundary
    area = ox.geocode_to_gdf(place_name).to_crs(epsg=4326)

    # Road graph
    graph = ox.graph_from_bbox(bbox, network_type="all")
    nodes, edges = ox.graph_to_gdfs(graph)

    # Buildings & POIs
    buildings = ox.features_from_bbox(bbox, tags={"building": True})
    pois = None
    if poi_tags:
        pois = ox.features_from_bbox(bbox, tags=poi_tags)

    # Ensure correct geometry column
    nodes = nodes.set_geometry("geometry")
    edges = edges.set_geometry("geometry")
    buildings = buildings.set_geometry("geometry")
    if pois is not None:
        pois = pois.set_geometry("geometry")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    area.plot(ax=ax, color="tan", alpha=0.5)
    if not buildings.empty:
        buildings.plot(ax=ax, facecolor="gray", edgecolor="gray", linewidth=0.5)
    edges.plot(ax=ax, color="black", linewidth=1, alpha=0.3, column=None)
    nodes.plot(ax=ax, color="black", markersize=1, alpha=0.3, column=None)
    if pois is not None and not pois.empty:
        pois.plot(ax=ax, color="green", markersize=5, alpha=1, column=None)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_title(place_name, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()



def node_standardise(y0, y1, y2, *, nodewise=True):
    if nodewise:
        mu  = y0.mean(0)
        std = y0.std(0)
        std[std == 0] = 1
    else:
        mu, std = [], []
        for i in range(y0.shape[-1]):
            mu.append(y0[...,i].mean())
            tmp = y0[...,i].std() if y0[...,i].std() != 0 else 1
            std.append(tmp)
        mu  = np.array(mu).reshape(1, y0.shape[-1])
        std = np.array(std).reshape(1, y0.shape[-1])

    ny0 = (y0 - np.resize(mu, y0.shape))/np.resize(std, y0.shape)
    ny1 = (y1 - np.resize(mu, y1.shape))/np.resize(std, y1.shape)
    ny2 = (y2 - np.resize(mu, y2.shape))/np.resize(std, y2.shape)
    return ny0, ny1, ny2, mu, std


def extract_data(
    data_path = "matpower_format",
    seed=1234,
    *,
    remove_duplicates=True,
    nodewise=False,
    standardise_outputs=True,
):
    Ybus_data = io.loadmat(data_path + '/grids_Ybus_batch1')['y_array']
    mpc_data  = io.loadmat(data_path + '/grids_pre_soln_batch1')['mpc_array']
    acopf_data = io.loadmat(data_path + '/grids_post_acopf_batch1')['res_acpf_array']
    dcopf_data = io.loadmat(data_path + '/grids_post_dcopf_batch1')['res_dcpf_array']

    # A stores adj. information, B stores grid info. for power analysis, X stores inputs, Y stores outputs
    A, X, Y, B = [], [], [], []
    N = mpc_data.shape[1]

    for i in range(N):
        # Extract relevant bus, generator and branch features
        bus_data = mpc_data[0, i]['bus'][0, 0]
        branch_data = mpc_data[0, i]['branch'][0, 0]
        gen_cost_data = mpc_data[0, i]['gencost'][0,0]

        # Base power value
        base_mva = mpc_data[0, i]['baseMVA'][0, 0][0][0]

        bus_solns = acopf_data[0, i]['bus'][0, 0]
        gen_data = mpc_data[0, i]['gen'][0, 0]

        # Extract Yb, Yf, Yt
        Yb, Yf, Yt = Ybus_data[0, i], Ybus_data[1, i], Ybus_data[2, i],

        #branch_dcopf_solns = dcopf_data[0, i]['branch'][0, 0]
        bus_dcopf_solns = dcopf_data[0, i]['bus'][0, 0]
        gen_dcopf_solns = dcopf_data[0, i]['gen'][0, 0]
        gen_dcopf_solns[:,[1, 2, 3, 4, 8, 9]] /= base_mva

        # Convert all power quantities to per unit
        bus_data[:, [2, 3, 4, 5]] /= base_mva
        gen_data[:, [1, 2, 3, 4, 8, 9]] /= base_mva

        # Number of buses
        n_buses = bus_data.shape[0]

        ## Construct Undirected Sparse Adj.
        adj = np.zeros((n_buses,n_buses), dtype=np.int8)
        # change from 1-indexing to 0-indexing
        row_indx = branch_data[:,0] - 1
        col_indx = branch_data[:,1] - 1
        # collect line status from branch data
        line_status = branch_data[:,10]
        adj[row_indx.astype(int), col_indx.astype(int)] = line_status.astype(np.int8)
        adj[col_indx.astype(int), row_indx.astype(int)] = line_status.astype(np.int8)
        sparse_adj = np.array(adj.nonzero())#.t().contiguous()

        Pd = bus_data[:,2] # real power demand
        Qd = bus_data[:,3] # reactive power demand
        Gs = bus_data[:,4] # shunt conductance
        Bs = bus_data[:,5] # shunt susceptance

        Pg = np.zeros(n_buses) # real power generation
        Qg = np.zeros(n_buses) # reactive power generation

        active_gens = gen_data[:,7] > 0
        gen_indx = gen_data[active_gens,0] - 1

        ## rewrite for if/when gen info varies between samples
        Pg[gen_indx.astype(int)] = acopf_data[0, i]['gen'][0, 0][:,1]/base_mva
        Qg[gen_indx.astype(int)] = acopf_data[0, i]['gen'][0, 0][:,2]/base_mva

        ## Output Target Va and Vm
        Vm = bus_solns[:, 7]
        Va = (bus_solns[:, 8]/180) * np.pi

        complex_v = np.multiply(Vm, np.exp(1j * Va))
        I_conj = Yb.dot(complex_v).conj()

        Sg = Pg + 1j*Qg # reactive power generation
        mag_Sg = np.absolute(Sg)
        ang_Sg = np.angle(Sg)

        Sd = Pd + 1j*Qd # reactive power demand
        mag_Sd = np.absolute(Sd)
        ang_Sd = np.angle(Sd)

        feat_list = [
            Pd,
            Qd,
            mag_Sd,
            ang_Sd,
            #Bs,
            #Gs,
            ]
        num_feat = len(feat_list)
        inp = np.zeros((n_buses,num_feat))
        for k in range(num_feat):
            inp[:,k] = feat_list[k]

        net_S = Sg - Sd

        # possible output targets Voltage, Conjugate Current, Generator Power Output, Net Nodal Injection
        # in polar and rectangular form focus in this notebook on voltage in polar form
        targ_list = [
            Vm,
            Va,
            #np.real(complex_v),
            #np.imag(complex_v),
            #np.abs(I_conj),
            #np.angle(I_conj),
            #np.real(I_conj),
            #np.imag(I_conj),
            #mag_Sg,
            #ang_Sg,
            #Pg,
            #Qg,
            #np.abs(net_S),
            #np.angle(net_S),
            #np.real(net_S),
            #np.imag(net_S),
        ]

        num_targ = len(targ_list)
        out = np.zeros((n_buses,num_targ))
        for k in range(num_targ):
            out[:,k] = targ_list[k]


        grid_info = {
            "gencost" : gen_cost_data,
            "gen_indx" : gen_indx,
            "Pg" : acopf_data[0, i]['gen'][0, 0][:,1],
            "Qg" : acopf_data[0, i]['gen'][0, 0][:,2],
            "MAX_Pg" : gen_data[:,8],
            "MIN_Pg" : gen_data[:,9],
            "MAX_Qg" : gen_data[:,3],
            "MIN_Qg" : gen_data[:,4],
            "f" : acopf_data[0, i]['f'][0,0],
            "branch" : branch_data[:,[0,1,5,11,12]],
            "V-dcopf" : bus_dcopf_solns[:,[7,8]],
            "S-dcopf" : gen_dcopf_solns[:,[1,2]],
            "f-dcopf" : dcopf_data[0, i]['f'][0,0],
            "Ybus" : Yb,
            "Yt" : Yt,
            "Yf" : Yf
        }

        A.append(sparse_adj)
        X.append(inp)
        Y.append(out)
        B.append(grid_info)

    # Convert lists to np array
    A, X, Y, B = np.array(A), np.array(X), np.array(Y), np.array(B)

    # Get indices of unique rows and create new array without duplicates
    if remove_duplicates:
        _, indx = np.unique(X, axis=0, return_index=True)
        nX, nA, nY, nB = X[indx], A[indx], Y[indx], B[indx]
    else:
        nX, nA, nY, nB = X, A, Y, B
    print("{} of {} dataset entries are unique".format(nY.shape[0], Y.shape[0]))

    X_train, X_test, \
    A_train, A_test, \
    Y_train, Y_test, \
    B_train, B_test = train_test_split(nX, nA, nY, nB, test_size = 0.25, random_state = seed)

    X_train, X_val, \
    A_train, A_val, \
    Y_train, Y_val, \
    B_train, B_val  = train_test_split(X_train, A_train, Y_train, B_train, test_size = 0.20, random_state = seed)

    if standardise_outputs:
        nY_train, nY_val, nY_test, mu, std = node_standardise(Y_train, Y_val, Y_test, nodewise=nodewise)
    else:
        nY_train, nY_val, nY_test, mu, std = Y_train, Y_val, Y_test, np.array([0]), np.array([1])

    train_dict = {"X":X_train, "A":A_train, "Y":nY_train, "B":B_train}
    val_dict   = {"X":X_val, "A":A_val, "Y":nY_val, "B":B_val}
    test_dict  = {"X":X_test, "A":A_test, "Y":nY_test, "B":B_test}
    data_transform_dict = {"std":std, "mu":mu}

    data_dict = {"train":train_dict, "val":val_dict, "test":test_dict, "transform_info":data_transform_dict}

    print('Dataset ready!')
    return data_dict


def prep_loaders(data, keys, scaler_inp=None, scaler_out=None, no_shuffle=False):
    loaders = []
    num_nodes = data[keys[0]]['X'].shape[1]
    bool_nonzero_indx = (data[keys[0]]['X'][:,:,0] + data[keys[0]]['X'][:,:,1]).mean(0) != 0
    for i in range(3):
        do_shuffle = True if i == 0 and not no_shuffle else False
        num_samples = data[keys[i]]['X'].shape[0]
        inp_shape = (num_samples, int(2*bool_nonzero_indx.sum()))
        sel_inp = data[keys[i]]['X'][:, bool_nonzero_indx, :2].reshape(inp_shape)
        sel_out = data[keys[i]]['Y'].reshape((num_samples, num_nodes*2))

        rescaled_inp = scaler_inp.transform(sel_inp) if scaler_inp is not None else sel_inp
        rescaled_out = scaler_out.transform(sel_out) if scaler_out is not None else sel_out

        torch_inp = torch.tensor(rescaled_inp, dtype=torch.float32)
        torch_out = torch.tensor(rescaled_out, dtype=torch.float32)

        dataset = TensorDataset(torch_inp, torch_out)

        loaders.append(DataLoader(dataset, batch_size=32, shuffle=do_shuffle))
    return loaders
