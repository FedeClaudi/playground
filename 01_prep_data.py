
# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from scipy import stats
import multiprocessing as mp
from sklearn import decomposition, manifold


def get_cell_frate(args):
    time, cells, cell, sigma = args
    print(f'processing cell {cell} of {cells.shape[1]}')

    res = np.zeros_like(time).ravel()
    spike_times = np.where(cells[:, cell] == 1)[0]

    for spike in tqdm(spike_times):
        gauss = stats.norm(spike, sigma)
        support = np.linspace(0, len(time), len(time))
        density = gauss.pdf(support)

        res = res + cells[:, cell] * density
    return res

def run():
    from oneibl.onelight import ONE

    # Get a session
    one = ONE()
    one.set_figshare_url = "https://figshare.com/articles/steinmetz/9974357"

    sessions =  one.search(['spikes'])



    sess_n = 36
    sess = sessions[36]

    # ----------------------------- Get metadata/data ---------------------------- #
    df_fp = os.path.join('data', f'sess{sess_n}_alldata.h5')
    if not os.path.isfile(df_fp):
        # Download some data
        channels_loc = one.load_dataset(sess, 'channels.brainLocation')

        spikes_clust = one.load_dataset(sess, 'spikes.clusters')
        spikes_times = (one.load_dataset(sess, 'spikes.times').ravel() * 1000).astype(np.int32) # convert to msecs

        clust_probe = one.load_dataset(sess, 'clusters.probes')
        clust_channel = one.load_dataset(sess, 'clusters.peakChannel')

        print(f"Regions in session: {sorted(set(channels_loc.allen_ontology))}")

        # Assign a spike to each cluster and each cluster to a brain region
        spikes = pd.DataFrame(dict(cluster=spikes_clust.ravel(), times=spikes_times.ravel()))

        spikes['channel']= [clust_channel[s[0]].ravel()[0] for s in spikes_clust]

        channels_loc_dic = {i:row.allen_ontology for i,row in channels_loc.iterrows()}
        spikes['region'] = [channels_loc_dic[int(ch-1)] for ch in spikes['channel']]

        print(spikes)
        print(set(spikes.region.values))

        spikes.to_hdf(df_fp, key='hdf')
    else:
        spikes = pd.read_hdf(df_fp)


    # Get spikes for one brain region at the time
    max_T = 10 * 60 * 1000 # 10 minutes
    time = np.zeros((max_T, 1))
    time_vals = np.arange(0, len(time))
    sigma = 100 # 100ms std


    for region in ['SCm']:
        # prep some file paths
        frates_fp = os.path.join('data', f'sess_{sess_n}_{region}_frates.npy')

        reg = spikes.loc[spikes.region==region]
        reg_clusters = list(set(reg.cluster.values))
        print(f"Found {len(reg_clusters)} clusters for {region}")

        # Create an array that is N_samples x N_cells and is 1 when a cell fires
        print('Getting cells spikes')
        cells = np.zeros((time.shape[0], len(reg_clusters)))

        for n, clust in enumerate(reg_clusters):
            clust_spikes = spikes.loc[(spikes.cluster == clust)&(spikes.times < max_T)]
            cells[clust_spikes.times, n] = 1
        cells = cells[:max_T, :]

        # Convert spikes to rates by summing with a gaussian kernel
        print('Getting firing rates')
        if not os.path.isfile(frates_fp):
            frates = np.zeros_like(cells)

            pool = mp.Pool(mp.cpu_count()-2)
            res = pool.map(get_cell_frate, [(time, cells, cell, sigma) for cell in np.arange(frates.shape[1])])
            pool.close()

            for n, rate in enumerate(res):
                frates[:, n] = rate

            np.save(frates_fp, frates)

        else:
            frates = np.load(frates_fp)


        # TODO Bin spike counts in 1s bins for persistent homology

        # embedd data into lower dimensional embeddiing spaces to facilitate future analysi
        print('Reducing dimensionality')
        stable_frates = np.sqrt(frates)

        iso_instance = manifold.Isomap(5, 20) # 5 nearest neightbours and 20 dimensional embedding
        proj_data = iso_instance.fit_transform(stable_frates)
        np.save(os.path.join('data', 'SCm_iso_20.npy'), proj_data)



if __name__ == "__main__":
    run()

