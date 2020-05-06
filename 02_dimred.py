# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
import umap
from vtkplotter import *
from sklearn import decomposition, manifold
from sklearn.decomposition import PCA
import os

from brainrender.scene import Scene

from random  import choices

import pickle 


def save_pickle(filepath, data):
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filepath):
    """
	Load a pickle file
	:param filepath: path to pickle file
	"""
    if filepath is None or not os.path.isfile(filepath):
        raise ValueError("unrecognized file path: {}".format(filepath))
    if not "pkl" in filepath and not "pickle" in filepath:
        raise ValueError("unrecognized file path: {}".format(filepath))

    with open(filepath, 'rb') as handle:
        data = pickle.load(handle)
    return data


# %%
# ------------------------------ Params and load ----------------------------- #
load_isomap = True
load_umap = False

transform_all_iso = True
N_samples = 40000

data = np.load(os.path.join('data', 'SCm_iso_20.npy'))

rates = np.load(os.path.join('data', 'sess_36_SCm_frates.npy'))
# idxs = choices(np.arange(rates.shape[0]), k=N_samples)
# rates = rates[idxs, :]
rates = rates[:N_samples, :]

# rates = rates.reshape((-1, rates.shape[1], 10)).mean(axis=2) # reshape and bin

# %%
# ------------------------------------ PCA ----------------------------------- #
print('PCA')

pca = PCA(n_components=3).fit_transform(data)

coords = pd.DataFrame(dict(x=pca[:, 0], y=pca[:, 1], z=pca[:, 2]))

# isoscene = Scene(add_root=False, display_inset=False, title='pca')
# isoscene.add_cells(coords, radius=.05, color='green', res=24)
# isoscene.render()


# show(pca_points)
# %%
# ---------------------------------- Isomap ---------------------------------- #
print('Isomap')
isopath = os.path.join('data', 'SCm_iso_3.pkl')
iso20path = os.path.join('data', 'SCm_iso20.pkl')

if load_isomap:
    proj_data = np.load(os.path.join('data', 'SCm_iso_3.npy'))
    iso_instance = load_pickle(isopath)
else:
    iso_instance = manifold.Isomap(4, 3)

    proj_data = iso_instance.fit_transform(data)
    np.save(os.path.join('data', 'SCm_iso_3.npy'), proj_data)
    save_pickle(isopath, iso_instance)

if transform_all_iso:
    print(f'Transforming all data with dimension: {rates.shape}')
    iso20 = load_pickle(iso20path)
    print('All data to 20')
    proj_data = iso20.transform(np.sqrt(rates))
    print('All data to 3')
    proj_data = iso_instance.transform(proj_data)

starts = proj_data[1:, :]
ends = proj_data[:-1, :]
lines = Lines(starts, endPoints=ends)


coords = pd.DataFrame(dict(x=proj_data[:, 0], y=proj_data[:, 1], z=proj_data[:, 2]))

isoscene = Scene(add_root=False, display_inset=False, title='isomap')
isoscene.add_cells(coords, radius=0.1, color='salmon', res=24)
isoscene.add_vtkactor(lines)
isoscene.render()
isoscene.close()


# %%
# ----------------------------------- UMAP ----------------------------------- #
print('Umap')
_umap_params = dict(
    n_neighbors=5,  # higher values favour global vs local structure
    n_components=3,
    min_dist=0.1, # min distance between point in low D embedding space. Low vals favour clustering
)

if not load_umap:
    umapper = umap.UMAP(**_umap_params)

    umapped = umapper.fit_transform(data)
    np.save(os.path.join('data', 'SCm_umap_3.npy'), umapped)
else:
    umapped = np.load(os.path.join('data', 'SCm_umap_3.npy'))

# Plot
coords = pd.DataFrame(dict(x=umapped[:, 0], y=umapped[:, 1], z=umapped[:, 2]))

isoscene = Scene(add_root=False, display_inset=False, title='umap')
isoscene.add_cells(coords, radius=.5, color='blue', res=24)
isoscene.render()

