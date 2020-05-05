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

# %%
# ------------------------------ Params and load ----------------------------- #
load_isomap = False
load_umap = False
N_SAMPLES =  100000

regions = ['SCm']
regdata = namedtuple('regdata', 'spikes rates')
data = {r:regdata(
            pd.read_hdf(os.path.join('data', 'sess36_alldata.h5')), 
                    np.load(os.path.join('data', f'sess_36_{r}_frates.npy'))) for r in regions }

random_idxs = choices(np.arange(data['SCm'].rates.shape[0]), k=N_SAMPLES)

# %%
# ------------------------------------ PCA ----------------------------------- #
print('PCA')

pca = PCA(n_components=3).fit_transform(data['SCm'].rates[random_idxs, :])

coords = [pca[i, :] for i in np.arange(pca.shape[0])]
pca_points = Spheres(coords, r=.25, c='green', alpha=.8)
# show(pca_points)
# %%
# ---------------------------------- Isomap ---------------------------------- #
print('Isomap')
if load_isomap:
    proj_data = np.load('SCm_iso_3.npy')
else:
    iso_instance = manifold.Isomap(4, 3)

    proj_data = iso_instance.fit_transform(data['SCm'].rates[random_idxs, :])
    np.save('SCm_iso_3.npy', proj_data)


# coords = [proj_data[i, :] for i in np.arange(proj_data.shape[0])]
# iso_points = Spheres(coords, r=.005, c='salmon', alpha=.8)

coords = pd.DataFrame(dict(x=proj_data[:, 0], y=proj_data[:, 1], z=proj_data[:, 2]))

isoscene = Scene(add_root=False, display_inset=False, title='isomap')
isoscene.add_cells(coords, radius=0.0025, color='salmon', res=24)
isoscene.render()
# show(iso_points, newPlotter=True)


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

    umapped = umapper.fit_transform(data['SCm'].rates[random_idxs, :])
    np.save('SCm_umap_3.npy', umapped)
else:
    umapped = np.load('SCm_umap_3.npy')

# coords = [umapped[i, :] for i in np.arange(umapped.shape[0])]
# umap_points = Spheres(coords, r=.25, c='blue')


coords = pd.DataFrame(dict(x=umapped[:, 0], y=umapped[:, 1], z=umapped[:, 2]))

isoscene = Scene(add_root=False, display_inset=False, title='umap')
isoscene.add_cells(coords, radius=.5, color='skyeblue', res=24)
isoscene.render()

