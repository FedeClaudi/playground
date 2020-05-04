# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
import umap
from vtkplotter import *
from sklearn import decomposition, manifold
from sklearn.decomposition import PCA

# %%
# ------------------------------ Params and load ----------------------------- #
load_isomap = False
load_umap = True


regions = ['RSP', 'SCm']
regdata = namedtuple('regdata', 'spikes cells binned')
data = {r:regdata(
            pd.read_hdf(f'{r}_spikes.h5'), np.load(f'{r}_cells.npy'), np.load(f'{r}_cells_binned.npy')
                ) for r in regions }

# %%
# ------------------------------------ PCA ----------------------------------- #
print('PCA')
pca = PCA(n_components=3).fit_transform(data['SCm'].binned)

coords = [pca[i, :] for i in np.arange(pca.shape[0])]
pca_points = Spheres(coords, r=.25, c='green', alpha=.8)

# %%
# ---------------------------------- Isomap ---------------------------------- #
print('Isomap')
if load_isomap:
    proj_data = np.load('SCm_iso_3.npy')
else:
    iso_instance = manifold.Isomap(15, 3)
    proj_data = iso_instance.fit_transform(data['SCm'].binned)
    np.save('SCm_iso_3.npy', proj_data)


coords = [proj_data[i, :] for i in np.arange(proj_data.shape[0])]
iso_points = Spheres(coords, r=.25, c='salmon', alpha=.8)




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
    umapped = umapper.fit_transform(data['SCm'].binned)
    np.save('SCm_umap_3.npy', umapped)
else:
    umapped = np.load('SCm_umap_3.npy')

coords = [umapped[i, :] for i in np.arange(umapped.shape[0])]
umap_points = Spheres(coords, r=.25, c='blue')



# %%
# -------------------------------- Show render ------------------------------- #
show(pca_points, iso_points, umap_points, shape=[1, 3])

