# %%
import pandas as pd
import numpy as np
from oneibl.onelight import ONE

one = ONE()
one.set_figshare_url = "https://figshare.com/articles/steinmetz/9974357"

sessions =  one.search(['spikes'])
sess = sessions[2]
# print(one.list(sess))

# %%
channels_loc = one.load_dataset(sess, 'channels.brainLocation')

spikes_clust = one.load_dataset(sess, 'spikes.clusters')
spikes_times = one.load_dataset(sess, 'spikes.times')

clust_probe = one.load_dataset(sess, 'clusters.probes')
clust_channel = one.load_dataset(sess, 'clusters.peakChannel')

print(f"Regions in session: {set(sorted(channels_loc))}")

# %%
spikes = pd.DataFrame(dict(cluster=spikes_clust.ravel(), times=spikes_times.ravel()))

# %%
spikes['channel']= [clust_channel[s].ravel()[0] for s in spikes_clust]
spikes

# %%
channels_loc_dic = {i:row.allen_ontology for i,row in channels_loc.iterrows()}
spikes['region'] = [channels_loc_dic[int(ch)] for ch in spikes['channel']]

# %%
spikes
# %%
sorted(set(spikes.region))

# %%
