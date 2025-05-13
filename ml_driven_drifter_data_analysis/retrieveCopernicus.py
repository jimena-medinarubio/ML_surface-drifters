#%%

import copernicusmarine


copernicusmarine.subset(
  dataset_id="cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i",
  variables=["uo", "vo"],
  minimum_longitude=2,
  maximum_longitude=9,
  minimum_latitude=53,
  maximum_latitude=57,
  start_datetime="2024-04-26T00:00:00",
  end_datetime="2024-07-03T00:00:00",
)
# %%
