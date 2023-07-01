import numpy as np
import xarray as xr
import rioxarray as rxr
from pygeotools.lib import iolib, warplib

"""
# crap
evhr_2m = '/explore/nobackup/projects/ilab/projects/AIML_CHM/DEM_CHM/data/WV03_20160616_P1BS_104001001EBDB400_104001001E13F600-toa-stacked.tif'
dsm_24m = '/explore/nobackup/projects/ilab/projects/AIML_CHM/DEM_CHM/labels/WV03_20160616_104001001E13F600_104001001EBDB400-DEM_24m.tif'
output_filename = '/explore/nobackup/projects/ilab/projects/AIML_CHM/DEM_CHM/labels/WV03_20160616_P1BS_104001001EBDB400_104001001E13F600-DEM_24m_to_1m.tif'
"""

evhr_2m = '/explore/nobackup/projects/ilab/projects/AIML_CHM/DEM_CHM/data/WV02_20120924_P1BS_103001001C087E00_103001001CB20900-toa-stacked.tif'
dsm_24m = ''

warp_2m_dsm_list = warplib.memwarp_multi_fn(
    [dsm_24m],
    res=evhr_2m,
    extent=evhr_2m,
    t_srs=evhr_2m,
    r='mode',
    dst_ndv=-10001
)

raster = warp_2m_dsm_list[0].GetRasterBand(1)
raster = raster.ReadAsArray()

print(raster.shape)

original_raster = rxr.open_rasterio(evhr_2m)
print(original_raster.shape)


image = original_raster.drop(
    dim="band",
    labels=original_raster.coords["band"].values[1:],
)


# Get metadata to save raster
prediction = xr.DataArray(
    np.expand_dims(raster, axis=0),
    name='1m-dem',
    coords=image.coords,
    dims=image.dims,
    attrs=image.attrs
)

nodata = prediction.rio.nodata
prediction = prediction.where(image != nodata)
prediction.rio.write_nodata(-10001, encoded=True, inplace=True)

# Save output raster file to disk
prediction.rio.to_raster(
    output_filename,
    BIGTIFF="IF_SAFER",
    compress='LZW',
    driver='GTiff',
    dtype='float32'
)


"""
image = rxr.open_rasterio(filename)
# Transpose the image for channel last format
image = image.transpose("y", "x", "band")

# Drop image band to allow for a merge of mask
image = image.drop(
    dim="band",
    labels=image.coords["band"].values[1:],
)

# Get metadata to save raster
prediction = xr.DataArray(
    np.expand_dims(prediction, axis=-1),
    name=self.conf.experiment_type,
    coords=image.coords,
    dims=image.dims,
    attrs=image.attrs
)

# Add metadata to raster attributes
prediction.attrs['long_name'] = (self.conf.experiment_type)
prediction.attrs['model_name'] = (self.conf.model_filename)
prediction = prediction.transpose("band", "y", "x")

# Set nodata values on mask
nodata = prediction.rio.nodata
prediction = prediction.where(image != nodata)
prediction.rio.write_nodata(
    self.conf.prediction_nodata, encoded=True, inplace=True)

# Save output raster file to disk
prediction.rio.to_raster(
    output_filename,
    BIGTIFF="IF_SAFER",
    compress=self.conf.prediction_compress,
    driver=self.conf.prediction_driver,
    dtype=self.conf.prediction_dtype
)
"""
