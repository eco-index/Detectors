#support functions to get data together to train a UNET.
#input imagery is currently planet data either 4 or 8 band images merged into single image
#input mask is a shapefile with only single type of landcover, cropped to same size as input image

import geopandas as gpd
import rasterio
from osgeo import gdal, ogr
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Polygon

import numpy as np
from PIL import Image

import os

#return the image and mask crs and bounds in a list 4 long with TRUE/FALSE at start indicating crs matches (TRUE/FALSE, image crs, image bounds, mask crs, mask bounds)
def check_bounds_crs(image_name, mask_name):
	image = rasterio.open(image_name)
	mask = gpd.read_file(mask_name)

	image_crs = image.crs
	image_bounds = image.bounds

	mask_crs = mask.crs
	mask_bounds = mask.total_bounds

	if (image_crs == mask_crs):
		match_crs = True
	else: match_crs = False

	return([match_crs, image_crs, image_bounds, mask_crs, mask_bounds])

#return shapefile mask with new crs (to_crs)
def change_mask_crs(mask_name, to_crs):
	mask = gpd.read_file(mask_name)
	return(mask.to_crs(to_crs))

#returns name of new image with crs saved to file. to_crs needs to be in format: 'ESPG:2193' dictated by rasterio.warp
def change_image_crs(image_name, to_crs):
	image = rasterio.open(image_name)
	dstCrs = {'init': to_crs}
	#calculate transform array and shape of reprojected raster
	transform, width, height = calculate_default_transform(image.crs, dstCrs, image.width, image.height, *image.bounds)

	#working of the meta for the destination raster
	kwargs = image.meta.copy()
	kwargs.update({
	        'crs': dstCrs,
	        'transform': transform,
	        'width': width,
	        'height': height
	    })

	save_name = 'image_new_crs.tif'
	#open destination raster
	dstRst = rasterio.open(save_name, 'w', **kwargs)

	#reproject and save raster band data
	for i in range(1, image.count + 1):
	    reproject(
	        source=rasterio.band(image, i),
	        destination=rasterio.band(dstRst, i),
	        #src_transform=srcRst.transform,
	        src_crs=image.crs,
	        #dst_transform=transform,
	        dst_crs=dstCrs,
	        resampling=Resampling.nearest)

	#close destination raster
	dstRst.close()
	return(save_name)

#save geotiff image cropped in by buffer amount. Buffer in decimal eg 0.15 for 15%.
def clipping_box(image_name, mask_name, buffer):
	image = rasterio.open(image_name)
	mask = gpd.read_file(mask_name)
	xrange = image.bounds[2]-image.bounds[0]
	yrange = image.bounds[3]-image.bounds[1]
	#buffer to crop image from edge in percent of range

	#define the clipping box as a polygon coordinates:
	TL = (image.bounds[0]+(xrange*buffer),image.bounds[3]-(yrange*buffer))
	TR = (image.bounds[2]-(xrange*buffer), image.bounds[3]-(yrange*buffer))
	BR = (image.bounds[2]-(xrange*buffer), (image.bounds[1]+(yrange*buffer)))
	BL = ((image.bounds[0]+(xrange*buffer)), (image.bounds[1]+(yrange*buffer)))

	#create a polygon bounding box
	bounding_box = Polygon([TL, TR, BR, BL])

	#clip the shapefile using the bounding box and save to disk
	shpclip = gpd.clip(mask, bounding_box)
	saved_mask = 'mask_cropped.shp'
	shpclip.to_file(saved_mask, driver='ESRI Shapefile')

	upper_left_x = (image.bounds[0]+(xrange*buffer))
	upper_left_y = (image.bounds[3]-(yrange*buffer))
	lower_right_x = (image.bounds[2]-(xrange*buffer))
	lower_right_y = (image.bounds[1]+(yrange*buffer))
	window = (upper_left_x,upper_left_y,lower_right_x,lower_right_y)

	saved_image = 'image_cropped.tif'
	gdal.Translate(saved_image, image_name, projWin = window)
	return(saved_image, saved_mask)

#experimental function to convert shapefile to geotiff in python code rather than having to use terminal, not tested yet
def shape_to_tiff(image_name, mask_name):
	fn_ras = image_name
	fn_vec = mask_name
	ras_ds = gdal.Open(fn_ras)
	vec_ds = ogr.Open(fn_vec)
	lyr = vec_ds.GetLayer()
	geot = ras_ds.GetGeoTransform()
	drv_tiff = gdal.GetDriverByName("GTiff")
	save_name = 'mask_rasterized.tif'
	chn_ras_ds = drv_tiff.Create(save_name, ras_ds.RasterXSize, ras_ds.RasterYSize, 1, gdal.GDT_Float32)
	chn_ras_ds.SetGeoTransform(geot)
	gdal.RasterizeLayer(chn_ras_ds, [1], lyr)
	chn_ras_ds.GetRasterBand(1).SetNoDataValue(0.0)
	chn_ras_ds = None
	return(save_name)

#creates a png image of the shapefile from the geotiff created in shape_to_tiff
def shapetiff_to_png(image_name, mask_name):
	ds = gdal.Open(mask_name)
	numpy_mask = np.array(ds.GetRasterBand(1).ReadAsArray())

	img_in = rasterio.open(image_name)
	img_bounds = img_in.bounds

	mask_in = rasterio.open(mask_name)
	mask_bounds = mask_in.bounds

	##create a numpy array of zeros with the image size and place the mask where it needs to go somehow.....
	img_x = img_bounds[2] - img_bounds[0]
	img_y = img_bounds[3] - img_bounds[1]

	mask_x = mask_bounds[2] - mask_bounds[0]
	mask_y = mask_bounds[3] - mask_bounds[1]

	img_shape = img_in.shape

	out_img = np.zeros(img_shape,np.uint8)
	out_img[:, int(img_in.shape[1]-numpy_mask.shape[1]):] = numpy_mask

	im = Image.fromarray(out_img)
	save_name = 'mask_cropped.shp.png'
	im.save(save_name)

#delete the files that were created mid process
def clean_tmp_files(del_file):
	#check file exists
	os.path.exists(path_to_file)
	#delete temp files
	os.remove(del_file)