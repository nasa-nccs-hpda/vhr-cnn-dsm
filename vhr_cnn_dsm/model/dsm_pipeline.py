import os
import sys
import time
import logging
import rasterio
import osgeo.gdal
import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from rioxarray.merge import merge_arrays

from tqdm import tqdm
from glob import glob
from pathlib import Path
from itertools import repeat
from multiprocessing import Pool, cpu_count

from vhr_cnn_dsm.model.config import DSMConfig as Config
from tensorflow_caney.utils.system import seed_everything
from tensorflow_caney.model.pipelines.cnn_regression import CNNRegression
from tensorflow_caney.utils.data import gen_random_tiles, \
    get_dataset_filenames, get_mean_std_dataset
from tensorflow_caney.model.dataloaders.regression import RegressionDataLoader


class DSMPipeline(CNNRegression):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config_filename, logger=None):

        # Set logger
        self.logger = logger if logger is not None else self._set_logger()

        # Configuration file intialization
        self.conf = self._read_config(config_filename, Config)

        # Set experiment name
        self.experiment_name = self.conf.experiment_name.name

        # output directory to store metadata and artifacts
        self.metadata_dir = os.path.join(self.conf.data_dir, 'metadata')
        self.logger.info(f'Metadata dir: {self.metadata_dir}')

        # Set output directories and locations
        self.intermediate_dir = os.path.join(
            self.conf.data_dir, 'intermediate')
        self.logger.info(f'Intermediate dir: {self.intermediate_dir}')

        self.images_dir = os.path.join(self.conf.data_dir, 'images')
        self.logger.info(f'Images dir: {self.images_dir}')

        self.labels_dir = os.path.join(self.conf.data_dir, 'labels')
        self.logger.info(f'Labels dir: {self.labels_dir}')

        self.model_dir = self.conf.model_dir
        self.logger.info(f'Model dir: {self.labels_dir}')

        # Create output directories
        for out_dir in [
                self.images_dir, self.labels_dir,
                self.metadata_dir, self.model_dir]:
            os.makedirs(out_dir, exist_ok=True)

        # Seed everything
        seed_everything(self.conf.seed)

    # -------------------------------------------------------------------------
    # stack_rasters
    # -------------------------------------------------------------------------
    def stack_rasters(
                self,
                raster_list: list,
                output_filename: str = None,
                overwrite: bool = False):
        """
        Stack rasters from list
        Note: The overall concat function is extremely slow. Reconsider using
        other functions to achieve the same result. Consider loading the raster
        to memory to make this step faster.
        """
        logging.info('Entering stack_rasters function')

        # if file exists and no overwrite is specified, return raster obj
        if os.path.isfile(output_filename) and not overwrite:
            logging.info(f'File exists, opening {output_filename}')
            return rxr.open_rasterio(output_filename)

        # if not, merge raster
        merge_list = []
        for raster_filename in raster_list:
            merge_list.append(rxr.open_rasterio(raster_filename))
        merged_raster = xr.concat(merge_list, dim='band')
        logging.info(f'New shape of merged raster {merged_raster.shape}')

        if output_filename is not None:
            merged_raster.rio.to_raster(output_filename)
            logging.info(f'Saving raster to {output_filename}')

        return merged_raster

    def get_dsm(self, dsm_regex: str):
        """
        Find a and return DSM rioxarray object
        """
        dsm_list = glob(dsm_regex)
        assert len(dsm_list) == 1, \
            f'{dsm_regex} does not return one element, {dsm_list} found'
        dsm_raster = rxr.open_rasterio(dsm_list[0])
        return dsm_raster.where(dsm_raster != -99)

    # -------------------------------------------------------------------------
    # preprocess
    # -------------------------------------------------------------------------
    def preprocess(self):

        logging.info('Entering preprocess step')

        # read stereopair, stack stereo pair
        assert len(self.conf.stereo_dirs) > 0, \
            'stereo_pair configuration option must have a list of directories'

        # iterate over each directory and get the respective data files
        for index_id, stereo_output_dir in enumerate(self.conf.stereo_dirs):

            # get stereo pairs
            stereo_pair_list = glob(
                os.path.join(stereo_output_dir, self.conf.stereo_pair_regex))
            assert len(stereo_pair_list) == 2, \
                f'len(stereo_pair_list) != 2, {stereo_pair_list} found'

            # output filename for stereo pair stack
            output_filename = \
                f'{Path(stereo_pair_list[0]).with_suffix("")}_stacked.tif'

            # -----------------------------------------------------------------
            # prepare experiments with stereo pairs
            # -----------------------------------------------------------------
            if self.experiment_name in ['stereo', 'stereo-disparity']:

                # raster stack of imagery, the first run might take some time
                # if the rasters are not stacked, after that, rasters are saved
                # and loaded from memory, helps when repeating the process
                stereo_pair_raster = self.stack_rasters(
                    stereo_pair_list, output_filename)
                logging.info(f'Stereo stack shape: {stereo_pair_raster.shape}')

            # -----------------------------------------------------------------
            # prepare experiments with disparity maps
            # -----------------------------------------------------------------
            if self.experiment_name in ['disparity', 'stereo-disparity']:

                # get disparity map
                disparity_map_list = glob(
                    os.path.join(
                        stereo_output_dir, self.conf.disparity_map_regex))
                assert len(disparity_map_list) == 1, \
                    f'len(disparity_map_list) != 1, {disparity_map_list} found'
                disparity_map_raster = rxr.open_rasterio(disparity_map_list[0])
                logging.info(
                    f'Disparity Map shape: {disparity_map_raster.shape}')

                # preprocess disparity map and drop the last band if needed
                if disparity_map_raster.shape[0] > 2:
                    logging.info('Removing angle band from disparity map')
                    disparity_map_raster = disparity_map_raster[:2, :, :]
                    logging.info(
                        f'New Disparity Map: {disparity_map_raster.shape}')

            # -----------------------------------------------------------------
            # prepare data stacks depending on the experiments
            # -----------------------------------------------------------------
            if self.experiment_name == 'stereo':
                image = stereo_pair_raster
                print(image.shape)
            elif self.experiment_name == 'disparity':
                image = disparity_map_raster
                print(image.shape)
                disparity_output_filename = \
                    f'{Path(stereo_pair_list[0]).with_suffix("")}' + \
                    '_disparity.tif'
                if not os.path.isfile(disparity_output_filename):
                    logging.info(f'Saving {disparity_output_filename}')
                    image.rio.to_raster(disparity_output_filename)
            elif self.experiment_name == 'stereo-disparity':

                # TODO: FIX THE CONCAT OPTION FOR LARGE RASTERS

                #print(stereo_pair_raster.shape, disparity_map_raster.shape)
                #print(stereo_pair_raster.rio.crs, disparity_map_raster.rio.crs)
                #disparity_map_raster['band'] = [3, 4]
                #print(disparity_map_raster)
                #image = xr.concat(
                #    [
                #         stereo_pair_raster,
                #         disparity_map_raster
                #    ], dim='band'
                #)
                #print(image)
                #print(stereo_pair_raster.min().values, stereo_pair_raster.max().values)
                #print(disparity_map_raster.min().values, disparity_map_raster.max().values)
                #print(image.min().values, image.max().values)


                #xr.concat(
                #    [
                #        raster_1,
                #        raster_2.reset_coords('band', drop=True).expand_dims(band=[232]),
                #    ], dim='band',
                #)
                #print(image)

                #stereo_disparity_output_filename = \
                #    f'{Path(stereo_pair_list[0]).with_suffix("")}' + \
                #    '_stereo-disparity.tif'
                #image.rio.to_raster(stereo_disparity_output_filename)
                #if not os.path.isfile(stereo_disparity_output_filename):
                #    logging.info(f'Saving {stereo_disparity_output_filename}')
                #    image.rio.to_raster(stereo_disparity_output_filename)

                # TODO: REMOVE THIS BANDAID
                disparity_map_list = glob(
                    os.path.join(
                        stereo_output_dir, 'stack_all_bands.tif'))
                assert len(disparity_map_list) == 1, \
                    f'len(disparity_map_list) != 1, {disparity_map_list} found'
                disparity_map_raster = rxr.open_rasterio(disparity_map_list[0])
                logging.info(
                    f'Disparity Map shape: {disparity_map_raster.shape}')
                image = disparity_map_raster
                print(image.shape)

            # -----------------------------------------------------------------
            # DSM ground truth preprocessing
            # -----------------------------------------------------------------

            # low res dsm raster
            lowres_dsm_raster = self.get_dsm(
                os.path.join(stereo_output_dir, self.conf.lowres_dsm_regex))
            logging.info(f'Low Res DSM shape: {lowres_dsm_raster.shape}')

            # mid res dsm raster
            midres_dsm_raster = self.get_dsm(
                os.path.join(stereo_output_dir, self.conf.midres_dsm_regex))
            logging.info(f'Mid Res DSM shape: {midres_dsm_raster.shape}')

            # mid res dsm raster
            highres_dsm_raster = self.get_dsm(
                os.path.join(stereo_output_dir, self.conf.highres_dsm_regex))
            logging.info(f'High Res DSM shape: {highres_dsm_raster.shape}')

            # -----------------------------------------------------------------
            # DSM exploratory data analysis
            # -----------------------------------------------------------------
            logging.info(
                f'low res min max {lowres_dsm_raster.min().values}, ' +
                f'{lowres_dsm_raster.max().values}')
            logging.info(
                f'mid res min max {midres_dsm_raster.min().values}, ' +
                f'{midres_dsm_raster.max().values}')
            logging.info(
                f'high res min max {highres_dsm_raster.min().values}, ' +
                f'{highres_dsm_raster.max().values}')

            # -----------------------------------------------------------------
            # DSM 1m to match native 0.5 m resolution
            # -----------------------------------------------------------------
            logging.info(f'{"=" * 5} Stereo DSM Match Disparity Map {"=" * 5}')
            lowres_dsm_raster = lowres_dsm_raster.rio.reproject_match(image)
            logging.info(f'Low Res DSM shape: {lowres_dsm_raster.shape}')

            midres_dsm_raster = midres_dsm_raster.rio.reproject_match(image)
            logging.info(f'Mid Res DSM shape: {midres_dsm_raster.shape}')

            highres_dsm_raster = highres_dsm_raster.rio.reproject_match(image)
            logging.info(f'High Res DSM shape: {highres_dsm_raster.shape}')

            # combine DSMs to fill voids in highres DSM
            logging.info(f'{"=" * 5} Stereo DSM Fill Voids {"=" * 5}')

            # lowres_output_filename = \
            #    f'{Path(stereo_pair_list[0]).with_suffix("")}_lowres_downsampled.tif'
            # lowres_dsm_raster.rio.to_raster(lowres_output_filename)

            # midres_output_filename = \
            #    f'{Path(stereo_pair_list[0]).with_suffix("")}_midres_downsampled.tif'
            # midres_dsm_raster.rio.to_raster(midres_output_filename)

            # highres_output_filename = \
            #    f'{Path(stereo_pair_list[0]).with_suffix("")}_highres_downsampled.tif'
            # highres_dsm_raster.rio.to_raster(highres_output_filename)

            # -----------------------------------------------------------------
            # DSM voids filled by coarser DSMs
            # -----------------------------------------------------------------
            midres_dsm_raster = midres_dsm_raster.fillna(lowres_dsm_raster)
            highres_dsm_raster = highres_dsm_raster.fillna(midres_dsm_raster)

            # midres_output_filename = \
            #    f'{Path(stereo_pair_list[0]).with_suffix("")}_midres_downsampled_filled.tif'
            # midres_dsm_raster.rio.to_raster(midres_output_filename)

            # highres_output_filename = \
            #    f'{Path(stereo_pair_list[0]).with_suffix("")}_highres_downsampled_filled.tif'
            # highres_dsm_raster.rio.to_raster(highres_output_filename)

            #highres_output_filename = \
            #    f'{Path(stereo_pair_list[0]).with_suffix("")}_highres_downsampled_filled-test.tif'
            #highres_dsm_raster.rio.to_raster(highres_output_filename)

            # -----------------------------------------------------------------
            # Generate invidual chips for training
            # -----------------------------------------------------------------
            image = image.values
            label = highres_dsm_raster.values

            # Move from chw to hwc, squeze mask if required
            image = np.moveaxis(image, 0, -1)
            label = np.squeeze(label) if len(label.shape) != 2 else label
            logging.info(f'Image: {image.shape}, Label: {label.shape}')
            logging.info(f'Label classes min {label.min()}, max {label.max()}')

            # Normalize values within [0, 1] range
            # image = normalize_image(image, self.conf.normalize)

            # Rescale values within [0, 1] range
            # image = rescale_image(image, self.conf.rescale)

            # Modify labels, sometimes we need to merge some training classes
            # label = modify_label_classes(
            #    label, self.conf.modify_labels, self.conf.substract_labels)
            # logging.info(f'Label classes min {label.min()}, max {label.max()}')

            # generate random tiles
            gen_random_tiles(
                image=image,
                label=label,
                expand_dims=self.conf.expand_dims,
                tile_size=self.conf.tile_size,
                index_id=index_id,
                num_classes=self.conf.n_classes,
                max_patches=self.conf.n_tiles,
                include=self.conf.include_classes,
                augment=self.conf.augment,
                output_filename=stereo_pair_list[0],
                out_image_dir=self.images_dir,
                out_label_dir=self.labels_dir,
                json_tiles_dir=self.conf.json_tiles_dir,
                dataset_from_json=self.conf.dataset_from_json,
                xp=np,
                use_case='regression'
            )

        # Calculate mean and std values for training
        data_filenames = get_dataset_filenames(self.images_dir)
        label_filenames = get_dataset_filenames(self.labels_dir)
        logging.info(f'Mean and std values from {len(data_filenames)} files.')

        # Temporarily disable standardization and augmentation
        current_standardization = self.conf.standardization
        self.conf.standardization = None
        metadata_output_filename = os.path.join(
            self.model_dir, f'mean-std-{self.conf.experiment_name}.csv')
        os.makedirs(self.model_dir, exist_ok=True)

        # Set main data loader
        main_data_loader = RegressionDataLoader(
            data_filenames, label_filenames, self.conf, False
        )

        # Get mean and std array
        mean, std = get_mean_std_dataset(
            main_data_loader.train_dataset, metadata_output_filename)
        logging.info(f'Mean: {mean.numpy()}, Std: {std.numpy()}')

        # Re-enable standardization for next pipeline step
        self.conf.standardization = current_standardization

        logging.info('Done with preprocessing stage')

        return
