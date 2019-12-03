"""
This module implements a Python class for the interactive analysis
of microscopy images of synapases imaged by fluorescence microscopy.
"""
# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3 License


from IPython.display import display, clear_output
from notebook.notebookapp import list_running_servers

import ipywidgets as ipw
from ipywidgets import ColorPicker, VBox, jslink

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.filters
import skimage.morphology
import skimage.io
import pandas as pd
import scipy.ndimage as ndi
import glob, os
import subprocess

import napari
import ipyvolume as ipv
    
class Improc:
    
    def __init__(self, folder_name = 'upload_data'):

        """Standard __init__ method.
        
        Parameters
        ----------
        file = str
            file name
        
        
        Attributes
        ----------
            
        files = list
            list of files to process
        
        ax:  AxesSubplot object
        implot : AxesImage object
        
        file: upload widget
        select_file : selection widget
        select_file_to_plot : selection widget
        sigma_slide : float slider widget
        out : output widget
        out_zip : output widget
        process_button : button widget
        zip_button : button widget
        myHTML : HTML widget
        
        """
        
        self.folder_name = folder_name
        self.folder_init()
        
        self.ax = None
        self.implot = None
        self.im_region_mask = None
        self.im_objects = None
        self.density = []
        
        
        #create widgets
        self.file = ipw.FileUpload(multiple = True)
        self.select_file = ipw.SelectMultiple(options = tuple(self.files),description = 'Select files to process')
        self.select_file_to_plot = ipw.Select(options = tuple(self.files), index = 0, description = 'Select file to plot')

        self.process_button = ipw.Button(description = 'Process')
        self.zip_button = ipw.Button(description = 'Zip for download')
        
        self.out = ipw.Output()
        with self.out:
            display(ipv.figure())
        self.out_zip = ipw.Output()
        self.out_process = ipw.Output()

        #connect widgets to actions
        self.process_button.on_click(self.do_processing)
        self.zip_button.on_click(self.do_zipping)
        self.file.observe(self.on_upload, names='value')
        self.select_file_to_plot.observe(self.on_select_to_plot, names = 'index')
        
        my_adress = next(list_running_servers())['base_url']
        self.myHTML = ipw.HTML("""<a href="https://hub.gke.mybinder.org"""+my_adress+"""notebooks/to_download.tar.gz" target="_blank"><b>5. Hit this link to download your data<b></a>""")

        
    def folder_init(self):
        """Initialize file list with lsm files present in folder"""
        
        if not os.path.isdir(self.folder_name):
            os.makedirs(self.folder_name, exist_ok=True)
        
        lsm_files = glob.glob(self.folder_name+'/*.lsm')
        self.files = ['None']+[os.path.split(x)[1] for x in lsm_files]
        
        
    def on_upload(self, change):
        """call-back function for file upload. Uploads selected files
        and completes the files list"""
        
        #upload all files
        for filename in change['new'].keys():            
            with open(self.folder_name+'/'+filename, "wb") as f:
                f.write(change['new'][filename]['content'])
        
        #update the file lists
        self.folder_init()
        self.select_file.options=tuple(self.files)
        self.select_file_to_plot.options=tuple(self.files)
        
        self.current_file = self.select_file.value
        
        
    def on_select_to_plot(self, change):
        """Call-back function for plotting a 3D visualisaiton of the segmentation"""
        
        #if the selected file has changed, import image, segmentation and global mask and plot
        if change['new'] != change['old']:
            print('new: '+ str(change['new']))
            print('old: '+ str(change['old']))
            
            image = skimage.io.imread(self.folder_name+'/'+self.select_file_to_plot.value, plugin = 'tifffile')
            image2 = skimage.io.imread(self.folder_name+'/'+os.path.splitext(self.select_file_to_plot.value)[0]+'_label.tif', plugin = 'tifffile')
            image3 = skimage.io.imread(self.folder_name+'/'+os.path.splitext(self.select_file_to_plot.value)[0]+'_region.tif', plugin = 'tifffile')

            #create ipyvolume figure
            ipv.figure()
            volume_image = ipv.volshow(image[0,:,:,:,1],extent=[[0,1024],[0,1024],[-20,20]],level=[0.3, 0.2,0.2], 
                               opacity = [0.2,0,0])
            volume_seg = ipv.plot_isosurface(np.swapaxes(image2>0,0,2),level=0.5,controls=True, color='green',extent=[[0,1024],[0,1024],[-20,20]])
            volume_reg = ipv.volshow(image3,extent=[[0,1024],[0,1024],[-20,20]],level=[0.3, 0.2,1], 
                               opacity = [0.0,0,0.5])
            volume_reg.brightness = 10
            volume_image.brightness = 10
            volume_image.opacity = 100 
            ipv.xyzlim(0,1024)
            ipv.zlim(-500,500)
            ipv.style.background_color('black')

            #create additional controls to show/hide segmentation
            color = ColorPicker(description = 'Segmentation color')
            visible = ipw.Checkbox()
            jslink((volume_seg, 'color'), (color, 'value'))
            jslink((volume_seg, 'visible'), (visible, 'value'))
            ipv.show()
            with self.out:
                clear_output(wait=True)
                display(VBox([ipv.gcc(),  color, visible]))
                
            viewer = napari.Viewer(ndisplay = 3)
            viewer.add_image(image, colormap = 'red')
            viewer.add_image(image2, colormap = 'green', blending = 'additive')
            viewer.add_image(image3, colormap = 'blue', blending = 'additive')


       
    def do_processing(self, b):
        """Call-back function for proessing button. Executes image processing and saves result."""

        for f in self.select_file.value:
            if f != 'None':
                image = self.import_image(self.folder_name+'/'+f)
                synapse_region = self.find_synapse_area(image)
                synapse_mask = self.detect_synapses(image, synapse_region)
                synapse_label = skimage.morphology.label(synapse_mask)

                #measure regions
                synapse_regions = pd.DataFrame(skimage.measure.regionprops_table(synapse_label,properties=('label','area')))
                #calculate density as ration of # of synapses per pixels in the synapse region mask
                density = np.sum(synapse_regions.area>50)/np.sum(synapse_mask)

                self.density.append({'filename': os.path.splitext(f)[0], 'density': density*100})

                skimage.io.imsave(self.folder_name+'/'+os.path.splitext(f)[0]+'_label.tif', synapse_label.astype(np.uint16))
                skimage.io.imsave(self.folder_name+'/'+os.path.splitext(f)[0]+'_region.tif', synapse_region.astype(np.uint16))
        with self.out_process:
            display(print('Finished processing'))   
        
    def do_zipping(self, b):
        """zip the output"""
        
        #save the summary file
        pd.DataFrame(self.density).to_csv(self.folder_name+'/summary.csv')
        
        subprocess.call(['tar', 'cfz', 'to_download.tar.gz', 'upload_data'])
        with self.out_zip:
            display(print('Finished zipping'))    
   

    def import_image(self, file):
        """Load file
        
        Parameters
        ----------
        file : str
            name of file to open
        Returns
        -------
        image: 3D numpy array
        
        """

        image = skimage.io.imread(file, plugin='tifffile')
        image = image[0,:,:,:,1]
        return image


    def find_synapse_area(self, image):
        """Finds large scale region where synapases are present
        
        Parameters
        ----------
        image : 3D numpy array

        Returns
        -------
        large_mask_dil: 3D numpy array
            mask of snyapase region
        
        """
        
        #smooth image on large scale to find region of synapses (to remove the dotted structures)
        image_gauss = skimage.filters.gaussian(image, 10)
        #create large mask of synapase region
        large_mask = image_gauss > skimage.filters.threshold_otsu(image_gauss)
        #find the largest region and define it as synapse region
        large_label = skimage.morphology.label(large_mask)
        regions = pd.DataFrame(skimage.measure.regionprops_table(large_label,properties=('label','area')))
        lab_to_keep = regions.sort_values(by = 'area', ascending = False).iloc[0].label
        large_mask2 = large_label == lab_to_keep
        #dilate the region to fill holes
        large_mask_dil = skimage.morphology.binary_dilation(large_mask2, np.ones((5,5,5)))

        return large_mask_dil

    def detect_synapses(self, image, synapse_region):
        """Finds large scale region where synapases are present
        
        Parameters
        ----------
        image : 3D numpy array
        synapse_region : 3D numpy array
            mask of synapse region

        Returns
        -------
        synapse_mask: 3D numpy array
            mask of synapses
        
        """
        
        #calculate a LoG image to highlight synapses
        image_log = ndi.gaussian_laplace(-image.astype(float), sigma=1)
        #within the synapse region calculate a threshold
        newth = skimage.filters.threshold_otsu(image_log[synapse_region])
        #create a synapse mask (and use the global mask too)
        synapse_mask = synapse_region*(image_log > newth)

        return synapse_mask
    
    
     

        
        
        
        
        
            
            