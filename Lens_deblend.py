#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Packages and setup
import numpy as np
import scarlet
import MuSCADeT as wine
from MuSCADeT  import colour_subtraction as cs
import sep
import scipy.signal as scp

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
from scarlet_extensions.initialization.detection import mad_wavelet, Data

import astropy.io.fits as fits
from astropy.wcs import WCS
import csv

# use a good colormap and don't interpolate the pixels
matplotlib.rc('image', cmap='inferno', interpolation='none', origin='lower')


# In[ ]:


group = 'Group2'

if group == 'Group1':
    cat = open(group+'/group1.csv')
    filenames = [] 
    ids = []
    for row in cat:
        filenames.append(row.split(',')[1])
        ids.append(filenames[-1][4:12])
elif group == 'Group2':
    cat = open(group+'/final_SL_b80-90.csv')
    filenames = [] 
    ids = []
    for row in cat:
        filenames.append(row.split(',')[19])
        ids.append(filenames[-1][4:12])
elif group == 'Maybe_lenses':
    cat = open(group+'/ML_final.csv')
    filenames = [] 
    ids = []
    for i, row in enumerate(cat):
        filenames.append(row.split(',')[21])
        ids.append(filenames[-1][4:12])


# In[ ]:


model_psf = scarlet.GaussianPSF(sigma = [[.5, 0.5]])
filters = ['g','r','i']

#Scarlet plots
norm = scarlet.display.AsinhMapping(minimum=-1, stretch=50, Q=10)
norm_psf = scarlet.display.AsinhMapping(minimum=0, stretch=0.01, Q=10)

def makeCatalog(datas, thresh = 3, lvl=3, wave=True):
    ''' Creates a detection catalog by combining low and high resolution data

    This function is used for detection before running scarlet.
    It is particularly useful for stellar crowded fields and for detecting high frequency features.

    Parameters
    ----------
    datas: array
        array of Data objects
    lvl: int
        detection lvl
    wave: Bool
        set to True to use wavelet decomposition of images before combination

    Returns
    -------
    catalog: sextractor catalog
        catalog of detected sources
    bg_rms: array
        background level for each data set
    '''
    if len(datas) == 1:
        hr_images = datas[0].images / np.sum(datas[0].images, axis=(1, 2))[:, None, None]
        # Detection image as the sum over all images
        detect_image = np.sum(hr_images, axis=0)
    else:
        data_lr, data_hr = datas
        # Create observations for each image
        # Interpolate low resolution to high resolution
        interp = interpolate(data_lr, data_hr)
        # Normalisation of the interpolate low res images
        interp = interp / np.sum(interp, axis=(1, 2))[:, None, None]
        # Normalisation of the high res data
        hr_images = data_hr.images / np.sum(data_hr.images, axis=(1, 2))[:, None, None]
        # Detection image as the sum over all images
        detect_image = np.sum(interp, axis=0) + np.sum(hr_images, axis=0)
        detect_image *= np.sum(data_hr.images)
    if np.size(detect_image.shape) == 3:
        if wave:
            # Wavelet detection in the first three levels
            wave_detect = scarlet.Starlet.from_image(detect_image.mean(axis=0), scales=lvl+1).coefficients
            wave_detect[:, -1, :, :] = 0
            detect = scarlet.Starlet(coefficients=wave_detect).image
        else:
            # Direct detection
            detect = detect_image.mean(axis=0)
    else:
        if wave:
            wave_detect = scarlet.Starlet.from_image(detect_image, scales=lvl+1).coefficients
            detect = np.mean(wave_detect[:-1], axis=0)
        else:
            detect = detect_image

    bkg = sep.Background(detect)
    catalog = sep.extract(detect, thresh, err=bkg.globalrms)

    if len(datas) ==1:
        bg_rms = mad_wavelet(datas[0].images)
    else:
        bg_rms = []
        for data in datas:
            bg_rms.append(mad_wavelet(data.images))

    return catalog, bg_rms



def make_obs(images, psf, wcs):
    
    data =  Data(images, wcs, psf, filters)
    catalog, bg_rms = makeCatalog([data], lvl =0, thresh = 1, wave=True)

    weights = np.ones_like(images) / (bg_rms**2)[:, None, None]
    
    model_frame = scarlet.Frame(
        images.shape,
        psf=model_psf,
        channels=filters)

    observation = scarlet.Observation(
        images, 
        psf=scarlet.ImagePSF(psf), 
        weights=weights, 
        channels=filters).match(model_frame)
    return model_frame, observation, catalog

def make_sources(observation, model_frame, catalog):
    starlet_sources = []
    
    n,n1,n2 = observation.data.shape
    pixels = np.stack((catalog['y'], catalog['x']), axis=1)
    if np.size(pixels)==0:
        pixels=np.array([[n1/2., n2/2.]])
        
    r = np.sqrt(np.sum((pixels-np.array([n1/2., n2/2.]))**2, axis = 1))
    lens = pixels[r == np.min(r)]
    
    sources = []
    if np.size(catalog['y']) == 0:
        new_source = scarlet.ExtendedSource(model_frame, 
                                            (n1/2., n2/2.), 
                                            observation, 
                                            K=1,
                                            compact = 1)
        sources.append(new_source)
    
    for k,src in enumerate(catalog):
        
        new_source = scarlet.ExtendedSource(model_frame, 
                                            (src['y'], src['x']), 
                                            observation, 
                                            K=1,
                                            compact = 1)
        sources.append(new_source)
    
    new_source = scarlet.StarletSource(model_frame, 
                                       (n1/2., n2/2.), 
                                       [observation], 
                                       spectrum = np.array([1.,1.,0.5]),
                                       starlet_thresh = 0.1)
    sources.append(new_source)
    print(np.size(sources))
    blend = scarlet.Blend(sources, observation)
    return sources, blend


def run_scarlet(blend, sources):
    blend.fit(200, e_rel = 1.e-6) #Set iterations to 200 for better results
    print("scarlet ran for {0} iterations to logL = {1}".format(len(blend.loss), -blend.loss[-1]))
    plt.plot(-np.array(blend.loss))
    plt.xlabel('Iteration')
    plt.ylabel('log-Likelihood')
    
    scarlet.display.show_scene(sources, 
                           norm=norm, 
                           observation=observation, 
                           show_rendered=True, 
                           show_observed=True, 
                           show_residual=True,
                          )
    scarlet.display.show_sources(sources,  
                             norm = norm,
                             observation=observation,
                             show_rendered=True, 
                             show_observed=True,
                             add_boxes=True
                            )
    plt.show()
    
def run_MuSCADeT(images, psf,A):
    images = images[::-1]
    psf = psf[::-1]
    S, An = wine.MCA.mMCA(images, A.T, 5, 200, mode = 'None', PSF=psf, plot = True, PCA= [2,50])
    n, n1,n2 = np.shape(images)
    A=An
    
    # Models as extracted by MuSCADeT for display
    model = np.dot(A,S.reshape([A.shape[1], n1*n2])).reshape(images.shape)
    for i in range(n):
        model[i] = scp.fftconvolve(model[i], psf[i], mode = 'same')
    
    normodel = cs.asinh_norm(model, Q=20, range = 50)
    normcube = cs.asinh_norm((images), Q = 20, range = 50)
    normres = cs.asinh_norm(images-model, Q = 10, range = 50)
    plt.figure(figsize = (15, 5))
    plt.subplot(131)
    plt.title('model')
    plt.imshow(normodel)
    plt.subplot(132)
    plt.title('data')
    plt.imshow(normcube)
    plt.subplot(133)
    plt.title('Residuals')
    plt.imshow(normres)
    plt.show()
    
    for i in range(A.shape[1]):
        
        C = A[:,i, np.newaxis, np.newaxis]*S[np.newaxis,i,:,:]
        for j in range(n):
            C[j] = scp.fftconvolve(C[j], psf[j], mode = 'same')
        normC = cs.asinh_norm(C, Q = 20, range = 50)
        normCres = cs.asinh_norm((images-C), Q = 20, range = 50)
        if i == 0:
            red = images-C
            red_model = C
        else:
            blue = images-C
            blue_model = C
        plt.figure(figsize = (15, 5))
        plt.subplot(131)
        plt.title('data')
        plt.imshow(normcube)
        plt.subplot(132)
        plt.title('component ' + str(i))
        plt.imshow(normC)
        plt.subplot(133)
        plt.title('data - component ' + str(i))
        plt.imshow(normCres)
        plt.show()
        
    image = images
    residuals = images-model
    return image, red, blue, red_model, blue_model, residuals


# In[ ]:



images_tab = []
reds = []
blues = []
red_models = []
blue_models = []
residuals = []
files = []
for i,f in enumerate(filenames[1:]):
    try:
        print(f)
        hdu = fits.open(group+'/data/'+f)
    except:
        print('failed file:', f)
        continue
    images = hdu[0].data
    wcs = WCS(hdu[0].header)
    psf_tab = []
    images = []
    size = []
    for j,n in enumerate(['g','r','i']):
        images.append(hdu[j].data)
        psf_hdu = fits.open(group+'/psf/'+n+'/PSF_'+ids[i+1]+'.fits')
        p = []
        c = 0
        while 1:
            try: 
                p.append(psf_hdu[c].data) 
            except:
                break
            c+=1
        size.append(p[-1].shape[0])
        psf_tab.append(p)
    npsf = np.max(size)
    n1,n2 = np.shape(hdu[j].data)
   
    psf = np.zeros((3, npsf, npsf))

    for j, p in enumerate(psf_tab):
        psf[j, 
            np.floor((npsf-size[j])/2.).astype(int):npsf-np.floor((npsf-size[j])/2.).astype(int),
            np.floor((npsf-size[j])/2.).astype(int):npsf-np.floor((npsf-size[j])/2.).astype(int)] =  p
    psf = np.array(psf)
    images = np.array(images)
    images_rgb = scarlet.display.img_to_rgb(images, norm=norm)
    psf_rgb = scarlet.display.img_to_rgb(psf, norm=norm_psf)
    
    plt.subplot(121)
    plt.imshow(images_rgb)
    plt.subplot(122)
    plt.imshow(psf_rgb)
    plt.show()
    
    frame, observation, cat = make_obs(images, psf, wcs)
    sources, blend = make_sources(observation, frame, cat)
    run_scarlet(blend, sources)
    
    psf = observation.renderer.diff_kernel._image
    A = []
    
    bluer = np.array([0.667,0.333,0])
    redder = np.array([0,0.333,0.667])
    spectrum = []
    rs = []
    for i,s in enumerate(sources):
        spec = s.get_model().sum(axis=(1, 2))
        spectrum.append(spec/np.sum(spec))
        origin = [s.bbox.origin[-2]+s.bbox.shape[-2]/2, s.bbox.origin[-1]+s.bbox.shape[-1]/2]
        rs.append(np.sqrt((n1/2.-origin[-2])**2+(n2/2.-origin[-1])**2))
    
    blue = np.sum(bluer * spectrum, axis = 1)
    red = np.sum(redder * spectrum, axis = 1)
    rs = rs[:-1]
    #Blue spectra
    
    
    if np.argmax(blue) == np.argmin(rs):
        rs[np.argmin(rs)] += n1/2
    print(rs, np.argmin(rs))
    #Red spectra
    A.append(spectrum[np.argmax(red)])#np.argmin(rs)])
    A.append(spectrum[np.argmax(blue)])

    image, red, blue, red_model, blue_model, res = run_MuSCADeT(images, psf, np.array(A)[:,::-1])
    images_tab.append(image)
    reds.append(red)
    blues.append(blue)
    red_models.append(red_model)
    blue_models.append(blue_model)
    residuals.append(res)
    files.append(f)


# In[ ]:


import pickle
f = open("MuSCADeT_models_"+group+".pkl","wb")
pickle.dump([files, images, blues, reds, blue_models, red_models, residuals], f)
f.close()


# In[ ]:


x = pickle.load(open("MuSCADeT_models_"+group+".pkl", "rb" ))


# In[ ]:


print(np.size(x[0]))


# In[ ]:


print(files)


# In[ ]:




