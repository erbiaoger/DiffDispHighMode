
# %matplotlib inline
import sys
import numpy as np
import matplotlib.pyplot as plt
import pylops
from PIL import Image

from functools import partial
from scipy.optimize import minimize, Bounds
from disba import PhaseDispersion

sys.path.append('../')

from Dispersion.surfacewaves import *
from Dispersion.dispersionspectra import *
from Dispersion.inversion import *

import ccfj
import scipy
from Dispersion.Dispersion.dispersion import get_dispersion


def get_cpr(thick, vs, period):
    
    true_model = np.vstack([thick, vs*4, vs, np.ones_like(vs)]).T

    # Rayleigh-wave fundamental model dispersion curve 
    pd = PhaseDispersion(*true_model.T)
    cpr = [pd(period, mode=imode, wave="rayleigh") for imode in range(3)]

    return cpr

def random_thick_vs(thick, vs, period, fluctuation_percentage=0.1):
    # 生成浮动值
    random_thick = thick * (1 + fluctuation_percentage * (2 * np.random.rand(len(thick)) - 1))
    random_vs = vs * (1 + fluctuation_percentage * (2 * np.random.rand(len(vs)) - 1))

    try:
        cpr = get_cpr(random_thick, random_vs, period)
        # plt.plot(1/cpr[0][0], cpr[0][1], 'k', lw=2, label='Original')
        return cpr
    except Exception as e:
        print(e)

def get_dshift(nt, dt, nx, dx, nfft, cpr):
    t, x = np.arange(nt)*dt, np.arange(nx)*dx

    # Wavelet
    wav = ormsby(t[:nt//2+1], f=[2, 4, 38, 40], taper=np.hanning)[0][:-1]
    wav = np.roll(np.fft.ifftshift(wav), 20) # apply small shift to make it causal

    # Data
    dshifts, fs, vfs = [], [], []
    for imode in range(3):
        dshift_, f_, vf_ = surfacewavedata(nt, dt, nx, dx, nfft, 
                                        np.flipud(1/cpr[imode][0]), np.flipud(cpr[imode][1]), wav)
        dshifts.append(1./(imode+1)**0.8 * dshift_[np.newaxis])
        fs.append(f_)
        vfs.append(vf_)
    dshift = np.concatenate(dshifts).sum(0)
    return dshift

def park(dshift, dx, dt, cmin, cmax, dc, fmin, fmax):
    f1, c1, img, U, t = get_dispersion(dshift.T, dx, dt, 
                                        cmin, cmax, dc, fmin, fmax)

    return f1, c1, img, U, t

def fj(dshift, dx, dt, cmin, cmax):
    nx, nt = dshift.shape
    x = np.arange(nx)*dx
    f = scipy.fftpack.fftfreq(nt,dt)[:nt//2]
    c = np.linspace(cmin, cmax, 1000)

    out = ccfj.fj_earthquake(dshift,x,c,f,fstride=1,itype=0,func=0)
    
    return f, c, out

def show_fj(f, c, out, fmin, fmax, ii, aa):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axis("off")
    ax.imshow(out, aspect='auto', cmap='gray',
            extent=(f.min(), f.max(),c.min(), c.max()),origin='lower')

    ax.margins(0)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(c.min(), c.max())
    fig.savefig(f'/home/lty/MyProjects/Seismology/diffseis/dataset/demultiple/data_train/test/{aa}{ii:03d}.png', 
                dpi=300,bbox_inches='tight', pad_inches=0)
    plt.close()

def show_label(f, c, out, cpr, fmin, fmax, ii, aa):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axis("off")
    ax.imshow(np.zeros_like(out), aspect='auto', cmap='gray',
            extent=(f.min(), f.max(),c.min(), c.max()),origin='lower')
    for imode in range(3):
        ax.plot(np.flipud(1/cpr[imode][0]), 1.e3*np.flipud(cpr[imode][1]), 
                    'white', lw=4)

    ax.margins(0)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(c.min(), c.max())

    # fig.savefig(f'/home/lty/MyProjects/Seismology/diffseis/dataset/demultiple/data_train/labels/{aa}{ii:03d}.png', 
    #             dpi=300,bbox_inches='tight', pad_inches=0)
    plt.close()


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def showOutput(out, cpr, f, c, fmin, fmax):
    from scipy.ndimage import zoom

    x_start = zoom(out, (128/out.shape[0], 128/out.shape[1]))

    x_start = torch.unsqueeze(torch.tensor(x_start), dim=0)
    x_start = torch.unsqueeze(x_start, dim=0)

    out1 = diffusion.inference(x_in=x_start.cuda())

    in_samples = out1[0,0].cpu().detach().numpy()
    out_samples = out1[1,0].cpu().detach().numpy()
    print(out_samples.shape)


    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    axes[0].imshow(out, aspect='auto', cmap='jet',
        extent=(f.min(), f.max(),c.min(), c.max()),origin='lower')
    axes[0].set_xlim(fmin, fmax)
    axes[0].set_ylim(c.min(), c.max())


    axes[1].imshow(np.zeros_like(out), aspect='auto', cmap='gray',
        extent=(f.min(), f.max(),c.min(), c.max()),origin='lower')
    for imode in range(3):
        axes[1].plot(np.flipud(1/cpr[imode][0]), 1.e3*np.flipud(cpr[imode][1]), 
                    'white', lw=4)
    axes[1].set_xlim(fmin, fmax)
    axes[1].set_ylim(c.min(), c.max())

    axes[2].imshow(out_samples, cmap="Greys_r", aspect='auto', extent=(f.min(), f.max(),c.min(), c.max()))
    # axes[2].set_title("Input "+str(i))
    # axes[2].set_xlim(fmin, fmax)
    # axes[2].set_ylim(c.min(), c.max())



