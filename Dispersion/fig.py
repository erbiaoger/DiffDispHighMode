import numpy as np
import matplotlib.pyplot as plt
import pathlib

files = pathlib.Path('.').glob('*.dat')
save_path = pathlib.Path('./datasets/car_some_data/testA')
data_list = []

for file in files:
    #filename = 'guangu/2023-07-20-18-40-58-out.dat'
    with open(file, 'rb') as fid:
        D = np.fromfile(fid, dtype=np.float32)

    fs = D[10]
    dt = 1 / fs
    dx = D[13]
    nx = int(D[16])
    nt = int(fs * D[17])

    data = D[64:].reshape((nx, nt), order='F').T  # 使用Fortran顺序进行数据的reshape

    pre_data = data.copy()
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    # data_list.append(data)
    
    # data = np.array(data_list)
    # data = data[3]
    scale = 0.01

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axis("off")
    ax.imshow(data[:, 10:100], cmap='rainbow', aspect='auto', vmin=-scale, vmax=scale)
    ax.margins(0)
    fig.savefig(f'{save_path/file.stem}.png', bbox_inches='tight', pad_inches=0, dpi=40)
    plt.close()