import numpy as np
import sklearn.cluster as sc
from scipy import interpolate
from scipy.cluster.vq import whiten
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist, squareform

np.set_printoptions(suppress=True)

def modeSeparation(curves, modes=2):
    """curves_class.py made by Zhiyu Zhang JiLin University in 2023-12-07 11h.
    
    
    Parameters
    ----------
    curves : numpy array
        Array shape (n, 3), where n is the number of classes.
        curves[:, -1] is the class number
        curves[:, 0] is the x-axis index
        curves[:, 1] is the y-axis index
        
        Example:
        curves = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 1], [3, 3, 1]])

    modes : int, optional
        The number of modes to separate the curves into, by default 2

    Returns
    -------
    out : ndarray
        Array shape (n, m+1), where n is the number of curves and m is the number of modes.
        The last column of the array contains the cluster labels for each curve.

    """
    curve_whiten = whiten(curves[:,0:2])      
    cluster_pred = sc.AgglomerativeClustering(n_clusters=int(modes),linkage='single',compute_full_tree=True).fit_predict(curve_whiten)
    
    m_value = np.zeros(modes)
    
    for mode in range(modes):
        m_value[mode] = np.mean(curves[cluster_pred==mode, 0]/1500*1.5)+ np.mean(curves[cluster_pred==mode, 1]/9)
    
    m_c = np.vstack([m_value, np.arange(modes)])

    m_c = m_c[:, m_c[0,:].argsort()]

    cluster_out = cluster_pred.copy()
    for mode in range(modes):

        cluster_out[cluster_pred==m_c[1,mode]] = mode
    
    if curves.shape[1] == 2 or curves.shape[1] == 4:
        out = np.column_stack([curves, cluster_out])
    
    else:
        out = np.column_stack([curves[:, 0:-1], cluster_out])
    
    out = out[np.argsort(out[:,-1])]
    for mode in range(modes):
        curveInMode = out[out[:,-1] == mode]
        out[out[:,-1] == mode] = curveInMode[np.argsort(curveInMode[:,0])]
    
    return out
    
def sortCurve(curve):
    modes = int(max(curve[:,-1])) + 1
    out = curve[np.argsort(curve[:,-1])]
    for mode in range(modes):
        curveInMode = out[out[:,-1] == mode]
        out[out[:,-1] == mode] = curveInMode[np.argsort(curveInMode[:,0])]
    
    return out
    
#TAG: autoSeparation
def autoSeparation(curves, to=0.04, maxMode=10):
    """
    Automatically separates curves based on certain criteria.

    Parameters:
    ----------
    curves : numpy.ndarray
        The input curves data.
    to : float, optional
        The threshold parameter for jump range limit. Defaults to 0.04.
    maxMode : int, optional
        The maximum number of modes to consider. Defaults to 10.

    Returns:
    -------
    numpy.ndarray: The processed curve data after separation.
    """

    # 计算曲线数据中第一列的最大值和最小值
    fMax = max(curves[:,0])
    fMin = min(curves[:,0])

    # 计算曲线数据中第二列的最大值和最小值
    cMax = max(curves[:,1])
    cMin = min(curves[:,1])
    
    # 计算搜索起始点，基于第一列数据的范围
    fSearchStart = 0.05 * (fMax - fMin) + fMin

    # 计算跳跃范围限制，基于第二列数据的范围和给定的阈值参数'to'
    cJumpRangeLimit = to * (cMax - cMin)

    # 初始化退出标志
    exitFlag = False

    # 循环遍历直到最大模式数
    for modePre in range(int(maxMode)):
        # 对曲线进行模式分离处理，可能是一种数据分析或处理
        curvePre = modeSeparation(curves, int(modePre)+1)

        # 再次循环，处理每个模式
        for mode in range(modePre+1):
            # 从处理后的曲线中选择特定模式的数据
            curveInMode = curvePre[curvePre[:,-1] == mode]

            # 过滤出大于搜索起始点的数据
            curveInMode = curveInMode[curveInMode[:,0]> fSearchStart]

            # 根据第一列数据对曲线进行排序
            curveInMode = curveInMode[np.argsort(curveInMode[:,0])]
            
            # 如果当前模式的曲线在第二列数据上的变化标准差大于跳跃范围限制，则不退出
            if np.std(np.diff(curveInMode[:,1])) > cJumpRangeLimit:
                exitFlag = False
                break
            else:
                # 否则，设置退出标志为True
                exitFlag = True

        # 如果设置了退出标志，则跳出循环
        if exitFlag:
            break
    
    # 返回处理后的曲线数据
    return curvePre

def getClassNum(curves_km):
    """curves_class.py made by Zhiyu Zhang JiLin University in 2023-12-07 11h.
    Parameters
    ----------
    curves_km : numpy array
        Array shape (n, 3), where n is the number of classes.
        curves_km[:, -1] is the class number
        curves_km[:, 0] is the x-axis index
        curves_km[:, 1] is the y-axis index
        
        >>> curves_km = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 1], [3, 3, 1]])

    Returns
    -------
    class_num : ndarray
        The number of classes

    """
    unique_list = list(dict.fromkeys(curves_km[...,-1]))        # 使用dict.fromkeys()去除重复元素
    class_num = len(unique_list)
    
    return class_num


def pickPoints(data, Nch, Nt, skip_Nch = 2, skip_Nt = 2, threshold = 17, model='max'):

    """car_class_success.ipynb made by Zhiyu Zhang JiLin University in 2023-12-07 17h.
    
    pick the points of the curves
    
    Parameters
    ----------
    data : numpy array
        Array shape (Nch, Nt)
    Nch : int
        Number of channels
    Nt : int
        Number of time samples
    skip_Nch : int, optional.  Default is 2
        Number of channels to skip
    skip_Nt : int, optional.  Default is 2
        Number of time samples to skip
    threshold : float, optional.  Default is 17
        Threshold value
    model : {'max', 'min'}, optional.  Default is 'max'
        Model to use, Take the maximum or minimum value

    Returns
    -------
    curves : ndarray
        Array shape (Ncurves, 2)
        Ncurves is the number of curves
        The first column is the time sample
        The second column is the channel

    """
    if Nch != data.shape[1] or Nt != data.shape[0]:
        data = data.T
    
    if model == 'max':
        curves = []
        for i in range(0, Nch, skip_Nch):
            flag = 0
            for j in range(0, Nt, skip_Nt):
                if data[j, i] > threshold and flag == 0:
                    flag = 1
                    begin_j = j

                elif flag == 1 and data[j, i] < threshold:
                    mid_j = begin_j + (j - begin_j)/2
                    curves.append([int(i), int(mid_j)])
                    flag = 0
                        
    if model == 'min':
        # curves = []
        # for i in range(0, Nch, skip_Nch):
        #     flag = 0
        #     for j in range(0, Nt, skip_Nt):
        #         if data[i, j] < threshold and flag == 0:
        #             flag = 1
        #             begin_j = j

        #         elif flag == 1 and data[i, j] > threshold:
        #             mid_j = begin_j + (j - begin_j)/2
        #             curves.append([int(mid_j), int(i)])
        #             flag = 0
        curves = []
        for i in range(0, Nch, skip_Nch):
            flag = 0
            for j in range(0, Nt, skip_Nt):
                if data[j, i] < threshold and flag == 0:
                    flag = 1
                    begin_j = j

                elif flag == 1 and data[j, i] > threshold:
                    mid_j = begin_j + (j - begin_j)/2
                    curves.append([int(i), int(mid_j)])
                    flag = 0

    return np.array(curves)

# def pickPointPlus(spec, threshold, freq, velo, searchStep=5, returnSpec=False):
#     fMax = max(freq)
#     fMin = min(freq)
#     cMax = max(velo)
#     cMin = min(velo)

#     # 	Preprocessing
#     spec = np.flipud(spec)
#     spec = tf.convert_to_tensor(spec)
#     spec = tf.expand_dims(spec, -1)
#     spec = tf.image.resize(spec, (512, 512))
#     spec = tf.expand_dims(spec, 0)

#     #	CAE Predict
#     output = model_das.predict(spec)
#     output = np.squeeze(output)

#     if np.ndim(output) < 3:
#         raise TypeError('The network type Error, the PLUS model is needed!')
        
#     outputMode = output[:,:,1]
#     output = output[:,:,0]

#     if returnSpec:
#         return output
#     #	Grid Search	

#     point = []
#     for f in range(0, 512, searchStep):
#         flag = 0
#         for c in range(0, 512):
#             if output[c,f] > threshold and flag == 0:
#                 flag = 1
#                 begin_c = c
        
#             elif flag == 1 and output[c,f] < threshold:
#                 mid_c = begin_c + (c - begin_c)/2
#                 modeNum =  np.around(5-outputMode[int(mid_c), f])
#                 if modeNum<1:
#                     modeNum = 0
#                 point.append([mid_c, f, modeNum])
#                 flag = 0
                
#     point = np.array(point)
#     if len(point) == 0:
#         return []

#     point[:,0] = cMax - point[:,0] * (cMax - cMin)/512

#     point[:,1] = fMin + point[:,1] * (fMax - fMin)/512
#     tmp = point[:,0].copy()
#     point[:,0] = point[:,1].copy()
#     point[:,1] = tmp
#     #	del tmp
#     print(point)
#     return point





def showClass(data, curves_km, id_list, t, x, ax, 
              s='a)', title="a) Raw data with classification", 
              model='vel', velocities=None,
              vmin=1, vmax=1):
    """Display the classification of curves on a plot.

    Parameters
    ----------
    data : numpy array
        Array of shape (Nch, Nt) representing the data.
    curves_km : numpy array
        Array of shape (N, 3) representing the curves in kilometers.
    id_list : list
        List of curve IDs to be displayed.
    t : numpy array
        Array of shape (Nt,) representing the time values.
    x : numpy array
        Array of shape (Nx,) representing the distance values.
    ax : matplotlib.axes.Axes
        The axes on which to plot the data.
    s : str, optional
        The label for the plot, by default 'a)'.
    model : str, optional
        The type of model to display ('vel' for velocity or 'class' for class), by default 'vel'.
    velocities : dict, optional
        A dictionary mapping curve IDs to velocities in km/h, by default None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    """
    print(vmin, vmax)
    letter_params = {
        "fontsize": 10,
        "verticalalignment": "top",
        "horizontalalignment": "left",
        "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}}
    
    color = ['orange', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'pink', 'brown', 'purple', 'gray']

    ax.imshow(data, extent=[0, x.max(), t.max(), 0], aspect="auto", 
              cmap="rainbow", vmin=vmin, vmax=vmax)
    ax.text(x=0, y=0, s=s, **letter_params)
    if model == 'vel' and len(velocities) != 0:
        for i, id in enumerate(id_list, 0):
            curve_in_mode = curves_km[curves_km[:,-1] == id]
            if velocities[id] > 0:
                ax.scatter(x[curve_in_mode[...,0]], t[curve_in_mode[...,1]], s=10, 
                           label=r"$\Longleftarrow$"+' car'+str(i)+f" {np.abs(velocities[id]):.2f} km/h", 
                           c=color[i])
            else:
                ax.scatter(x[curve_in_mode[...,0]], t[curve_in_mode[...,1]], s=10, 
                           label=r"$\Longrightarrow$" + ' car'+str(i)+f" {np.abs(velocities[id]):.2f} km/h", 
                           c=color[i])
    if model == 'class':
        for i, id in enumerate(id_list, 0):
            curve_in_mode = curves_km[curves_km[:,-1] == id]
            ax.scatter(x[curve_in_mode[...,0]], t[curve_in_mode[...,1]], s=10, label='class '+str(i), c=color[i])
        
    ax.legend()
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Distance (km)")
    ax.set_title(title)
    
    return ax


def deleteSmallClass(curves_km, class_num, minCarNum=10):
    '''
    delete the minimum value of each curve
    '''
    for ii in range(class_num):
        num = len(curves_km[curves_km[...,-1] == ii])
        print('class '+str(ii)+': '+str(num))
        if num < minCarNum:
            curves_km = np.delete(curves_km, np.where(curves_km[...,-1] == ii), axis=0)

    return curves_km



def getVelocity(curves_km, x, t, id_list):
    
    velocities = {}
    id_list_a = []
    for i, id in enumerate(id_list, 0):
        print(f"Class {i}: ")
        curve_in_mode = curves_km[curves_km[:,-1] == id]
        
        # 示例数据
        X = x[curve_in_mode[:, 0]]  # 自变量
        Y = t[curve_in_mode[:, 1]]  # 因变量

        # 将自变量数据转换为二维数组，因为scikit-learn需要输入为二维数组
        X = X.reshape(-1, 1)

        # 创建并拟合线性回归模型
        model = LinearRegression()
        model.fit(X, Y)

        # 获取回归方程的斜率和截距
        slope = 1/model.coef_[0]
        intercept = model.intercept_

        print('\t斜率: ', slope)
        print('\t截距: ', intercept)
        vel = slope*3.6 
        print('\t速度: ', vel, 'km/h')
        
        if np.abs(vel) > 0 and np.abs(vel) < 1000:
            id_list_a.append(id)
            velocities[id] = vel

    return velocities, id_list_a



def classCar(curves_km, id_list, scale=0.6):

    id_list_b = []
    # 分析每个聚类
    for i, id in enumerate(id_list, 0):
        cluster_data = curves_km[curves_km[:,-1] == id]
        
        # 计算聚类内部的所有点对的距离
        distances = pdist(cluster_data, 'euclidean')
        distance_matrix = squareform(distances)

        # 分析距离
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # 定义阈值来区分线和团
        # 这里的阈值需要根据实际数据进行调整
        if std_distance / mean_distance > scale:
            print(f"Cluster {i} is more likely a line.")
            id_list_b.append(id)
        else:
            print(f"Cluster {i} is more likely a cluster.")
        
    return id_list_b
