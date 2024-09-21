for h1 in np.linspace(0.005, 0.02, 3):
    for h2 in np.linspace(0.01, 0.04, 3):
        for h3 in np.linspace(0.02, 0.06, 3):
            # for h4 in np.linspace(0.03, 0.07, 3):
            for v1 in np.linspace(0.1, 0.3, 2):
                for v2 in np.linspace(0.3, 0.6, 3):
                    for v3 in np.linspace(0.3, 0.7, 3):
                        for v4 in np.linspace(0.7, 0.9, 3):
                            thick = np.array([h1, h2, h3, h4])
                            vs = np.array([v1, v2, v3, v4])
                            for ii in range(3):
                                try:
                                    # 生成浮动值
                                    cpr = random_thick_vs(thick, vs, period, fluctuation_percentage)
                                    # 生成合成面波数据
                                    dshift = get_dshift(nt, dt, nx, dx, nfft, cpr)
                                    # F-J
                                    f, c, out = fj(dshift, dx, dt, cmin, cmax)
                                    
                                    aa = str(h1)+'_'+str(h2)+'_'+str(h3)+'_'+str(h4)+'_'+str(v1)+'_'+str(v2)+'_'+str(v3)+'_'+str(v4)
                                    
                                    show_fj(f, c, out, fmin, fmax, ii, aa)
                                    show_label(f, c, out, cpr, fmin, fmax, ii, aa)
                                except Exception as e:
                                    print(e)
                                    continue
    concurrent.futures.wait(futures)