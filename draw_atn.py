import matplotlib.pyplot as plt
import numpy as np
import os


def get_atn_data_file_path(file_dir):
    data_file = []
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        if (path.endswith('.dat')):
            data_file.append(path)
    return data_file

def process_points(ptsL, ptsR):
    min_xl, min_yl, max_xl, max_yl = 999.9, 999.9, -999.9, -999.9
    min_xr, min_yr, max_xr, max_yr = 999.9, 999.9, -999.9, -999.9
    n = len(ptsR)
    for ptL, ptR in zip(ptsL, ptsR):
        xl,yl,xr,yr = ptL[0], ptL[1], ptR[0], ptR[1]
        if xl > max_xl:
            max_xl = xl
        if xl < min_xl:
            min_xl = xl
        if yl > max_yl:
            max_yl = yl
        if yl < min_yl:
            min_yl = yl
        if xr > max_xr:
            max_xr = xr
        if xr < min_xr:
            min_xr = xr
        if yr > max_yr:
            max_yr = yr
        if yr < min_yr:
            min_yr = yr
    for i in range(n):
        ptsL[i][0] -= min_xl
        ptsL[i][1] -= min_yl
        ptsR[i][0] -= min_xr
        ptsR[i][1] -= min_yr
    
    hr = max_yr - min_yr + 1.0
    for i in range(n):
        ptsR[i][1] -= hr

    wl, wr = max_xl - min_xl, max_xr - min_xr
    if wl < wr:
        wl = wr
    
    return ptsL, ptsR, wl + 0.1

def process_atn_val(atn_data):
    atn_data.sort(key=lambda x:x[1], reverse = True)
    n = len(atn_data)
    for i in range(n):
        val = pow(0.5,i) + 0.15
        if val > 1.0:
            val = 0.9999
        atn_data[i][1] = val
    return atn_data


def draw_atn(pts_file = 'test_atn.rst', atn_file = 'imgR_selfatn_layer6_head1.dat', indices = [65, 46, 194,121]):
    if(os.path.exists(pts_file) and os.path.exists(atn_file)):
        atn_data = []
        pt_IDs = []
        pts_L = []
        pts_R = []

        with open(pts_file, "r") as ifp1:
            lines = ifp1.readlines()
            for line in lines:
                str_nums = line.replace('\\n','').split()
                pt_ID, xl, yl, xr, yr = int(str_nums[0]), float(str_nums[1]), \
                    float(str_nums[2]),float(str_nums[3]),float(str_nums[4])
                pt_IDs.append(pt_ID)
                pts_L.append([xl,yl])
                pts_R.append([xr,yr])

        with open(atn_file, "r") as ifp2:
            lines = ifp2.readlines()
            for line in lines:
                str_nums = line.replace('\\n','').split()
                n_nums = len(str_nums)//2
                pt_atn = []
                for n in range(n_nums):
                    pt_index, atn_v = int(str_nums[2*n]), float(str_nums[2*n+1])
                    pt_atn.append([pt_index,atn_v])
                atn_data.append(pt_atn)
        
        atn_file = atn_file.split('\\')[1]
        s_filename = atn_file.split('_')
        if(s_filename[1] == 'selfatn'):#plot illustration figure of self attention
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            n_pts = len(pt_IDs)
            for i in range(n_pts):
                ID = pt_IDs[i]
                if(s_filename[0] == 'imgL'):
                    pt = pts_L[i]
                else:
                    pt = pts_R[i]
                if(ID<0):
                    circ = plt.Circle(pt, 0.01, color='r', alpha=0.8)
                else:
                    circ = plt.Circle(pt, 0.01, color='g', alpha=0.8)
                ax.add_patch(circ)

            for index in indices:
                ID = pt_IDs[index]
                atns = process_atn_val(atn_data[index])
                if(s_filename[0] == 'imgL'):
                    pt = pts_L[index]
                else:
                    pt = pts_R[index]
                for atn in atns:
                    atn_index, atn_val = atn[0], atn[1]
                    if(s_filename[0] == 'imgL'):
                        atn_pt = pts_L[atn_index]
                    else:
                        atn_pt = pts_R[atn_index]
                    if ID > 0:
                        cir = plt.Circle(pt, 0.05, color='g', alpha=0.9)
                        ln = plt.Line2D([pt[0],atn_pt[0]],[pt[1], atn_pt[1]],color='g',alpha=atn_val,linewidth=0.75)
                    else:
                        cir = plt.Circle(pt, 0.05, color='r', alpha=0.9)
                        ln = plt.Line2D([pt[0],atn_pt[0]],[pt[1], atn_pt[1]],color='r',alpha=atn_val,linewidth=0.75)
                    
                    ax.add_line(ln)
                    ax.add_patch(cir)
            ax.set_xticks([])
            ax.set_yticks([])
            sp_name = atn_file.split('.')
            # ax.set_title(sp_name[0])
            plt.axis('scaled')
            plt.savefig(sp_name[0] + '.jpg', dpi=800, bbox_inches='tight')
        
        else: # plot illustration figure of cross attetion
            pts_L, pts_R, w = process_points(pts_L, pts_R)
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            n_pts = len(pt_IDs)
            for i in range(n_pts):
                ID = pt_IDs[i]
                if(ID<0):
                    ptL = pts_L[i]
                    circL = plt.Circle(ptL, 0.01, color='r', alpha=0.8)
                    ptR = pts_R[i]
                    circR = plt.Circle(ptR, 0.01, color='r', alpha=0.8)
                else:
                    ptL = pts_L[i]
                    circL = plt.Circle(ptL, 0.01, color='g', alpha=0.8)
                    ptR = pts_R[i]
                    circR = plt.Circle(ptR, 0.01, color='g', alpha=0.8)
                ax.add_patch(circL)
                ax.add_patch(circR)
            
            # ln = plt.Line2D([0,w],[-0.5, -0.5],color='b',alpha=0.8,linewidth=0.5)
            # ax.add_line(ln)
            
            for index in indices:
                ID = pt_IDs[index]
                atns = process_atn_val(atn_data[index])
                if(s_filename[0] == 'imgL'):
                    pt = pts_L[index]
                else:
                    pt = pts_R[index]
                for atn in atns:
                    atn_index, atn_val = atn[0], atn[1]
                    if(s_filename[0] == 'imgL'):
                        atn_pt = pts_R[atn_index]
                    else:
                        atn_pt = pts_L[atn_index]
                    if ID > 0:
                        cir = plt.Circle(pt, 0.05, color='g', alpha=0.9)
                        ln = plt.Line2D([pt[0],atn_pt[0]],[pt[1], atn_pt[1]],color='g',alpha=atn_val,linewidth=0.75)
                    else:
                        cir = plt.Circle(pt, 0.05, color='r', alpha=0.9)
                        ln = plt.Line2D([pt[0],atn_pt[0]],[pt[1], atn_pt[1]],color='r',alpha=atn_val,linewidth=0.75)
                    ax.add_line(ln)
                    ax.add_patch(cir)

            ax.set_xticks([])
            ax.set_yticks([])
            plt.axis('scaled')
            sp_name = atn_file.split('.')
            # ax.set_title(sp_name[0])
            plt.savefig(sp_name[0] + '.jpg', dpi=800, bbox_inches='tight')


atn_files = get_atn_data_file_path('atn_data')
for file in atn_files:
    draw_atn(atn_file=file)


    








# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)


# plt.subplot(221)
# rect = plt.Rectangle((-0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
# circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
# pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]], color='g', alpha=0.5)
 
# ax.add_patch(rect)
# ax.add_patch(circ)
# ax.add_patch(pgon)
# plt.show()

# plt.savefig('figpath.svg', dpi=400, bbox_inches='tight')