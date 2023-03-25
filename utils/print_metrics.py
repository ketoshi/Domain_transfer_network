import json
import numpy
import os

def scan_scores(dir):
    osl = os.listdir(dir)
    osl.sort()
    tot_list = {'ssim_viton':[],'lpips_viton':[],'psnr_viton':[],'ssim_f550k':[],'lpips_f550k':[],'psnr_f550k':[],'fid':[],'filename':[]}
    for x in osl:
        sc = os.path.join(dir,x,"scores.json")
        if os.path.isfile(sc): 
            #print('yes')
            f = open(sc)
            y = json.load(f) 
            for i in y:
                tot_list[i].append(y[i])
            tot_list['filename'].append(x)
        #else: print('no')
    return tot_list


def get_format_params(is_str=True, *vals):
    if len(vals) ==1: vals = vals[0]
    params = []
    for x in vals:
        params.append(x)
        params.append(dist)
        if not is_str:
            params.append(decimals)
    return params

def table_format_print(is_str, format_name, *params):
    if len(params) ==1: params = params[0]
    params = get_format_params(is_str, params)
    print(  format_name(*params)  )





tot_list = scan_scores(dir = '/home/isac/data/tensorboard_info')
dist     = 15
decimals = 4
format_str = "{:<{}} {:<{}} {:<{}} {:<{}} {:<{}} {:<{}} {:<{}} ".format
format_float = "{:<{}.{}f} {:<{}.{}f} {:<{}.{}f} {:<{}.{}f} {:<{}.{}f} {:<{}.{}f} {:<{}.{}f} ".format

table_format_print(True,format_str,    'ssim_viton','lpips_viton','psnr_viton','ssim_f550k','lpips_f550k','psnr_f550k','fid')
table_format_print(True,format_str,    'high','low','high','high','low','high','low')
#print('first 15k')
for i in range(len(tot_list['fid'])):
    #if i==5:print('now 255');print(tot_list['filename'][i])
    ssim_viton  = tot_list['ssim_viton'][i] 
    lpips_viton = tot_list['lpips_viton'][i]
    psnr_viton  = tot_list['psnr_viton'][i] 
    ssim_f550k  = tot_list['ssim_f550k'][i] 
    lpips_f550k = tot_list['lpips_f550k'][i]
    psnr_f550k  = tot_list['psnr_f550k'][i] 
    fid         = tot_list['fid'][i]     
    table_format_print( False, format_float,   ssim_viton,lpips_viton,psnr_viton,ssim_f550k,lpips_f550k,psnr_f550k,fid  )        

