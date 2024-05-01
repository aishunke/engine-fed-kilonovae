#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:32:57 2023

@author: shunkeai
"""

import numpy as np
from mechanical_model import *
#from mechanical_model_copy import *
#from mechanical_model_non_conducting import *
import matplotlib.pyplot as plt
import argparse
import time
import os
import multiprocessing as mp
import imageio


tstart = time.time()
####### constant #######
ccont=2.99*10**10
m_p=1.67352e-24
Msun = 2.0e33
a = 7.5657e-15
h = 6.63e-27
kb=1.38*10**(-23)*10**7
sigmaT=6.652*10**(-25)
sigmaB=5.670373*10**(-5)
pi=np.pi

def create_gif(Path, image_list, gif_name, duration = 1.0):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(Path+'/'+image_name))
    imageio.mimsave(gif_name,frames,'GIF', duration = duration)
    return


def generate_input_mp(path):
    f = open(path+'/input_main.txt','r')
    Inputs_raw = f.readlines()
    Inputs = {}
    for i in range(len(Inputs_raw)):
         Input = split(Inputs_raw[i])
         Inputs[Input[0]] = np.linspace(float(Input[1]),float(Input[2]),int(Input[3]))
    f.close()




    M_ej_sun_array = Inputs.get('M_ej_sun')
    beta_min_array = Inputs.get('beta_min')
    beta_max_array = Inputs.get('beta_max')
    alpha_array = Inputs.get('alpha')
    kappa_array = Inputs.get('kappa')
    Lsd_0_array = 10**np.array(Inputs.get('Lsd_0'))
    E_rot_array = 10**np.array(Inputs.get('E_rot'))
    xi_sd_x_array = Inputs.get('xi_sd_x')
    
    file_number = 0
    for i in range(len(M_ej_sun_array)):
        for j in range(len(beta_min_array)):
            for k in range(len(beta_max_array)):
                for l in range(len(alpha_array)):
                    for m in range(len(kappa_array)):
                        for n in range(len(Lsd_0_array)):
                            for o in range(len(E_rot_array)):
                                for p in range(len(xi_sd_x_array)):
                                    str_M_ej_sun = "{:.2e}".format(M_ej_sun_array[i])
                                    str_beta_min = "{:.2e}".format(beta_min_array[j])
                                    str_beta_max = "{:.2e}".format(beta_max_array[k])
                                    str_alpha = "{:.2e}".format(alpha_array[l])
                                    str_kappa = "{:.2e}".format(kappa_array[m])
                                    str_Lsd_0 = "{:.2e}".format(Lsd_0_array[n])
                                    str_E_rot = "{:.2e}".format(E_rot_array[o])
                                    str_xi_sd_x = "{:.2e}".format(xi_sd_x_array[p])
                                    
                                    filename = 'input_process'+str(file_number)+'.txt'
                                    f1 = open(path+'/'+filename,'w')
                                    f1.write('M_ej_sun='+str_M_ej_sun+'\n')
                                    f1.write('beta_min='+str_beta_min+'\n')
                                    f1.write('beta_max='+str_beta_max+'\n')
                                    f1.write('alpha='+str_alpha+'\n')
                                    f1.write('kappa='+str_kappa+'\n')
                                    f1.write('Lsd_0='+str_Lsd_0+'\n')
                                    f1.write('E_rot='+str_E_rot+'\n')
                                    f1.write('xi_sd_x='+str_xi_sd_x)
                                    f1.close()
                                    file_number = file_number + 1
                                    
    
    return True
    
    
    

def seek_imagename(suffix,Path):
    image_list = []
    allfile_name = os.listdir(Path)
    
    for i in allfile_name:
        if os.path.splitext(i)[1] == suffix:
            image_list.append(i)
    image_list.sort(key=lambda x:int(x[:-4]))
    #print(image_list)
    return image_list

def split(ask):
    output=[]
    output1=''
    
    for i in range(len(ask)):
        digit=ask[i]
        if digit ==' ' or digit=='\t' or digit=='=' or digit==':' : 
            if len(output1)>0:
                output.append(output1)
            output1=''
            continue
        if digit == '\n':
            output.append(output1)
            output1=''
            break
        else: 
            if i == len(ask)-1:
                output1+=digit
                output.append(output1)            
        output1+=digit
    return output


def job(Input_filename):        
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, help='Input file name')
    # args = parser.parse_args()
    # print(args)
    # input_file_name = args.input
    f = open(Input_filename,'r')
    Inputs_raw = f.readlines()
    Inputs = {}
    for i in range(len(Inputs_raw)):
         Input = split(Inputs_raw[i])
         Inputs[Input[0]] = float(Input[1])
    # print(Inputs)




    M_ej_sun = Inputs.get('M_ej_sun')
    beta_min = Inputs.get('beta_min')
    beta_max = Inputs.get('beta_max')
    alpha = Inputs.get('alpha')
    kappa = Inputs.get('kappa')
    Lsd_0 = Inputs.get('Lsd_0')
    E_rot = Inputs.get('E_rot')
    xi_sd_x = Inputs.get('xi_sd_x')
    
    t_inj = E_rot/Lsd_0
    str_M_ej_sun = "{:.2e}".format(M_ej_sun)
    str_Lsd_0 = "{:.2e}".format(Lsd_0)
    str_kappa = "{:.2e}".format(kappa)
    str_t_inj = "{:.2e}".format(t_inj)
    str_E_rot = "{:.2e}".format(E_rot)
    str_xi_sd_x = "{:.2e}".format(xi_sd_x)
    
    tx = 1e0
    str_tx = str(tx)



# if M_ej_sun == None:
#     M_ej_sun = 0.01
# if beta_min == None:
#     beta_min = 0.05
# if beta_max == None:
#     beta_max = 0.20
# if alpha == None:
#     alpha = 1.01    
# if kappa == None:
#     kappa = 1.0    
# if Lsd_0 == None:
#     Lsd_0 = 1.0e45
# if E_rot == None:
#     E_rot = 1.0e52
    
    

    #M_ej_sun = 0.01
    #beta_min = 0.05
    #beta_max = 0.20
    #alpha = 1.01    
    #kappa = 1.0    
    #Lsd_0 = 1.0e47
    #E_rot = 1.0e52
    
    Output_file_name = 'Lsd_'+str_Lsd_0+'_E_rot_'+str_E_rot+'_kappa_'+str_kappa+'_M_ej_'+str_M_ej_sun+'_Lx'+str_xi_sd_x+'_ht.txt' 
    #Output_file_name = 'Lsd_'+str_Lsd_0+'_E_rot_'+str_E_rot+'_kappa_'+str_kappa+'_M_ej_'+str_M_ej_sun+'_Lx'+str_xi_sd_x+'.txt'    
    M_ej = M_ej_sun*Msun
    t0 = 3.3
    N_dbeta = 100
    N_dbeta_max = 300
    t0_max = 10
    while N_dbeta <= N_dbeta_max:
        results = mechanical_model_loop(Lsd_0, E_rot, M_ej, kappa, beta_min, beta_max, alpha, N_dbeta, t0, N_dbeta_max, Output_file_name, tx, xi_sd_x)
        #print(results)
        if len(results) == 2:
            if results[0] == False:
                if t0 <= t0_max:
                    t0 = t0*3
                    print('t0 =', t0, 'N_dbeta', N_dbeta)
                    continue
                else:
                    N_dbeta = int(N_dbeta * 3)
                    print('t0 =', t0, 'N_dbeta', N_dbeta)
                    if N_dbeta < N_dbeta_max:
                        continue
                    else:
                       N_dbeta_max = N_dbeta_max * 3
                       N_dbeta = N_dbeta_max
            elif results[1] == False:
                N_dbeta = int(N_dbeta * 3)
                print('t0 =', t0, 'N_dbeta', N_dbeta)
                continue
        elif len(results) == 1:
            N_dbeta_max = N_dbeta_max * 2
            N_dbeta = N_dbeta_max
            continue
        else:
            break
    if (N_dbeta > N_dbeta_max) and (len(results) <= 2):
        results = mechanical_model_loop(Lsd_0, E_rot, M_ej, kappa, beta_min, beta_max, alpha, N_dbeta, t0, N_dbeta_max, Output_file_name, tx, xi_sd_x)
    #print(results)
        
    


if __name__ == '__main__':  
    multiinputs = False
    if multiinputs == False:
        path = r"Inputs/Input"
        f = os.walk(path)
        for dirpath,dirnames,filenames in f:
            pass
        filename = path + '/' + filenames[0]
        #print(filename)
        res = job(filename)
    if multiinputs == True:
        path = r"Inputs/Input_mp"
        #os.remove(path+'/input_process*.txt')
        Return = generate_input_mp(path)
        
        f = os.walk(path)
        for dirpath,dirnames,filenames in f:
            pass
        filename_list = []
        for i in range(len(filenames)):
            if filenames[i][-4:] == '.txt' and filenames[i][-5].isdigit():
                filename_list.append(path + '/' + filenames[i])
        print(filename_list)
        pool = mp.Pool(processes = min([len(filename_list),4]))
        res = pool.map(job,filename_list)
        
tend = time.time()

print('duration = ', tend - tstart)

    
    #GIF_Figure_path = 'Results_Figure_sh'
    #image_list = seek_imagename('.png', GIF_Figure_path)
    #gif_name = 'T_Luminosity_profile_sh_x_hc.gif'
    #duration = 0.1
    #create_gif(GIF_Figure_path,image_list,gif_name,duration)

# t_obs_list = results[0]
# L_kno_list = results[1]
# nu_L_nu = results[2]
# nu_Lnu_U_list = nu_L_nu[0]
# nu_Lnu_V_list = nu_L_nu[1]
# nu_Lnu_R_list = nu_L_nu[2]
# nu_Lnu_H_list = nu_L_nu[3]
# Results_struc = np.array([t_obs_list,L_kno_list,nu_Lnu_U_list,nu_Lnu_V_list,nu_Lnu_R_list,nu_Lnu_H_list]).T
# np.savetxt('Results.txt',Results_struc)


# plt.loglog(t_obs_list, L_kno_list,'k', label = r'$L_{e}$')
# plt.loglog(t_obs_list, nu_Lnu_H_list,'r', label = 'H band')
# plt.xlabel(r'$t_{\rm obs}~[{\rm s}]$',fontsize = 14)
# plt.ylabel(r'$\nu L_{\nu}~(L_b)~[{\rm erg~s^{-1}}]$',fontsize = 14)
# plt.axis([1e2,1e6,1e38,1e46])
# plt.legend()
# plt.tick_params(labelsize = 12)
# plt.tight_layout()
# plt.savefig('lightcurve.pdf')
#plt.show()
