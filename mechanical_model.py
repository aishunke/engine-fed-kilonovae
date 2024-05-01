
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Dec  8 11:53:33 2021

@author: shunkeai
"""
from mechanical_model_initial_condition import *
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import imageio
import copy
from scipy import interpolate
from band import *


#constant
ccont=2.99*10**10
m_p=1.67352e-24
Msun = 2.0e33
a = 7.5657e-15
h = 6.63e-27
kb=1.38*10**(-23)*10**7
sigmaT=6.652*10**(-25)
sigmaB=5.670373*10**(-5)
Ln_Lambda = 10
pi=np.pi
day = 86400

####################################################################################
#####################################subprogram#####################################
#################################################################################### 
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def input_log(min_value,max_value,interval):
    index_array=np.arange(min_value,max_value,interval)
    log_array=10**index_array;
    return log_array

def gamma2beta(gamma):
    beta=np.sqrt(1-1/gamma**2)
    return beta

def beta2gamma(beta):
    gamma=np.sqrt(1/(1-beta**2))
    return gamma

##################### radioactive heating efficiency #############
def fun_fth(x,t): 
    a = 1.43
    b = 0.17
    d = 1.46
    
    x[x<0] = 0.0
    
    x1 = x[x<1.0]   
    X1 = t / (1 - x1**2)
    fth1 = 0.36 * (np.exp(-a*X1) + np.log(1+2*b*X1**d) / (2*b*X1**d))
    
    x2 = x[x>=1.0]
    fth2 = np.zeros(len(x2))
    
    fth = np.array(list(fth1)+list(fth2))
    return fth

######### calculate particle number density in the NS wind #######
def unshocked_wind(L_sd, Gamma_w, r, sigma):
    v_w = np.sqrt(1-1/Gamma_w**2)*ccont
    rho_w = L_sd/(4*pi*r**2*v_w*Gamma_w**2*ccont**2)/(1+sigma)
    n_w = rho_w / m_p 
    return n_w


# solving the jump condition for relativistic magnetized shocks #
def fgeneral(u_2s,gamma21,Gamma,sigma):
    
    x=u_2s**2

    A = Gamma*(2-Gamma)*(gamma21-1)+2

    B1 = -(gamma21+1)*((2-Gamma)*(Gamma*gamma21**2+1)+Gamma*(Gamma-1)*gamma21)*sigma
    B2 = -(gamma21-1)*(Gamma*(2-Gamma)*(gamma21**2-2)+(2*gamma21+3))
    B=B1+B2

    C1 = (gamma21+1)*(Gamma*(1-Gamma/4.)*(gamma21**2-1)+1)*sigma**2
    C2 = (gamma21**2-1)*(2*gamma21-(2-Gamma)*(Gamma*gamma21-1))*sigma
    C3 = (gamma21+1)*(gamma21-1)**2*(Gamma-1)**2
    C=C1+C2+C3

    D = -(gamma21-1)*(gamma21+1)**2*(2-Gamma)**2*(sigma**2/4.)
    return A*x**3+B*x**2+C*x+D


def u_downstream_s(gamma1,gamma2,sigma):
    beta1=gamma2beta(gamma1)
    beta2=gamma2beta(gamma2)
    beta21=np.abs(beta2-beta1)/(1-beta2*beta1)
    gamma21=beta2gamma(beta21)
    gamma=(4*gamma21+1)/3/gamma21
    #gamma = 5/3
    u_2s_0=np.sqrt((gamma21-1)*(gamma-1)**2/(gamma*(2-gamma)*(gamma21-1)+2))
    
    u_2s_low=1e-10
    u_2s_up=10000.0
    log_u_2s_list=np.arange(np.log10(u_2s_low),np.log10(u_2s_up),1e-3)
    u_2s_list=10**log_u_2s_list
    delta=fgeneral(u_2s_list,gamma21,gamma,sigma)
    
    u_2s_roots=[]
    for i in range(len(u_2s_list)-1):
        if delta[i]*delta[i+1]<0:
            u_2s_roots.append(u_2s_list[i])
    if sigma>0:
        u_2s_roots=np.array(u_2s_roots)
        u_2s=u_2s_roots[u_2s_roots>u_2s_0][0]
    else:
        u_2s=u_2s_0
    return u_2s

def downstream(gamma1,gamma2,sigma,n1):
    
    beta1=gamma2beta(gamma1)
    beta2=gamma2beta(gamma2)
    beta21=np.abs(beta2-beta1)/(1-beta2*beta1)
    gamma21=beta2gamma(beta21)
    gamma=(4*gamma21+1)/3/gamma21
    
    u_2s=u_downstream_s(gamma1,gamma2,sigma)
    u_1s=u_2s*gamma21+(u_2s**2+1)**(1/2)*(gamma21**2-1)**(1/2)
    n2=n1*(u_1s/u_2s)
    e2=(n2*m_p*ccont**2)*(gamma21-1)*(1-(gamma21+1)/(2*u_1s*u_2s)*sigma)
    p2=(gamma-1)*e2
    pb2=p2*(1/2/(gamma-1))*(u_1s/u_2s)*sigma*(e2/n2/m_p/ccont**2)**(-1)
    
    return p2,pb2,e2,n2,u_1s

################ evlotion term in the mechanical model ############
def evolving_term_B_new(Sigma_sph, Gamma, P_sph, H_sph, B_sph, beta, r_m, gamma_hat, Boundary, dP):   
    rho_f = Boundary[0]
    rho_r = Boundary[1]
    p_f = Boundary[2]
    p_r = Boundary[3]
    h_f = Boundary[4]
    h_r = Boundary[5]
    B_f = Boundary[6]
    B_r = Boundary[7]
    beta_f = Boundary[8]
    beta_r = Boundary[9]
    r_f = Boundary[10]
    r_r = Boundary[11]
    
    dP_sph_dr_exp = dP[0]
    dP_sph_dr_fs = dP[1]
    dP_sph_dr_rs = dP[2]
    dP_sph_dr_e = dP[3]
    dP_sph_dr_ra  = dP[4]
    dP_sph_dr_outflow = dP[5]

    
    
    A2_1 = 4*pi*Gamma*(rho_r*r_r**2*(beta-beta_r) + rho_f*r_f**2*(beta_f-beta))
       
    A2_2 = 4*pi*Gamma**2*beta*(h_r*r_r**2*(beta-beta_r) + h_f*r_f**2*(beta_f - beta)) \
           - Gamma**2*beta*(B_r**2*r_r**2*beta_r - B_f**2*r_f**2*beta_f) \
           - (1/2)*(1+beta**2)*Gamma**2*(B_f**2*r_f**2 - B_r**2*r_r**2) - 4*pi*(p_f*r_f**2 - p_r*r_r**2) + 2*P_sph/r_m
    
    A2_3 = 4*pi*Gamma**2*(h_r*r_r**2*(beta-beta_r) + h_f*r_f**2*(beta_f-beta)) \
           + 4*pi*(p_r*r_r**2*beta_r - p_f*r_f**2*beta_f) \
           - (1/2)*(1+beta**2)*Gamma**2*(B_r**2*r_r**2*beta_r - B_f**2*r_f**2*beta_f) \
           - Gamma**2*beta*(B_f**2*r_f**2 - B_r**2*r_r**2)

    A2_4 = 0
    A2_5 = dP_sph_dr_exp + dP_sph_dr_fs + dP_sph_dr_rs + dP_sph_dr_ra  + dP_sph_dr_e + dP_sph_dr_outflow
    A2_6 = 0
    A1 = np.array([[Gamma*beta, Sigma_sph*beta, 0, 0, 0, 0],\
                   [0, 2*Gamma*beta**2*(H_sph + B_sph/4/pi),0,Gamma**2*beta**2, beta**2*Gamma**2/4/pi,\
                   Gamma**2*beta*(H_sph + B_sph/4/pi)],\
                   [0,2*Gamma*beta*(H_sph + (1+beta**2)/8/pi*B_sph), -beta, beta*Gamma**2, \
                    (beta**2+1)*beta*Gamma**2/8/pi, beta**2*Gamma**2*B_sph/4/pi], \
                   [ccont**2, 0, gamma_hat/(gamma_hat-1), -1,0,0], \
                   [0, P_sph, Gamma, 0, 0, 0],\
                   [0, 1/(beta*Gamma**3), 0, 0, 0, -1]]) 


    A2 = np.array([A2_1, A2_2, A2_3, A2_4, A2_5, A2_6])
    x = np.linalg.solve(A1, A2)
    dSigma_sph_dr = x[0]
    dGamma_dr = x[1]
    dP_sph_dr = x[2]
    dH_sph_dr = x[3]
    dB_sph_dr = x[4]
    
    return dSigma_sph_dr, dGamma_dr, dP_sph_dr, dH_sph_dr, dB_sph_dr

######### physical quantities in the unshocked ejecta #########
def dynamic_radioactive_ht_log(Ejecta_initial, t, N_dbeta, alpha):   
    Le_p_out=[]
    beta_Lemax_list = []
    t0p=1.3 
    tsigmap=0.11

    Mej = Ejecta_initial[0]*Msun
    beta_max = Ejecta_initial[1]
    beta_min = Ejecta_initial[2]
    v_max = beta_max*ccont
    v_min = beta_min*ccont
    #N_dbeta = 100001
    beta_list = np.logspace(np.log10(beta_min), np.log10(beta_max), N_dbeta)
    v_list = beta_list*ccont
    beta_min_a = 10**(np.log10(beta_min) - (np.log10(beta_max)-np.log10(beta_min))/(N_dbeta-1))
    beta_max_a = 10**(np.log10(beta_max) - (np.log10(beta_max)-np.log10(beta_min))/(N_dbeta-1))
    beta_list_a = np.logspace(np.log10(beta_min_a), np.log10(beta_max_a), N_dbeta)
    dbeta_list = beta_list - beta_list_a
    dv = dbeta_list*ccont
    kappa=Ejecta_initial[3]
    delta_Eint_p_list = np.zeros(len(beta_list))
    delta_V_list = np.zeros(len(beta_list))
    delta_Mej_list = Mej*(-alpha+3)/(v_max**(-alpha+3) - v_min**(-alpha+3))*(beta_list*ccont)**(-alpha+2)*dv
    
    for i in range(len(t)):
        if i == 0:
            dt = t[i]
        else:
            dt = t[i] - t[i-1]
        delta_ej = dv*t[i]
        R_list = beta_list*ccont*t[i]
        delta_V_list = delta_ej*4*np.pi*R_list**2
        
        rho_list = delta_Mej_list/4/np.pi/R_list/R_list/delta_ej
        d_tau_list = kappa*rho_list*delta_ej
        

        
        d_delta_V_list = delta_ej*8*np.pi*R_list*beta_list*ccont*dt + 4*np.pi*R_list**2*dv*dt       
        ask_coe=(1/2)-(1/3.141592654)*np.arctan((t[i]-t0p)/tsigmap)
        delta_Lra_p_list = (4*10**49*(delta_Mej_list/(2*10**33)*10**2)*ask_coe**1.3)
        delta_Lrap_1 = delta_Lra_p_list 
        f_R_process = 1.0
        N_R_process = int(len(R_list)*f_R_process)
        delta_Lra_p_list = delta_Lrap_1
        delta_Lra_p_list[0:N_R_process] = delta_Lrap_1[0:N_R_process]#*sum(delta_Lrap_1)/sum(delta_Lrap_1[0:N_R_process])
        delta_Lra_p_list[N_R_process:len(delta_Lra_p_list)] = 0
        
        tau_list = np.zeros(len(beta_list))
        for j in reversed(range(len(beta_list))):
            tau_list[j] = tau_list[j] + d_tau_list[j]
        R_list_norm = (R_list - R_list[0]) / (R_list[len(R_list)-1] - R_list[0])
        xi_t_ra = 2.0*fun_fth(R_list_norm, t[i]/day)
        
        tau_dif_pro = tau_list
        tau_dif_pro[tau_dif_pro<1.0] = 1.0
        t_d_pro = (R_list[len(R_list)-1] - R_list + R_list[len(R_list)-1] - R_list[len(R_list)-2])/ccont*tau_dif_pro
        tau_ee = tau_ee = 4.6e13/Ln_Lambda/(rho_list/delta_V_list)
        delta_Lra_p_list = delta_Lra_p_list*np.exp(-tau_ee/t_d_pro)
        
        
        p = delta_Eint_p_list/3/delta_V_list
        delta_Eint_p_list = delta_Eint_p_list + xi_t_ra*delta_Lra_p_list*dt
        pdV = p*d_delta_V_list
        delta_Eint_p_list = delta_Eint_p_list - pdV
        E_emit1 = np.zeros(len(beta_list))
        #tau_list = 0.0#d_tau_list[-1]
        for j in reversed(range(len(beta_list))):
            beta = beta_list[j]
            ############### some part of the energy would be released ##############
            E_emit_n = delta_Eint_p_list[j] * np.exp(-tau_list[j])
            E_emit1[j] = E_emit_n/(1-beta)**2

            delta_Eint_p_list[j] = delta_Eint_p_list[j] - E_emit_n  
        L_emit = np.sum(E_emit1)/dt
        Le_p_out.append(L_emit)
        index_max = list(E_emit1/dv).index(max(E_emit1/dv))
        beta_Lemax = beta_list[index_max]
        beta_Lemax_list.append(beta_Lemax)
    return delta_Eint_p_list, Le_p_out, delta_V_list, p, delta_Mej_list, v_list, beta_Lemax_list,dv

def region1_pressure(Mej,kappa,beta_min,beta_max,alpha,N_dbeta,tc):
    dlogt = 1e-3
    t=input_log(-2,np.log10(tc)+dlogt,dlogt)
    #R0 = 1e7
    Ejecta_initial=[Mej/Msun,beta_max,beta_min,kappa]
    results=dynamic_radioactive_ht_log(Ejecta_initial, t, N_dbeta,alpha)
    delta_Eint_p_list = results[0]
    Lep = results[1]
    delta_V_list = results[2]
    p_list = results[3]
    delta_Mej_list =  results[4]
    v_list = results[5]
    dv_list = results[7]
    
    return delta_Eint_p_list, Lep, delta_V_list, p_list, delta_Mej_list, v_list, dv_list


############################### main function #############################
###########################################################################

def mechanical_model_loop(Lsd_0, E_rot, M_ej, kappa, beta_min, beta_max, alpha, N_dbeta, t0, N_dbeta_max, Output_file_name,tx,xi_sd_x):
    bands = ['U','B','V','R','I','J','H','K','g','r','i','z']
    mkdir('Results_sh_x')
    f = open('Results_sh_x/'+Output_file_name,'w')
    f.write('#t'+'\t'+'Le'+'\t')
    for index_band in range(len(bands)):
        f.write(bands[index_band])
        if index_band < len(bands) - 1:
            f.write('\t')
        else:
            f.write('\n')
    #f.write('#t_obs L_kno nu_Lnu_U nu_Lnu_V nu_Lnu_R nu_Lnu_H L_sh_o T_eff\n')
    f.write('#----------------------------------------------- \n')
    ################################ wind property ###########################
    
    gamma_w0 = 1000
    sigma_w0 = 10000
    t_sd = E_rot/Lsd_0     # spin-down timescale
    xi_sd_equ = 0.1        # beaming factor for NS wind (not isotropic)
    
    ############################## ejecta property ###########################
    sigma_1 = 0
    gamma_hat = 4/3

    
    ##############################################################################
    ####################### generate initial condition ###########################
    ##############################################################################
    
    tp0 = t0
    r0 = t0*beta_min*ccont 
    ts = 0.1
    
    Para_w = [Lsd_0 * xi_sd_equ, gamma_w0, sigma_w0]
    Para_m = [M_ej, sigma_1, beta_max, beta_min, alpha]
    Results = BW_integral_params(Para_w, Para_m, r0, ts)
    Sigma_sph = Results[0]
    P_sph = Results[1]
    H_sph = Results[2]
    B_sph = Results[3]
    Gamma = Results[4]
    #print('Gamma = ', Gamma)
    
    beta = np.sqrt(1 - 1/Gamma**2)
    if beta < beta_min:
        beta = beta_min
    r_r = r0
    r_f = r0

    
    
    beta_array = np.logspace(np.log10(beta_min), np.log10(beta_max), N_dbeta)
    beta_min_a = 10**(np.log10(beta_min) - (np.log10(beta_max)-np.log10(beta_min))/(N_dbeta-1))
    beta_max_a = 10**(np.log10(beta_max) - (np.log10(beta_max)-np.log10(beta_min))/(N_dbeta-1))
    beta_list_a = np.logspace(np.log10(beta_min_a), np.log10(beta_max_a), N_dbeta)
    dbeta = beta_array - beta_list_a
    beta_pro = beta_array
    beta_1 = beta_min
    gamma_1 = beta2gamma(beta_min)
    t0_p = 1.3
    t_sigma_p = 0.11
    ################ evolution of ejecta before t < tc ######################
    tc = t0
    v_min = beta_min*ccont
    v_max = beta_max*ccont
    delta_v = dbeta*ccont
    results = region1_pressure(M_ej,kappa,beta_min,beta_max,alpha,N_dbeta,tc) # Assume the outputs are in the rest frame of ejecta
    delta_Eint_p_pro = results[0]#np.zeros(len(results[0]))#results[0]
    delta_Eint_l_ghost = np.zeros(len(delta_Eint_p_pro))
    delta_E_emit_p = np.zeros(len(delta_Eint_p_pro))
    delta_V_pro = results[2]
    p_pro = results[3]
    delta_Mej = results[4]
    v_pro = results[5]
    dv_pro = results[6]
    
    
    #print('------------- Mechanical model starts ----------------')
    
    
    
    
    E_bw0 = Gamma**2*H_sph - P_sph  + Gamma**2*B_sph/4/pi - B_sph/8/pi 
    dP_sph = 0
    
    E_ISM = gamma_1**2*Sigma_sph*ccont**2
    E_inj = E_bw0 - E_ISM
    t = t0
    tp = tp0
    j = 0
    kkk = 0
    t_sc = 0

    
    Bool_sc = False # if FS crossed the ejecta
    Bool_gamma = False # if RS vanished
    Bool_FS = False # if FS canished
    Bool_t = True # if an appropriate initial time
    Bool_N = True # if an appropriate space resolution 
    Bool_r = True # if an appropriate time resolution
    
    Delta_rd = []
    delta_rd = []
    R_pro = np.zeros(len(v_pro))
    d_delta_V_pro_dt = np.zeros(len(v_pro))
    d_delta_V_pro_dt_pp = np.zeros(len(v_pro))
    
    #########################  Evolving domain #######################
    log_r_min = np.log10(r0)
    log_r = log_r_min
    r = 10**log_r
    r_mag = 1.0e7
    
    ############################## observational band ################
    nu_V = 6.16e14
    nu_U = 8.28e14
    nu_R = 4.27e14
    nu_H = 1.87e14
    nu_10eV = 2.4e15
    nu_100eV = 2.4e16
    nu_1keV = 2.4e15
    nu_01eV = 2.4e13
    lambda_c_list = np.zeros(len(bands))
    for i in range(len(bands)):
        lambda_c_list[i] = Lambda_c[bands[i]]
    nu_list = ccont / (lambda_c_list * 1.0e-4) 
    
    ############################## initializing #######################
    
    dGamma_dr = 0
    
    Gamma_list = []
    dGamma_list = []
    gamma_w_list = []
    Sigma_sph_list = []
    P_sph_list = [] 
    H_sph_list = []
    B_sph_list = []
    Br_list = []
    pr_list = []
    pf_list = []
    prb_list = []
    rho_r_list = []
    rho_f_list = []
    r_list = []
    r_r_list = []
    r_f_list = []
    r_m_list = []
    beta_r_list = []
    beta_f_list = []
    E_bw_list = []
    E_bw_k_list = []
    E_bw_g_list = []
    E_bw_b_list = []
    E_ISM_list = []
    E_inj_list = []
    E_bw_net_list = []
    t_list = []
    N_list = []
    dE_inj_list = []
    #dE_ISM_list = []
    dE_bw_g_list = []
    dE_bw_k_list = []
    dE_bw_b_list = []
    dE_bw_list = []
    dE_bw_net_list = []
    dt_list = []
    xi_t_list = []
    xi_list = []
    beta_list = []
    t_list = []
    tp_list = []
    dt_list =[]
    r_sh_list = []
    xi_Bin_list = []
    E_ej_g_p_list = []
    Vp_list = []
    L_kn_list = []
    L_kno_list = []
    L_sho_list = []
    nu_Lnu_U_list = []
    nu_Lnu_V_list = []
    nu_Lnu_R_list = []
    nu_Lnu_H_list = []
    nu_Lnu_10eV_list = []
    D_eff_list = []
    Eint_tot_list = []
    
    #########################################################################
    ############################### main loop ###############################
    #########################################################################
    t_obs_shell = np.zeros(len(v_pro))
    kk = 0
    kk_list = []
    while t < 5e6:
        #print('--------------------------------------------------')        
        if j <= 0:
            dt_former = 0
        else:
            dt_former = dt
        delta_Eint_p_pro = np.array(list(delta_Eint_p_pro))
        p_pro = delta_Eint_p_pro/3/delta_V_pro
        rho_pro = delta_Mej/delta_V_pro
        ne_pro = rho_pro / m_p
        tau_ee = 4.6e13/Ln_Lambda/ne_pro
        T_pro = (delta_Eint_p_pro / a / delta_V_pro)**(1/4)
        beta = gamma2beta(Gamma)
        Lsd = Lsd_0 * (1 + (t - t0)/t_sd)**(-2) * xi_sd_equ
        
        if not E_inj > E_rot:
            r_sd = r
        if E_inj > E_rot:
            #print('----------central engine spins down-----------')
            Lsd = 1.0e-20
    
        if r_r > r_mag:
            gamma_w = gamma_w0
            sigma_w = sigma_w0
        else: 
            #print('————————————————————Reverse shock vanishes——————————')
            gamma_w = Gamma
            r_r = r_mag
            Bool_gamma = True
    
    
        #### FS shock crossing judgement; if true,  updata region parameters ####
        beta_1 = r_f/t/ccont
        gamma_1 = beta2gamma(beta_1)
        rho_ej = M_ej/4/pi/(v_max - v_min)/r_f**2/t/gamma_1
        rho_1 = rho_ej
        n_1 = rho_1/m_p
        
        if r_f > v_max*t or j >= len(v_pro) or Bool_sc:    
            #print('-----------Forward shock crossing ejecta------------', r)
            # break
            if not Bool_sc:
                r_sc = r 
                t_sc = t
                tp_sc = tp
            Bool_sc = True
            n_1 = 0.1
            rho_1 = n_1*m_p
            sigma_1 = 0.0
            gamma_1 = 1.0
            beta_1 = 0.0
            
    
    
    
    
    
    
    ######################## start to revise the code here ###############
    
    ######################### wind properties ##########################       
            
        n_w = unshocked_wind(Lsd, gamma_w, r_r, sigma_w)
        #print('n_w =', n_w)
        if np.isnan(n_w):
            print('Quantities are nan')
            break
        beta_w = gamma2beta(gamma_w)
        B_w2 = 4*np.pi*n_w*m_p*ccont**2 * sigma_w 
        B_w = np.sqrt(B_w2)
        rho_w = n_w*m_p
    
        beta_bww = (beta_w - beta)/(1-beta_w*beta)
        Gamma_bww = beta2gamma(beta_bww)
        u_bww = Gamma_bww*beta_bww
        u_w = beta_w*gamma_w
        u_bw = beta*Gamma
        
    ######################## jump condition forward shock ######################        
        
        u_bwfs = u_downstream_s(gamma_1,Gamma,sigma_1)
        [p_f_sh, p_fb, e_f, n_f, u_fs1] = downstream(gamma_1,Gamma,sigma_1,n_1)
        rho_f = n_f*m_p
        Gamma_eff_ej = gamma_1*gamma_hat - (gamma_hat-1)/gamma_1
        Gamma_eff_bw = Gamma*gamma_hat - (gamma_hat-1)/Gamma
        if j < len(R_pro):
            p_ra = p_pro[j]
        else: 
            p_ra = 0.0
        if (p_ra > p_f_sh) or (j<100):
            p_ra = 0.0
        if Bool_sc:
            p_ra = 0.0
        p_f_ra = p_ra
        p_f = p_f_sh + p_f_ra
        
        h_f = rho_f*ccont**2 + e_f + p_f
        B_f = 0   
        beta_bwfs = u_bwfs / np.sqrt(1+u_bwfs**2)        
        beta_f = (beta + beta_bwfs)/(1 + beta*beta_bwfs)    
        gamma_f = np.sqrt(1/(1-beta_f**2))  
        
        
        r_m = r
        
    ############## The step length can be set based on FS and ejecta ###########
        #print('-------------- beta_1 = ', beta_1,',beta =', beta)
        #if(( beta_1 > beta) or (beta_f < beta_1)) and (j > 100):
        if (beta_1 > beta)  and (j > 100):
            #print('-----------------FS error--------------')
            #break
            Bool_FS = True
        if Bool_FS:
            beta_f = beta_1
            beta = beta_1
            gamma_f = gamma_1
            #p_f = p_ra
            p_f = 0
            rho_f = rho_1
            e_f = p_f/(gamma_hat - 1)
            h_f = rho_f * ccont**2 + e_f + p_f
       # else:
            #print('beta_1 =', beta_1, 'beta =', beta)    
       # print('---------------------Bool_sc = ', Bool_sc, 'Bool_FS =', Bool_FS)
        if (not Bool_sc) and (not Bool_FS): #and (not E_inj > E_rot):
            dt = delta_v[j]*t/(beta_f - beta_1)/ccont
            #print('------------ dt =', dt)
            t_1 = t + dt
            if dt < 0:
                Bool_t = False
                return Bool_t, Bool_N
            log_t_1 = np.log10(t_1)
            log_t = np.log10(t)
            dlog_t = log_t_1 - log_t
            if not E_inj > E_rot:
                dlog_t0 = dlog_t
            else:
                dlog_t = dlog_t0
                log_t = np.log10(t)        
                logt_new = log_t + dlog_t
                t_new = 10**logt_new
                dt = t_new - t
            
            if beta_f < beta_pro[j]:
                Bool_FS = True
                dlog_t = dlog_t0
                log_t = np.log10(t)        
                logt_new = log_t + dlog_t
                t_new = 10**logt_new
                dt = t_new - t
            if (dlog_t > 1e-1) and (j < 100) and len(R_pro) < N_dbeta_max:
                #print('-------- dlog_t = --------', dlog_t)
                Bool_N = False
                return Bool_t, Bool_N
            
        else:
           # dt1 = delta_v[j]*t/(beta_f - beta_1)/ccont
            dlog_tp = dlog_t
            log_t = np.log10(t)        
            logt_new = log_t + dlog_tp
            t_new = 10**logt_new
            dt = t_new - t
           # if dt > dt1:
           #     dt = dt1
        dr_f = beta_f*ccont*dt
        dr = dr_f / beta_f * beta
        
        dtp = dt/Gamma 

        
    #############################################################################
    
        if (not Bool_sc) and (not Bool_FS):
            #print('ask=', len(Delta_rd))
            Delta_rd = list(Delta_rd)
            delta_rd = list(delta_rd)
            Delta_rd.append(r_f + dr_f - r - dr)
            Delta_rd = np.array(Delta_rd)
        j = len(Delta_rd) - 1
        #j = min([len(Delta_rd) - 1, len(R_pro)-1])
        #print('len(Delta_rd) =', len(Delta_rd))
        #print('----- j =', j, 'Bool_sc =', Bool_sc, '-----')
        #print('---- E_rot > E_inj ?', E_rot > E_inj, '------' )
        #print('Lsd = ', Lsd)
        
        if not Bool_sc:
            R_pro[j] = r_f + dr_f
            if j > 0:
                R_pro[0:j] = R_pro[0:j] + dr
            R_pro[j+1:] = (v_pro[j+1:]+dv_pro[j+1:])*(t+dt)
        else:
            if j > 0:
                R_pro[0:] = R_pro[0:] + dr
        R_pro_norm = (R_pro - R_pro[0]) / (R_pro[len(R_pro)-1] - R_pro[0])
        R_pro_norm1 = R_pro/ R_pro[len(R_pro)-1]
            
        R_shift = np.array([r+dr] + list(R_pro[0:len(R_pro)-1]))
        delta_rd = R_pro[0:j+1] - R_shift[0:j+1]
        delta_tau_pro = delta_Mej/4/pi/R_pro/R_pro*kappa
        delta_tau_x_pro = delta_tau_pro
        tau_pro = np.zeros(len(delta_tau_pro))
        part_p_part_r = np.zeros(len(delta_tau_pro)+1)
        L_trans = np.zeros(len(delta_tau_pro)+1)
        d_delta_Eint_p_pro_ht = np.zeros(len(delta_tau_pro)+1)
        delta_Eint_p_pro_s = delta_Eint_p_pro
        
        
        
        for k in reversed(range(len(delta_tau_pro))):
            if k == len(delta_tau_pro)-1:
                tau_pro[k] = delta_tau_pro[k]
                part_p_part_r[k+1] = (0 - p_pro[k])/(R_pro[k] - R_pro[k-1])
                part_p_part_r[k] = (p_pro[k] - p_pro[k-1])/(R_pro[k] - R_pro[k-1])

            elif k > 0:
                tau_pro[k] = tau_pro[k+1] + delta_tau_pro[k]
                part_p_part_r[k] = (p_pro[k] - p_pro[k-1])/(R_pro[k] - R_pro[k-1])
            else:
                tau_pro[k] = tau_pro[k+1] + delta_tau_pro[k]
                part_p_part_r[k] = 0
        tau_x_pro = tau_pro
                
                
        if j <= 0:      
            k_N_back = np.zeros(len(delta_tau_pro) + 1, dtype=int) 
            k_N_forward = np.zeros(len(delta_tau_pro) + 1, dtype=int) 
            k_N_forward_x = 0
            k_N_forward_gh = np.zeros(len(delta_tau_pro) + 1, dtype=int) 
        else:
            k_N_back = k_N_back_s
            k_N_forward = k_N_forward_s
            k_N_back[0] = 0
            k_N_forward[-1] = 0
            k_N_forward_x = k_N_forward_x_s
            
        
        Delta_R = delta_V_pro / 4 / pi / R_pro**2
        for k in reversed(range(len(delta_tau_pro))):
            if k == len(delta_tau_pro)-1:
                  t_dif_back = 0
                  tau_dif_back = sum(delta_tau_pro[k-k_N_back[k+1]+1:k+1])
                  rho_deltaR = sum(rho_pro[k-k_N_back[k+1]+1:k+1]*Delta_R[k-k_N_back[k+1]+1:k+1])
                  #print('--------------------------------',k_N_back[k+1])
                  while t_dif_back < dtp and k - k_N_back[k+1] >= 0:
                      #print('--------------------------------',k_N_back[k+1])
                      tau_dif_back = tau_dif_back + delta_tau_pro[k-k_N_back[k+1]]
                      rho_deltaR = rho_deltaR + rho_pro[k-k_N_back[k+1]] * Delta_R[k-k_N_back[k+1]]
                      N_scatter = tau_dif_back + tau_dif_back**2
                      rho_mean = rho_deltaR / (R_pro[k] - R_pro[k-1-k_N_back[k+1]])
                      l_mean = 1.0 / kappa / rho_mean
                      t_dif_back = l_mean/ccont * N_scatter
                      k_N_back[k+1] = k_N_back[k+1] + 1
                      
                                     
            if k > 0:
                  t_dif_back = 0
                  tau_dif_back = sum(delta_tau_pro[k-k_N_back[k]:k])
                  rho_deltaR = sum(rho_pro[k-k_N_back[k]:k]*Delta_R[k-k_N_back[k]:k])
                  while t_dif_back < dtp and k - k_N_back[k] - 1 >= 0:
                      tau_dif_back = tau_dif_back + delta_tau_pro[k-k_N_back[k]-1]
                      rho_deltaR = rho_deltaR + rho_pro[k-k_N_back[k]-1] * Delta_R[k-k_N_back[k]-1]
                      N_scatter = tau_dif_back+tau_dif_back**2
                      rho_mean = rho_deltaR / (R_pro[k] - R_pro[k-1-k_N_back[k]])
                      l_mean = 1.0 / kappa / rho_mean
                      t_dif_back = l_mean/ccont * N_scatter
                      k_N_back[k] = k_N_back[k] + 1
                      
                     
                  t_dif_forward = 0
                  tau_dif_forward = sum(delta_tau_pro[k:k+k_N_forward[k]])
                  rho_deltaR = sum(rho_pro[k:k+k_N_forward[k]]*Delta_R[k:k+k_N_forward[k]])
                  while t_dif_forward < dtp and k + k_N_forward[k] <= len(delta_tau_pro)-1:
                      tau_dif_forward = tau_dif_forward + delta_tau_pro[k+k_N_forward[k]]
                      rho_deltaR = rho_deltaR + rho_pro[k+k_N_forward[k]] * Delta_R[k+k_N_forward[k]]
                      N_scatter = tau_dif_forward+tau_dif_forward**2
                      rho_mean = rho_deltaR / (R_pro[k+k_N_forward[k]] - R_pro[k-1])
                      l_mean = 1.0 / kappa / rho_mean
                      t_dif_forward = l_mean/ccont * N_scatter
                      #t_dif_forward = (R_pro[k+k_N_forward[k]] - R_pro[k-1])*max([tau_dif_forward,1])/ccont#l_mean / ccont * N_scatter
                      k_N_forward[k] = k_N_forward[k] + 1
                      
                      
                  t_dif_forward_gh = 0
                  while t_dif_forward_gh < dtp and k + k_N_forward_gh[k] <= len(delta_tau_pro)-1:
                      t_dif_forward_gh = (R_pro[k+k_N_forward_gh[k]] - R_pro[k-1]) / ccont
                      k_N_forward_gh[k] = k_N_forward_gh[k] + 1
            else:
                  t_dif_forward = 0
                  tau_dif_forward = sum(delta_tau_pro[k:k+k_N_forward_x])
                  rho_deltaR = sum(rho_pro[k:k+k_N_forward_x]*Delta_R[k:k+k_N_forward_x])
                  while t_dif_forward < dtp and k + k_N_forward_x < len(delta_tau_pro)-1:
                      tau_dif_forward = tau_dif_forward + delta_tau_x_pro[k+k_N_forward_x]
                      rho_deltaR = rho_deltaR + rho_pro[k+k_N_forward_x] * Delta_R[k+k_N_forward_x]
                      N_scatter = tau_dif_forward+tau_dif_forward**2
                      rho_mean = rho_deltaR / (R_pro[k+k_N_forward_x+1] - R_pro[k])
                      l_mean = 1.0 / kappa / rho_mean
                      t_dif_forward = l_mean/ccont * N_scatter
                      #t_dif_forward = (R_pro[k+k_N_forward[k]+1] - R_pro[k])*max([tau_dif_forward,1])/ccont#l_mean / ccont * N_scatter
                      k_N_forward_x = k_N_forward_x + 1
                      
                  t_dif_forward_gh = 0
                  while t_dif_forward_gh < dtp and k + k_N_forward_gh[k] <= len(delta_tau_pro)-1:
                     t_dif_forward_gh = (R_pro[k+k_N_forward_gh[k]]+1 - R_pro[k]) / ccont
                     k_N_forward_gh[k] = k_N_forward_gh[k] + 1
                     
                  t_dif_forward = 0
                  tau_dif_forward = sum(delta_tau_pro[k:k+k_N_forward[k]])
                  rho_deltaR = sum(rho_pro[k:k+k_N_forward[k]]*Delta_R[k:k+k_N_forward[k]])
                  while t_dif_forward < dtp and k + k_N_forward[k] < len(delta_tau_pro)-1:
                      tau_dif_forward = tau_dif_forward + delta_tau_pro[k+k_N_forward[k]]
                      rho_deltaR = rho_deltaR + rho_pro[k+k_N_forward[k]] * Delta_R[k+k_N_forward[k]]
                      N_scatter = tau_dif_forward+tau_dif_forward**2
                      rho_mean = rho_deltaR / (R_pro[k+k_N_forward[k]+1] - R_pro[k])
                      l_mean = 1.0 / kappa / rho_mean
                      t_dif_forward = l_mean/ccont * N_scatter
                      #t_dif_forward = (R_pro[k+k_N_forward[k]+1] - R_pro[k])*max([tau_dif_forward,1])/ccont#l_mean / ccont * N_scatter
                      k_N_forward[k] = k_N_forward[k] + 1
                      
        k_N_forward_s = k_N_forward - 1
        k_N_back_s = k_N_back - 1
        k_N_forward_x_s = k_N_forward_x - 1

        for k in reversed(range(len(delta_tau_pro))):
            if k == len(delta_tau_pro)-1:

                Sum_delta_Eint_back = sum(delta_Eint_p_pro_s[k+1-k_N_back[k+1]:k+1])
                L_trans[k+1] = -4*pi*R_pro[k]**2*ccont/kappa/rho_pro[k]*part_p_part_r[k+1]
                d_delta_Eint_p_pro_ht[k+1] = min([L_trans[k+1] * dtp, delta_Eint_p_pro_s[k]*0.1])
            if k > 0:
                L_trans[k] = -4*pi*R_pro[k]**2*ccont/kappa/rho_pro[k]*part_p_part_r[k]

                if k_N_back[k] <= 1 and k_N_forward[k] <= 1:
                    if k_N_back[k] == 0 or k_N_forward[k] == 0:
                        d_delta_Eint_p_pro_ht[k] = 0
                    else:
                    
                        Sum_delta_Eint_back = sum(delta_Eint_p_pro_s[k-k_N_back[k]:k])
                        Sum_delta_Eint_LTE = sum(delta_Eint_p_pro_s[k-k_N_back[k]:k+k_N_forward[k]])
                        Sum_delta_V_back = sum(delta_V_pro[k-k_N_back[k]:k])
                        Sum_delta_V_LTE = sum(delta_V_pro[k-k_N_back[k]:k+k_N_forward[k]])
                        Sum_delta_Mej_back = sum(delta_Mej[k-k_N_back[k]:k])
                        Sum_delta_Mej_LTE = sum(delta_Mej[k-k_N_back[k]:k+k_N_forward[k]])
                        p_avg = Sum_delta_Eint_LTE  / Sum_delta_V_LTE
                        Sum_delta_Eint_back_avg = p_avg * Sum_delta_V_back
                        d_delta_Eint_p_pro_ht[k] = Sum_delta_Eint_back - Sum_delta_Eint_back_avg
                        if np.abs(d_delta_Eint_p_pro_ht[k]) > np.abs(L_trans[k]*dtp):
                            d_delta_Eint_p_pro_ht[k] = L_trans[k]*dtp
                        
                        
                else:

                    if k >= k_N_back[k] and k < len(delta_tau_pro) - k_N_forward[k]:
                        k_N_back[k] = k_N_back[k] - 1
                        k_N_forward[k] = k_N_forward[k] - 1
                        Sum_delta_Eint_back = sum(delta_Eint_p_pro_s[k-k_N_back[k]:k])
                        Sum_delta_Eint_LTE = sum(delta_Eint_p_pro_s[k-k_N_back[k]:k+k_N_forward[k]])
                        Sum_delta_V_back = sum(delta_V_pro[k-k_N_back[k]:k])
                        Sum_delta_V_LTE = sum(delta_V_pro[k-k_N_back[k]:k+k_N_forward[k]])
                        Sum_delta_Mej_back = sum(delta_Mej[k-k_N_back[k]:k])
                        Sum_delta_Mej_LTE = sum(delta_Mej[k-k_N_back[k]:k+k_N_forward[k]])
                        p_avg = Sum_delta_Eint_LTE /Sum_delta_V_LTE
                        Sum_delta_Eint_back_avg = p_avg * Sum_delta_V_back
                        d_delta_Eint_p_pro_ht[k] = Sum_delta_Eint_back - Sum_delta_Eint_back_avg
                    elif k < k_N_back[k]:
                        k_N_forward[k] = k_N_forward[k] - 1
                        if k > 1:
                            k_N_back[k] = k_N_back[k] - 1
                        Sum_delta_Eint_back = sum(delta_Eint_p_pro_s[0:k])
                        Sum_delta_Eint_LTE = sum(delta_Eint_p_pro_s[0:k+k_N_forward[k]])
                        Sum_delta_V_back = sum(delta_V_pro[0:k])
                        Sum_delta_V_LTE = sum(delta_V_pro[0:k+k_N_forward[k]])
                        Sum_delta_Mej_back = sum(delta_Mej[0:k])
                        Sum_delta_Mej_LTE = sum(delta_Mej[0:k+k_N_forward[k]])
                        
                        p_avg = Sum_delta_Eint_LTE / Sum_delta_V_LTE
                        Sum_delta_Eint_back_avg = p_avg * Sum_delta_V_back
                        d_delta_Eint_p_pro_ht[k] = Sum_delta_Eint_back - Sum_delta_Eint_back_avg
                    else:
                        k_N_back[k] = k_N_back[k] - 1
                        if k < len(delta_tau_pro) - 1:
                            k_N_forward[k] = k_N_forward[k] - 1
                        Sum_delta_Eint_back = sum(delta_Eint_p_pro_s[k-k_N_back[k]:k])
                        Sum_delta_Eint_LTE = sum(delta_Eint_p_pro_s[k-k_N_back[k]:len(delta_tau_pro)])
                        Sum_delta_V_back = sum(delta_V_pro[k-k_N_back[k]:k])
                        Sum_delta_V_LTE = sum(delta_V_pro[k-k_N_back[k]:len(delta_tau_pro)])
                        Sum_delta_Mej_back = sum(delta_Mej[k-k_N_back[k]:k])
                        Sum_delta_Mej_LTE = sum(delta_Mej[k-k_N_back[k]:len(delta_tau_pro)])
                        p_avg = Sum_delta_Eint_LTE / Sum_delta_V_LTE
                        Sum_delta_Eint_back_avg = p_avg * Sum_delta_V_back
                        d_delta_Eint_p_pro_ht[k] = Sum_delta_Eint_back - Sum_delta_Eint_back_avg
                        
                    #t_d_pro = (R_pro[len(delta_tau_pro)-1] - R_pro[k] + dbeta[k]*ccont*t)/ccont * max([1,tau_pro[k]])#*np.exp(tau_pro-1)
            else:
                d_delta_Eint_p_pro_ht[k] = 0 
            
            #if tau_pro[k] < 1e8:
            #    d_delta_Eint_p_pro_ht[k] = 0
            
            
            
                 
            if delta_Eint_p_pro_s[k] + d_delta_Eint_p_pro_ht[k] - d_delta_Eint_p_pro_ht[k+1] < 0:
                d_delta_Eint_p_pro_ht[k] = d_delta_Eint_p_pro_ht[k+1] - delta_Eint_p_pro_s[k]
            
        for k in range(len(delta_tau_pro)):
            delta_Eint_p_pro[k] = delta_Eint_p_pro_s[k] + d_delta_Eint_p_pro_ht[k] - d_delta_Eint_p_pro_ht[k+1]

            
        if np.isinf(delta_Eint_p_pro).any():
            print('--------inf in delta_Eint_p_pro------')
            break
        if np.isnan(delta_Eint_p_pro).any():
            print('--------nan in delta_Eint_p_pro------')
            break
        
        tau_c = 1/ccont/rho_pro/kappa
        N_scatt = dtp / tau_c + 1
        frac_release = 1 - (1 - np.exp(-tau_pro))**N_scatt
        frac_release[frac_release > 1.0] = 1.0
        
        tau_dif_pro = copy.deepcopy(tau_pro)
        tau_dif_pro[tau_dif_pro<1.0] = 1.0
        t_d_pro = (R_pro[len(R_pro)-1] - R_pro + R_pro[len(R_pro)-1] - R_pro[len(R_pro)-2] )/ccont*tau_dif_pro
        eec_dif_ratio = tau_ee / t_d_pro
        eec_dif_index = len(eec_dif_ratio[eec_dif_ratio < 1.0]) - 1 
        if eec_dif_index < 0: eec_dif_index = 0
        
        delta_Eint_p_pro = delta_Eint_p_pro - delta_Eint_p_pro/3/delta_V_pro*d_delta_V_pro_dt_pp*dtp
        k_N_th = k_N_forward[0]
        tau_th = tau_x_pro[0]
        f_X_obsorb = (1 - np.exp(-tau_th))
        Lx = Lsd * xi_sd_x
        Ex_tot = Lx*dtp*f_X_obsorb
        if Ex_tot > 0:
            if k_N_th > 0:
                Ex_dis = Ex_tot * delta_V_pro[0:k_N_th] / sum(delta_V_pro[0:k_N_th])
                delta_Eint_p_pro[0:k_N_th] = delta_Eint_p_pro[0:k_N_th] + Ex_dis
                #print('Ex_dis =', Ex_dis)
                
                
                
            
            
            

        xi_t_sh = (1.0 - np.exp(-tau_pro[j]))
        xi_t_ra = 2.0*fun_fth(R_pro_norm, t/day)
        #xi_t_ra = 1.0
        ask_coe=(1/2)-(1/3.141592654)*np.arctan((t-t0_p)/t_sigma_p)
        delta_Lrap = (4.0e49*(delta_Mej/(2.0e33)*1.0e2)*ask_coe**1.3)*xi_t_ra
        delta_Lrap_1 = delta_Lrap 
        
        
        f_R_process = 1.0
        N_R_process = int(len(R_pro)*f_R_process)
        
        delta_Lrap = delta_Lrap_1
        delta_Lrap[0:N_R_process] = delta_Lrap_1[0:N_R_process]
        delta_Lrap[N_R_process:len(delta_Lrap)] = 0.0
    
        
        d_delta_V_pro_dt_pp[0:j+1] = Gamma*Gamma* 8*pi*R_pro[0:j+1]*delta_rd*beta*ccont \
            + Gamma * 4*pi*R_pro[0:j+1]* R_pro[0:j+1] * delta_rd * dGamma_dr * beta*ccont
        d_delta_V_pro_dt_pp[j+1:] = 12*pi*Gamma*Gamma*delta_v[j+1:] * v_pro[j+1:] * v_pro[j+1:] * t * t
        
        d_delta_V_pro_dt[0:j+1] = 8 * pi * R_pro[0:j+1] * delta_rd*beta*ccont
        d_delta_V_pro_dt[j+1:] = 12 * pi * delta_v[j+1:] * v_pro[j+1:] * v_pro[j+1:] * t * t
        
        
        delta_Le_pro = np.zeros(len(v_pro))
        
        
        if (not Bool_sc) and (not Bool_FS):
            L_sh =  4*pi*r_f**2*(beta_f-beta)*ccont*p_f_sh*Gamma / (gamma_hat-1)*xi_t_sh
        else:
            L_sh = 0.0
            
        delta_Eint_p_pro = delta_Eint_p_pro + delta_Lrap*dtp
        delta_Eint_p_pro[j] = delta_Eint_p_pro[j] + L_sh*dtp
        Gamma_pro = np.sqrt(1/(1-beta_pro**2))
        D_pro = 1/Gamma_pro/(1-beta_pro)
        
        T_pro = (delta_Eint_p_pro/delta_V_pro/a)**(1/4)
        
        delta_Eint_l_ghost = delta_Eint_p_pro*frac_release
        delta_Eint_p_pro = delta_Eint_p_pro - delta_Eint_l_ghost #delta_Eint_p_pro*frac_release
        delta_E_emit_p = delta_Eint_l_ghost
            
        delta_Le_pro = delta_E_emit_p / dtp
        
        Le = sum(delta_Le_pro) + L_trans[k+1]
        Le_o = sum(delta_E_emit_p) / dt * Gamma_pro[len(delta_E_emit_p)-1] * D_pro[len(delta_E_emit_p)-1]**2 #+ L_trans[k+1]*Gamma_pro[len(delta_E_emit_p)-1] * D_pro[len(delta_E_emit_p)-1]**2 
        
        
        Eint_tot = np.sum(delta_Eint_p_pro)
        D_eff = D_pro[len(R_pro)-1]
        
        
        log_nu_p = np.arange(11,19,0.01)
        log_nu_p_1 = np.array(list(log_nu_p[1:])+[19.00])
        nu_p = 10**log_nu_p
        nu_p_1 = 10**log_nu_p_1
        delta_nu_p = nu_p_1 - nu_p

        nu_Lnu_list = np.zeros(len(bands))
        for k in range(len(T_pro)):
            K_coe = delta_Le_pro[k] * D_pro[k]* D_pro[k] / sum((h*nu_p)**4/(np.exp(h*nu_p/(kb*T_pro[k]))-1) / nu_p * delta_nu_p)
            nu_Lnu_list1 = K_coe * (h*nu_list/D_pro[k])**4/(np.exp(h*nu_list/D_pro[k]/(kb*T_pro[k]))-1)
            if not (True in np.isnan(nu_Lnu_list1)):
                nu_Lnu_list = nu_Lnu_list + nu_Lnu_list1
        L_kn_list.append(Le)
        L_kno_list.append(Le_o)
        Eint_tot_list.append(Eint_tot)
        L_sho_list.append(L_sh/(1-beta)**2)
        D_eff_list.append(D_eff)
            
    ############################# Across the reverse shock ######################
        if not Bool_gamma:
            [p_r, p_rb, e_r, n_r, u_wrs]= downstream(gamma_w,Gamma,sigma_w,n_w)   
            u_bwrs = u_downstream_s(gamma_w,Gamma,sigma_w) 
            B_r = B_w*(u_wrs/u_bwrs)
            rho_r = n_r*m_p
            beta_bwrs = u_bwrs / np.sqrt(1+u_bwrs**2)
            beta_r = (beta - beta_bwrs)/(1-beta*beta_bwrs)  
            
        if Bool_gamma:
            #print('----------------------------------- RS vanish----------------------------')
            B_r = B_w
            rho_r = rho_w
            p_rb = B_r**2/8/pi
            beta_r = 0
            p_r = 0
            e_r = 0
            kkk = kkk + 1
        if E_inj > E_rot:
            B_r = 0
            rho_r = 0
            beta_r = 0
            p_r = 0
            e_r = 0
    
        dP_sph_rs = 4*pi*r_r**2*(beta-beta_r)*ccont*dt*p_r*Gamma
        dP_sph_dr_rs = dP_sph_rs/dr
        if j <= 0:
            dGamma = 0        
        dP_sph_exp = -(gamma_hat-1)*sum(p_pro[0:j+1]*d_delta_V_pro_dt[0:j+1]*dt)- (gamma_hat - 1)*P_sph*dGamma
        dP_sph_dr_exp = dP_sph_exp/dr
        dP_sph_fs = 4*pi*r_f**2*(beta_f-beta)*ccont*dt*p_f*Gamma
        dP_sph_dr_fs = dP_sph_fs/dr
        if j <= 0:
            dP_sph_dr_e = 0
        else:
            dP_sph_dr_e = -(gamma_hat-1)*sum(delta_Le_pro[0:j+1])*dt/dr
            
        if not Bool_sc:
            dMra_p = 4*pi*r_f**2*rho_1*(beta_f-beta_1)*ccont*dt*gamma_1
        else:
            dMra_p = 0
        if j <= 0:
            Mra_p = 0
        else:
            Mra_p = Mra_p + dMra_p
        Lra_p = 4e49*(Mra_p/(1e-2*2e33))*(1/2 - 1/pi*np.arctan((tp - t0_p)/t_sigma_p))**1.3
        dP_sph_dr_ra = (gamma_hat-1)*Lra_p/Gamma/beta/ccont
        dP_sph_dr_outflow = -d_delta_Eint_p_pro_ht[j+1]/3/dr
            
                
        h_r = rho_r*ccont**2 + e_r + p_r             
            

        
        dP = [dP_sph_dr_exp, dP_sph_dr_fs, dP_sph_dr_rs,dP_sph_dr_e, dP_sph_dr_ra, dP_sph_dr_outflow]
        Boundary = [rho_f, rho_r, p_f, p_r, h_f, h_r, B_f, B_r, beta_f, beta_r, r_f, r_r]
    
    
    
    
    ###################### the evolution of blastwave starts here #################3
        [dSigma_sph_dr, dGamma_dr, dP_sph_dr, dH_sph_dr, dB_sph_dr] \
            = evolving_term_B_new(Sigma_sph, Gamma, P_sph, H_sph, B_sph, beta, r_m, gamma_hat, Boundary, dP)
        dSigma_sph = dSigma_sph_dr*dr
        dGamma = dGamma_dr*dr
        dP_sph = dP_sph_dr*dr
        dH_sph = dH_sph_dr*dr
        dB_sph = dB_sph_dr*dr
            
        dr_r_dr = beta_r/beta
            
        Sigma_sph_p = Sigma_sph + dSigma_sph
        Gamma_p = Gamma + dGamma
        P_sph_p = P_sph + dP_sph
        H_sph_p = H_sph + dH_sph
        B_sph_p = B_sph + dB_sph
            
            
        dE_inj = Lsd/(beta*ccont) * (beta_w - beta_r)/beta_w *   dr
        E_inj_p = E_inj + dE_inj
        
    
    ############################### boundry of inner while loop ###############################            
    ######### ##Stop the propagation of the RS at r_r = r_{mag} (magnetosphere) ############### 
    
        r_f = r_f + dr_f 
        r_r = r_r + dr_r_dr*dr
        r = r + dr
    
    ###############################  values after iteration ####################################
                
        Gamma = Gamma_p
        if Gamma < 1:
            Gamma = 1.0 + 1e-6
        Sigma_sph = Sigma_sph_p
        P_sph = P_sph_p
        H_sph = H_sph_p
        B_sph = B_sph_p
    
        t = t + dt
        tp = tp + dtp
        t_list.append(t)
        tp_list.append(tp)
        r_list.append(r)
        
        E_inj = E_inj_p
        #E_ISM = E_ISM_p
        delta_V_pro = delta_V_pro + d_delta_V_pro_dt_pp*dtp
            
        if (not Bool_sc) and (not E_inj > E_rot): 
            j = j + 1
        elif (E_inj > E_rot) and (r_f > R_pro[j]) and (not Bool_sc):
            j = j + 1
            
        
        print('t = ', t, 'beta_f =', beta_f, 'beta_1 =', beta_1)

        
        if Bool_gamma and (r_r > r_mag):
            print('RS out of magnetosphere again')
            break
        
        f.write(str(tp/D_eff)+'\t'+str(Le_o)+'\t')
        for band_index in range(len(bands)):
            f.write(str(nu_Lnu_list[band_index]))
            if band_index < len(bands) - 1:
                f.write('\t')
            else:
                f.write('\n')
    f.close()
    t_obs_list = np.array(t_list)/np.array(D_eff_list)    
    nu_L_nu = [nu_Lnu_U_list, nu_Lnu_V_list, nu_Lnu_R_list, nu_Lnu_H_list]
    if len(t_obs_list[np.array(L_kno_list)>0]) < 20:
        Bool_r = False
        return [Bool_r]
    

    return t_obs_list, L_kno_list, nu_L_nu

