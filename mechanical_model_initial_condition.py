#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 14:44:41 2021

@author: shunkeai
"""

import matplotlib.pyplot as plt
import numpy as np

#constant
ccont=2.99*10**10
mp=1.67352e-24
Msun = 2.0e33
a = 7.5657e-15
h = 6.63e-27
kb=1.38*10**(-23)*10**7
sigmaT=6.652*10**(-25)
sigmaB=5.670373*10**(-5)
pi=np.pi


def unshocked_wind1(L_ej, Gamma_ej, Gamma_ej_p, r, tau, rho_1):
    v_ej = np.sqrt(1-1/Gamma_ej**2)*ccont
    rho_ej = (L_ej/(4*pi*r**2*v_ej*Gamma_ej**2*ccont**2)) * (1-(r/ccont)*(Gamma_ej_p / (Gamma_ej**2-1)**(3/2)))**(-1)
    n_w = rho_ej/mp
    return n_w

def unshocked_wind(L_sd, Gamma_w, r, sigma):
    v_w = np.sqrt(1-1/Gamma_w**2)*ccont
    rho_w = L_sd/(4*pi*r**2*v_w*Gamma_w**2*ccont**2)/(1+sigma)
    n_w = rho_w / mp 
    return n_w

def unshocked_wind_flux(F, Gamma_w, sigma):
    v_w = np.sqrt(1-1/Gamma_w**2)*ccont
    rho_w = (F/(v_w*Gamma_w**2*ccont**2)/(1+sigma)) 
    n_w = rho_w / mp 
    return n_w

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


def gamma2beta(gamma):
    beta=np.sqrt(1-1/gamma**2)
    return beta

def beta2gamma(beta):
    gamma=np.sqrt(1/(1-beta**2))
    return gamma

def u_downstream_s(gamma1,gamma2,sigma):
    beta1=gamma2beta(gamma1)
    beta2=gamma2beta(gamma2)
    beta21=np.abs(beta2-beta1)/(1-beta2*beta1)
    gamma21=beta2gamma(beta21)
    Gamma=(4*gamma21+1)/3/gamma21
    u_2s_0=np.sqrt((gamma21-1)*(Gamma-1)**2/(Gamma*(2-Gamma)*(gamma21-1)+2))
    
    u_2s_low=1e-2
    u_2s_up=100000.0
    log_u_2s_list=np.arange(np.log10(u_2s_low),np.log10(u_2s_up),1e-3)
    u_2s_list=10**log_u_2s_list
    delta=fgeneral(u_2s_list,gamma21,Gamma,sigma)
    
    u_2s_roots=[]
    for i in range(len(u_2s_list)-1):
        if delta[i]*delta[i+1]<0:
            u_2s_roots.append(u_2s_list[i])
    if sigma>0:
        u_2s_roots=np.array(u_2s_roots)
        if len(u_2s_roots)>0:
            u_2s=u_2s_roots[u_2s_roots>u_2s_0][0]
        else:
            #print('wrong')
            u_2s=u_2s_0
    else:
        u_2s=u_2s_0
    return u_2s

def downstream(gamma1,gamma2,sigma,n1):
    
    beta1=gamma2beta(gamma1)
    beta2=gamma2beta(gamma2)
    beta21=np.abs(beta2-beta1)/(1-beta2*beta1)
    gamma21=beta2gamma(beta21)
    Gamma=(4*gamma21+1)/3/gamma21
    
    u_2s=u_downstream_s(gamma1,gamma2,sigma)
    u_1s=u_2s*gamma21+(u_2s**2+1)**(1/2)*(gamma21**2-1)**(1/2)
    n2=n1*(u_1s/u_2s)
    e2=(n2*mp*ccont**2)*(gamma21-1)*(1-(gamma21+1)/(2*u_1s*u_2s)*sigma)
    p2=(Gamma-1)*e2
    pb2=p2*(1/2/(Gamma-1))*(u_1s/u_2s)*sigma*(e2/n2/mp/ccont**2)**(-1)
    
    return p2,pb2,e2,n2,u_1s


def BW_params(gamma_w, sigma_w, n_w, gamma_m, sigma_m, n_m):
    
    B_w2 = 4*np.pi*n_w*mp*ccont**2 * sigma_w
    B_w = np.sqrt(B_w2)
    
    beta_w = gamma2beta(gamma_w)
    beta_m = gamma2beta(gamma_m)
    
    N_step = 10000
    Gamma_up = gamma_w-0.001
    Gamma_low = beta2gamma(beta_m)
    
    
    for j in range(N_step):
        Gamma_mid = (Gamma_up + Gamma_low)/2
        [p_r_up, p_rb_up, e_r, n_r, u_wrs]= downstream(gamma_w,Gamma_up,sigma_w,n_w)
        [p_f_up, p_fb_up, e_f, n_f, u_mfs]= downstream(gamma_m,Gamma_up,sigma_m,n_m)
    
    
        [p_r_mid, p_rb_mid, e_r, n_r, u_wrs]= downstream(gamma_w,Gamma_mid,sigma_w,n_w)
        [p_f_mid, p_fb_mid, e_f, n_f, u_mfs]= downstream(gamma_m,Gamma_mid,sigma_m,n_m)
        
        if (p_r_up + p_rb_up - p_f_up)*(p_r_mid + p_rb_mid - p_f_mid) < 0:
            Gamma_low = Gamma_mid
        else:
            Gamma_up = Gamma_mid
        #print('Gamma_mid =', Gamma_mid, 'Gamma_m =', gamma_m)  
        #print('p_r =', p_r_mid + p_rb_mid, 'p_f = ', p_f_mid, 'Gamma_mid =', Gamma_mid) 
        if abs(p_r_mid + p_rb_mid - p_f_mid) < 1e-6*p_f_mid or (Gamma_up - Gamma_low) < 1e-5:
            Gamma = Gamma_up
            beta = gamma2beta(Gamma)
            
            beta_wm = (beta_w - beta_m)/(1-beta_w*beta_m) #??
            Gamma_wm = beta2gamma(beta_wm)
            if sigma_w < 8/3 * Gamma_wm**2 * n_m/n_w:


                p_r = p_r_mid
                p_f = p_f_mid
            
                u_bwrs = u_downstream_s(gamma_w,Gamma,sigma_w)
                u_bwfs = u_downstream_s(gamma_m,Gamma,sigma_m)
                B_r = B_w*(u_wrs/u_bwrs)
                rho_r = n_r*mp
                
                
                h_r = rho_r*ccont**2 + e_r + p_r
                rho_f = n_f*mp
                h_f = rho_f*ccont**2 + e_f + p_f
            
                beta_bwrs = u_bwrs / np.sqrt(1+u_bwrs**2)
                beta_r = (beta - beta_bwrs)/(1-beta*beta_bwrs)
            
                beta_bwfs = u_bwfs / np.sqrt(1+u_bwfs**2)
                beta_f = (beta + beta_bwfs)/(1 + beta*beta_bwfs)
            #print('Gamma_mid =', Gamma_mid)  
            break
        
    return Gamma, rho_f, rho_r, p_f, p_f, h_f, h_r, B_r, beta_r, beta_f


def BW_integral_params(Para_w, Para_m, r0, ts):
    Lsd = Para_w[0]
    gamma_w = Para_w[1]
    sigma_w = Para_w[2]
    
    Mej = Para_m[0]
    sigma_m = Para_m[1]
    beta_ej_max = Para_m[2]
    beta_ej_min = Para_m[3]
    alpha = Para_m[4]
    
    r_r = r0
    r_f = r0
    r = r0
    
    v_min = beta_ej_min*ccont
    v_max = beta_ej_max*ccont
    t0 = r0/beta_ej_min/ccont
    #print('t0 = ', t0)
    t_list = np.logspace(np.log10(t0),np.log10(t0 + ts),10)
    dv = 0.01*ccont
    for i in range(len(t_list)):
        #print('i = ', i,'t_list_i = ', t_list[i])
        delta_ej = dv * t_list[i]
        dV_ej = 4*np.pi*r_f**2*delta_ej       
        beta_c = r_f/t_list[i]/ccont
        gamma_m = beta2gamma(beta_c)
        #print('beta_c = ', beta_c)
        if beta_c > beta_ej_max:
            break
        delta_Mej = Mej*(-alpha+1)/(v_max**(-alpha+1) - v_min**(-alpha+1))*(beta_c*ccont)**(-alpha)*dv       
        n_m = delta_Mej / dV_ej / mp
        n_w = unshocked_wind(Lsd, gamma_w, r_r, sigma_w)
        #print('n_w = ', n_w)
        #print('n_m =', n_m)
        [Gamma, rho_f, rho_r, p_f, p_r, h_f, h_r, B_r, beta_r, beta_f] = \
            BW_params(gamma_w, sigma_w, n_w, gamma_m, sigma_m, n_m)
        beta = gamma2beta(Gamma)
        
        Sigma_sph = 0
        P_sph = 0
        H_sph = 0
        B_sph = 0
        if i > 0:
            dt = t_list[i] - t_list[i-1]            
            r_r = r_r + beta_r * ccont * dt
            r_f = r_f + beta_f * ccont * dt
            r = r + beta * ccont * dt
            V_ej_s = 4/3 * np.pi * (r_f**3 - r**3)
            V_wind_s = 4/3 * np.pi * (r**3 - r_r**3)
            Sigma_sph = rho_f * V_ej_s + rho_r * V_wind_s
            P_sph = p_f * V_ej_s + p_r * V_wind_s
            H_sph = h_f * V_ej_s + h_r * V_wind_s
            B_sph = B_r**2 * V_wind_s
            
    

    return Sigma_sph, P_sph, H_sph, B_sph, Gamma

if __name__ == '__main__':      
    r0 = 1e10
    ts = 0.1
    Lsd = 1.0e49
    sigma_w = 10000
    gamma_w = 1000
    Para_w = [Lsd, gamma_w, sigma_w]
    
    Mej = 1e-2*Msun
    beta_ej_min = 0.05
    beta_ej_max = 0.2
    alpha = 1.01
    sigma_m = 0
    Para_m = [Mej, sigma_m, beta_ej_max, beta_ej_min, alpha]
    
    Results = BW_integral_params(Para_w, Para_m, r0, ts)



# Para_w = [Lsd_0, gamma_w0, sigma_w0]
# Para_m = [M_ej, sigma_1, beta_max, beta_min, alpha]
# Results = BW_integral_params(Para_w, Para_m, r0, ts)
# Sigma_sph = Results[0]
# P_sph = Results[1]
# H_sph = Results[2]
# B_sph = Results[3]
# Gamma = Results[4]
# r_r = r0
# r_f = r0

############################## pressure balance #######################


    

# ## pressure balance ############# 
# log_r = np.arange(15.0,19.0,0.001)
# r = 10**log_r
# tau = 0
# tau_end = 1e16
# sigma_w = 10
# #t_eng = 0


# ########### In the wind ##############
# Lsd=1e47
# gamma_w0=500
# gamma_w=gamma_w0

# beta_w = gamma2beta(gamma_w)
# Delta_w = beta_w*ccont*tau_end
# ########################## In the ISM ##########################
# n_m = 1
# gamma_m = 1
# beta_m = 0
# sigma_m = 0
# rho_1 = n_m*mp
# #print('rho_1 =',rho_1)

# ##############################################################
# r_r = r[0]
# r_f = r[0]
# Delta_3 = 0
# Delta_2 = 0
# E_w = 0
# dtau = 0
# Gamma_list = np.zeros(len(r))
# E_bw_list = np.zeros(len(r))
# E_bwp_list = np.zeros(len(r))
# E_ISM_list = np.zeros(len(r))
# E_ISMp_list = np.zeros(len(r))
# E_w_list = np.zeros(len(r))
# E_inj_list = np.zeros(len(r))
# pr_list = np.zeros(len(r))
# pf_list = np.zeros(len(r))
# prb_list = np.zeros(len(r))
# rho_r_list = np.zeros(len(r))
# rho_f_list = np.zeros(len(r))
# r_r_list = np.zeros(len(r))
# r_f_list = np.zeros(len(r))
# beta_r_list = np.zeros(len(r))
# tau_list = np.zeros(len(r))
# dN1_list = []
# dN2_list = []
# K1_list = []
# K2_list = []

# E_inj=0
# E_bwpp = 0
# E_bwp = 0
# Gamma_w_p = -9
# BBi = 0
# Hi=0
# Pi=0
# for i in range(1,len(r)):
#     if i%100 == 0: print(i/len(r))
#     dr = r[i] - r[i-1]
#     #n_w = unshocked_wind1(Lsd, gamma_w,Gamma_w_p, r_r, tau ,rho_1)  
#     n_w = unshocked_wind(Lsd, gamma_w, r_r, sigma_w)
#     B_w2 = 4*np.pi*n_w*mp*ccont**2 * sigma_w
#     B_w = np.sqrt(B_w2)
#     rho_w = n_w*mp
    
#     #print(n_w*mp*beta_w*gamma_w**2)
#     #print('rho_w =', rho_w)
#     #print('gamma_w =',gamma_w)


# ######################### input value in the blast wave #######









#     N_step = 10000

#     Gamma_up = gamma_w-0.001
#     Gamma_low = 1
    
#     for j in range(N_step):
#         Gamma_mid = (Gamma_up + Gamma_low)/2
#         [p_r_up, p_rb_up, e_r, n_r, u_wrs]= downstream(gamma_w,Gamma_up,sigma_w,n_w)
#         [p_f_up, p_fb_up, e_f, n_f, u_mfs]= downstream(gamma_m,Gamma_up,sigma_m,n_m)
    
    
#         [p_r_mid, p_rb_mid, e_r, n_r, u_wrs]= downstream(gamma_w,Gamma_mid,sigma_w,n_w)
#         [p_f_mid, p_fb_mid, e_f, n_f, u_mfs]= downstream(gamma_m,Gamma_mid,sigma_m,n_m)
        
#         if (p_r_up + p_rb_up - p_f_up)*(p_r_mid + p_rb_mid - p_f_mid) < 0:
#             Gamma_low = Gamma_mid
#         else:
#             Gamma_up = Gamma_mid
#         if abs(Gamma_up - Gamma_low) < 1e-6:
#             Gamma = Gamma_up
#             beta = gamma2beta(Gamma)
            
#             beta_bww = (beta_w - beta)/(1-beta_w*beta)
#             Gamma_bww = beta2gamma(beta_bww)
#             #print(Gamma_bww)
#             if Gamma_bww > np.sqrt(1+sigma_w):
#                 kappa10 = 1/3*(1+1/Gamma_bww)
#                 kappa20 = 1/3*(1+1/Gamma)
            
#                 pm10 = kappa10**(5/2)*(2/3-kappa10)**(5/2)/(kappa10-1/3)**4
#                 pm20 = kappa20**(5/2)*(2/3-kappa20)**(5/2)/(kappa20-1/3)**4
            
#                 p_r = p_r_mid
#                 p_rb = p_rb_mid
#                 p_f = p_f_mid
            
#                 K1 = p_r / pm10
#                 K2 = p_f / pm20
            
#                 print(gamma_w,Gamma,Gamma_bww)
#                 u_bwrs = u_downstream_s(gamma_w,Gamma,sigma_w)
#                 u_bwfs = u_downstream_s(gamma_m,Gamma,sigma_m)
#                 B_r = B_w*(u_wrs/u_bwrs)
#                 rho_r = n_r*mp
#                 sigma = B_r**2/4/pi/rho_r/ccont**2
                
#                 #dN = Lsd/(1+sigma_w)/(beta*ccont) * (beta_w - beta_r)/beta_w * dr /(gamma_w*ccont**2*mp)
#                 #dBBi = sigma*4*pi*ccont**2*mp*dN/Gamma
#                 #BBi = BBi + dBBi
                
#                 h_r = rho_r*ccont**2 + e_r + p_r
#                 rho_f = n_f*mp
#                 h_f = rho_f*ccont**2 + e_f + p_f
            
#                 beta_bwrs = u_bwrs / np.sqrt(1+u_bwrs**2)
#                 beta_r = (beta - beta_bwrs)/(1-beta*beta_bwrs)
            
#                 beta_bwfs = u_bwfs / np.sqrt(1+u_bwfs**2)
#                 beta_f = (beta + beta_bwfs)/(1 + beta*beta_bwfs)
            
#                 dr_r_dr = beta_r/beta
#                 dr_f_dr = beta_f/beta
#                 r_r = r_r + dr_r_dr*dr
#                 r_f = r_f + dr_f_dr*dr
#                 dDelta3_dr = 1 - beta_r/beta
#                 dDelta2_dr = beta_f/beta - 1 
            
#                 Delta_3 = Delta_3 + dDelta3_dr*dr
#                 Delta_2 = Delta_2 + dDelta2_dr*dr
                
#                 Delta_w = Delta_w - (beta_w - beta_r)/beta * dr 
                
#                 #dN = Lsd/(1+sigma_w)/(beta*ccont) * (beta_w - beta_r)/beta_w * dr /(gamma_w*ccont**2*mp)
#                 #dBBi = sigma*4*pi*ccont**2*mp*dN/Gamma
#                 #BBi = BBi + dBBi
#                 if len(Gamma_list[Gamma_list>0]) == 0:
#                     B_r1 = B_r
#                     h_r1 = h_r
#                     p_r1 = p_r
#                     h_f1 = h_f
#                     p_f1 = p_f
                
#                     Delta_3_1 = Delta_3
#                     Delta_2_1 = Delta_2
#                     BB1 = B_r1*Delta_3_1
#                     H1 = h_r1*Delta_3_1 + h_f1*Delta_2_1
#                     P1 = p_r1*Delta_3_1 + p_f1*Delta_2_1
                    
#                     dBB = B_r**2*Delta_3 - BB1
#                     dH = h_r*Delta_3 + h_f*Delta_2 - H1
#                     dP = p_r*Delta_3 + p_f*Delta_2 - P1
                
                
            
#                 BB = B_r**2*Delta_3
#                 H = h_r*Delta_3 + h_f*Delta_2
#                 P = p_r*Delta_3 + p_f*Delta_2
                
#                 dBB = BB - BB1
#                 dH = H - H1
#                 dP = P - P1
                
#                 dBBi = 4*pi*r_r**2*dBB + 8*pi*r[i]*BB
#                 dHi = 4*pi*r_r**2*dH + 8*pi*r[i]*H
#                 dPi = 4*pi*r_r**2*dP + 8*pi*r[i]*P
                
#                 BBi = BBi + dBBi
#                 Hi = Hi + dHi
#                 Pi = Pi + dPi
                
                

                
                
#                 B_r1 = B_r
#                 h_r1 = h_r
#                 p_r1 = p_r
#                 h_f1 = h_f
#                 p_f1 = p_f
                
#                 Delta_3_1 = Delta_3
#                 Delta_2_1 = Delta_2
#                 BB1 = B_r1*Delta_3_1
#                 H1 = h_r1*Delta_3_1 + h_f1*Delta_2_1
#                 P1 = p_r1*Delta_3_1 + p_f1*Delta_2_1
                
                
    
#                 E_inj = E_inj + Lsd/(beta*ccont) * (beta_w - beta_r)/beta_w *   dr
#                 E_ISM = rho_1*ccont**2*(4/3*pi)*((10**18)**3 - r_f**3)
#                 E_ISMp = rho_1*ccont**2*(4/3*pi)*(r_f**3 - r_r**3)
#                 #E_bw = 4*pi*((r_r+r_f)/2)**2* (Gamma**2*h_r-p_r+Gamma**2*B_r**2/4/pi)*(r[i]-r_r) \
#                 #        + 4*pi*((r_r+r_f)/2)**2* (Gamma**2*h_f-p_f)*(r_f-r[i])
#                 #E_bw = 4*pi/3*(Gamma**2*h_r-p_r+Gamma**2*B_r**2/4/pi)*(r[i]**3 - r_r**3)\
#                 #        + 4*pi/3*(Gamma**2*h_f-p_f)*(r_f**3 - r[i]**3)
#                 E_bw = (Gamma**2*H - P + Gamma**2*BB/(4*np.pi))*4*pi*r[i]**2
#             #print(beta)
#                 E_bwp = Gamma**2*Hi - Pi + Gamma**2*BBi/4/pi
            
#             #dV = Lsd*dtau_dr*dr*(gamma43-1)/(3*gamma43)/p_r/gamma_w
#             #E_bwp = E_bwp + (Gamma**2*(p_r + e_r) - p_r)*(dV/Gamma)
#                 #dN1 = Lsd*dtau/(gamma_w*ccont**2*mp)
#                 #dN2 = (4*pi*r_f**2*beta_f/beta*dr)*n_m
            
#             #E_bwpp = E_bwpp + 4*pi*r_r**2*(h_r - p_r + \
#             #Gamma**2*B_r**2/(4*pi))*(beta-beta_r)/beta*dr + 4*pi*r_f**2*(h_r - p_f)*(beta_f-beta)/beta*dr
#             #E_bwp = Gamma**2*E_bwpp
            
#                 pr_list[i] = p_r
#                 pf_list[i] = p_f
#                 prb_list[i] = p_rb
#                 rho_r_list[i]=rho_r
#                 rho_f_list[i]=rho_f
#                 r_r_list[i]=r_r
#                 r_f_list[i]=r_f
#                 beta_r_list[i] = beta_r
            
    
    
#                 Gamma_list[i] = Gamma
#                 E_bw_list[i] = E_bw
#                 E_bwp_list[i] = E_bwp
#                 E_ISM_list[i] = E_ISM
#                 E_ISMp_list[i] = E_ISMp
#                 E_inj_list[i] = E_inj
# #                dN1_list.append(dN1)
# #                dN2_list.append(dN2)
# #                K1_list.append(K1)
# #                K2_list.append(K2)
            
#                 print('Gamma=',Gamma)
#             else: 
#                 r_r = r[i]
#                 r_f = r[i]
#             #print('p_r =', p_r, 'p_rb =', p_rb, 'p_f =', p_f)
#             #print(B_w**2/(8*np.pi))
#             #print('Gamma_bww =', Gamma_bww)
#             #print('sigma_w', sigmaw, 'rhs =',8/3*gamma_w**2*(n_m/n_w))
            
#             break
     
    

    
    
#     #if tau > 50:
#     #    break
#     if Delta_w < 0:
#         break


            
            
# np.save('pressure_balance/new/Gamma'+str(sigma_w)+'.npy',Gamma_list)
# np.save('pressure_balance/new/pr'+str(sigma_w)+'.npy',pr_list)
# np.save('pressure_balance/new/pf'+str(sigma_w)+'.npy',pf_list)
# np.save('pressure_balance/new/prb'+str(sigma_w)+'.npy',prb_list)
# np.save('pressure_balance/new/rho_r'+str(sigma_w)+'.npy',rho_r_list)
# np.save('pressure_balance/new/rho_f'+str(sigma_w)+'.npy',rho_f_list)
# np.save('pressure_balance/new/r_r'+str(sigma_w)+'.npy',r_r_list)
# np.save('pressure_balance/new/r_f'+str(sigma_w)+'.npy',r_f_list)
# np.save('pressure_balance/new/beta_r'+str(sigma_w)+'.npy',beta_r_list)
# np.save('pressure_balance/new/E_w'+str(sigma_w)+'.npy',E_w_list)
# np.save('pressure_balance/new/E_bw'+str(sigma_w)+'.npy',E_bw_list)
# np.save('pressure_balance/new/E_bwp'+str(sigma_w)+'.npy',E_bwp_list)
# np.save('pressure_balance/new/E_ISM'+str(sigma_w)+'.npy',E_ISM_list)
# np.save('pressure_balance/new/E_ISMp'+str(sigma_w)+'.npy',E_ISMp_list)
# np.save('pressure_balance/new/E_inj'+str(sigma_w)+'.npy',E_inj_list)
# np.save('pressure_balance/new/r'+str(sigma_w)+'.npy',r)
# np.save('pressure_balance/new/tau'+str(sigma_w)+'.npy',tau_list)


