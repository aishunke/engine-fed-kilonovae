Change input file, run main.py 
Annotation of the code is not complete now, I will update the instruction soon. 

Input file: 

  Inputs/Input/Input.txt: 
  
  M_ej_sun: ejecta mass in unit of solar mass 
    
  beta_min: minimum dimensionless velocity for the ejecta velocity profile 
    
  beta_max: maximum dimensionless velocity for the ejecta velocity profile 
    
  alpha: the profile index for the density of the ejecta as a function of velocity $\rho \propto v^{-\alpha}$ 

  kappa: opacity of the ejecta in unit of ${\rm cm^2/g}$
  
  Lsd_0: initial spin-down luminosity of the post merger magnetar in unit of ${\rm erg/s}$

  E_tot: total energy that would be relased through magnetar wind

  xi_sd_x: fraction of spin-down energy that would dissipate into X-ray photons.

  Inputs/Input_mp/Input_main.txt:

  The input file for multi-threaded computing. One needs to set "multiinputs = True" in "main.py", which is default to be "False".

  The input parameters in "Input_main.txt" are similar as those in "Input.txt", but with parameter ranges and resolutions listed.



  Output file:

  Output files is stored in the folder "Results_sh_x”. The filename contains the information of spin-down luminosity, total energy budget, opacity of the ejecta, mass of the ejecta, and fraction of spin-down energy that would dissipate into X-ray photons. 

  Column 1: time (seconds) in observer’s frame

  Column 2: bolometric luminosity in unit of $erg/s$

  Column 3 - 14: $\nu L_{\nu}$ in unit of $erg/s$, in observational bands collected in “band.py”

  
    
