#%% CSAR
'''
Rutina para leer .csv del sensor Rugged y calcular dT/dt 
en la Temperatura de Equilibrio 
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from glob import glob
from datetime import datetime
from uncertainties import ufloat, unumpy
from scipy.optimize import curve_fit
#%% Lector Templog
def lector_templog(path):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt). 
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura 
    '''
    data = pd.read_csv(path,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python') 
    temp_CH1  = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2  = pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']]) 

    return timestamp,temp_CH1, temp_CH2
#%% Levanto data
#NE
# path_agua='250624_120427_agua.csv'
path_NE_300_150='NE@citrato - coprecipitacion/CSAR/260205_044538_300_150.csv'
path_NE_300_100='NE@citrato - coprecipitacion/CSAR/260205_045254_300_100.csv'
path_NE_300_050='NE@citrato - coprecipitacion/CSAR/260205_045843_300_050.csv' 

path_NE_212_150='NE@citrato - coprecipitacion/CSAR/260205_051305_212_150.csv'
path_NE_212_100='NE@citrato - coprecipitacion/CSAR/260205_052035_212_100.csv'
path_NE_212_050='NE@citrato - coprecipitacion/CSAR/260205_052900_212_050.csv'

path_NE_135_150='NE@citrato - coprecipitacion/CSAR/260205_042615_135_150.csv'

t_NE_300_150,T_NE_300_150,_=lector_templog(path_NE_300_150)
t_NE_300_100,T_NE_300_100,_=lector_templog(path_NE_300_100)
t_NE_300_050,T_NE_300_050,_=lector_templog(path_NE_300_050)

t_NE_212_150,T_NE_212_150,_=lector_templog(path_NE_212_150)
t_NE_212_100,T_NE_212_100,_=lector_templog(path_NE_212_100)
t_NE_212_050,T_NE_212_050,_=lector_templog(path_NE_212_050)

t_NE_135_150,T_NE_135_150,_=lector_templog(path_NE_135_150)

t_NE_300_150 = np.array([(t-t_NE_300_150[0]).total_seconds() for t in t_NE_300_150])
t_NE_300_100 = np.array([(t-t_NE_300_100[0]).total_seconds() for t in t_NE_300_100])
t_NE_300_050 = np.array([(t-t_NE_300_050[0]).total_seconds() for t in t_NE_300_050])

t_NE_212_150 = np.array([(t-t_NE_212_150[0]).total_seconds() for t in t_NE_212_150])
t_NE_212_100 = np.array([(t-t_NE_212_100[0]).total_seconds() for t in t_NE_212_100])
t_NE_212_050 = np.array([(t-t_NE_212_050[0]).total_seconds() for t in t_NE_212_050])

t_NE_135_150 = np.array([(t-t_NE_135_150[0]).total_seconds() for t in t_NE_135_150])



#%%NF 
path_agua='NF@citrato - solvotermal - concentrada/CSAR/260210_011936_135_150_agua.csv'
path_NF_300_150='NF@citrato - solvotermal - concentrada/CSAR/260210_023412_NF_300_150.csv'
path_NF_300_100='NF@citrato - solvotermal - concentrada/CSAR/260210_023827_NF_300_100.csv'
path_NF_300_050='NF@citrato - solvotermal - concentrada/CSAR/260210_024507_NF_300_050.csv'
path_NF_212_150='NF@citrato - solvotermal - concentrada/CSAR/260210_021428_NF_212_150.csv'
path_NF_212_100='NF@citrato - solvotermal - concentrada/CSAR/260210_022109_NF_212_100.csv'
path_NF_135_150='NF@citrato - solvotermal - concentrada/CSAR/260210_015944_NF_135_150.csv'

t_eq,T_eq,_=lector_templog(path_agua)
t_eq = np.array([(t-t_eq[0]).total_seconds() for t in t_eq])

fig,ax=plt.subplots(figsize=(18,10), constrained_layout=True)
ax.plot(t_eq,T_eq,'.-')
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
ax.grid()
plt.title('Agua - NF@citrato - solvotermal - concentrada    ',loc='left')


t_NF_300_150,T_NF_300_150,_=lector_templog(path_NF_300_150)
t_NF_300_100,T_NF_300_100,_=lector_templog(path_NF_300_100)
t_NF_300_050,T_NF_300_050,_=lector_templog(path_NF_300_050)

t_NF_212_150,T_NF_212_150,_=lector_templog(path_NF_212_150)
t_NF_212_100,T_NF_212_100,_=lector_templog(path_NF_212_100)

t_NF_135_150,T_NF_135_150,_=lector_templog(path_NF_135_150) 

t_NF_300_150 = np.array([(t-t_NF_300_150[0]).total_seconds() for t in t_NF_300_150])
t_NF_300_100 = np.array([(t-t_NF_300_100[0]).total_seconds() for t in t_NF_300_100])
t_NF_300_050 = np.array([(t-t_NF_300_050[0]).total_seconds() for t in t_NF_300_050])

t_NF_212_150 = np.array([(t-t_NF_212_150[0]).total_seconds() for t in t_NF_212_150])
t_NF_212_100 = np.array([(t-t_NF_212_100[0]).total_seconds() for t in t_NF_212_100])

t_NF_135_150 = np.array([(t-t_NF_135_150[0]).total_seconds() for t in t_NF_135_150])
#%% Ploteo NE
fig, (ax,ax1,ax2)=plt.subplots(3,1,figsize=(10,8),constrained_layout=True,sharex=True)

ax.set_title('300 kHz',loc='left')

ax.plot(t_NE_300_150,T_NE_300_150,label='300_150')
ax.plot(t_NE_300_100,T_NE_300_100,label='300_100')
ax.plot(t_NE_300_050,T_NE_300_050,label='300_050')

ax1.set_title('212 kHz',loc='left')
ax1.plot(t_NE_212_150,T_NE_212_150,label='212_150')
ax1.plot(t_NE_212_100,T_NE_212_100,label='212_100')
ax1.plot(t_NE_212_050,T_NE_212_050,label='212_050')

ax2.set_title('135 kHz',loc='left')
ax2.plot(t_NE_135_150,T_NE_135_150,label='135_150')    
# ax.plot(t_FF2_0,T_FF2,label='FF2')
for a in (ax,ax1,ax2):
    a.grid()
    a.legend()
    a.set_xlim(0,)
plt.suptitle('NE@citrico - coprecipitacion',fontsize=16)
plt.savefig('T_vs_t.png', dpi=300)
plt.show()
#%% Ploteo NF
fig, (ax,ax1,ax2)=plt.subplots(3,1,figsize=(9,7),constrained_layout=True,sharex=True,sharey=True)

ax.set_title('300 kHz',loc='left')
ax.plot(t_NF_300_150,T_NF_300_150,'.-',label='300_150')
ax.plot(t_NF_300_100,T_NF_300_100,'.-',label='300_100')
ax.plot(t_NF_300_050,T_NF_300_050,'.-',label='300_050')

ax1.set_title('212 kHz',loc='left')
ax1.plot(t_NF_212_150,T_NF_212_150,'.-',label='212_150')
ax1.plot(t_NF_212_100,T_NF_212_100,'.-',label='212_100')

ax2.set_title('135 kHz',loc='left')
ax2.plot(t_NF_135_150,T_NF_135_150,'.-',label='135_150')    
# ax.plot(t_FF2_0,T_FF2,label='FF2')
for a in (ax,ax1,ax2):
    a.grid()
    a.legend()
    a.set_xlim(0,)
plt.suptitle('NF@citrico - solvotermal - concentrada',fontsize=16)
plt.savefig('T_vs_t_NF.png', dpi=300)
# plt.xlim(0,150)
plt.show()
#%% Rcorto a valor minimo
# indx_min=np.nonzero(T_NE_300_150==T_NE_300_150.min())[0][0]
indx_min=[]
for i,e in enumerate([T_NE_300_150,T_NE_300_100,T_NE_300_050,
                     T_NE_212_150,T_NE_212_100,T_NE_212_050,
                     T_NE_135_150]):
    indx_min.append(np.nonzero(e==e.min())[0][0])
    print(f'Indice minimo {i} : {np.nonzero(e==e.min())[0][0]}')


#%%
fig, (ax,ax1,ax2)=plt.subplots(3,1,figsize=(10,8),constrained_layout=True,sharex=True)

ax.set_title('300 kHz',loc='left')

ax.plot(t_NE_300_150,T_NE_300_150,'.-',label='300_150')
# ax.plot(t_NE_300_150[indx_min[0]:],T_NE_300_150[indx_min[0]:],'.',label='300_150')
ax.plot(t_NE_300_100,T_NE_300_100,'.-',label='300_100')
ax.plot(t_NE_300_050,T_NE_300_050,'.-',label='300_050')

ax1.set_title('212 kHz',loc='left')

ax1.plot(t_NE_212_150,T_NE_212_150,'.-',label='212_150')
ax1.plot(t_NE_212_100,T_NE_212_100,'.-',label='212_100')
ax1.plot(t_NE_212_050,T_NE_212_050,'.-',label='212_050')

ax2.set_title('135 kHz',loc='left')
ax2.plot(t_NE_135_150,T_NE_135_150,'.-',label='135_150')    
# ax2.plot(t_NE_135_150[indx_min[6]:],T_NE_135_150[indx_min[6]:],'.',label='135_150')    

for a in (ax,ax1,ax2):
    a.grid()
    a.legend()
    a.set_xlim(0,)
plt.suptitle('NE@citrico - coprecipitacion',fontsize=16)
plt.show()

#%% Funcion Ajuste lineal T arbitraria, intervalo arbitario
def ajustes_lineal_T_arbitraria(Tcentral, t, T, label, x=1.0,guardar=False):
    """
    Realiza ajustes lineal alrededor de Tcentral ± x usando curve_fit.
    
    Args:
        Tcentral (float): Temperatura de equilibrio
        t (np.array): Array de tiempos
        T (np.array): Array de temperaturas
        x (float): Rango alrededor de Tcentral (default=1.0)
        
    Returns:
        tuple: (dict_lin, dict_exp) donde:
            - dict_lin: Diccionario con resultados del ajuste lineal
    """
    # Definir la función lineal para curve_fit
    def linear_func(x, a, b):
        return a * x + b
    
    # Crear máscara para el intervalo de interés
    mask = (T >= Tcentral - x) & (T <= Tcentral + x)
    t_interval = t[mask]
    T_interval = T[mask]
    
    # Ajuste lineal con curve_fit
    popt, pcov = curve_fit(linear_func, t_interval, T_interval)
    perr = np.sqrt(np.diag(pcov))  # Desviaciones estándar de los parámetros
    
    # Crear función de ajuste
    poly_lin = lambda x: linear_func(x, *popt)
    
    # Calcular R²
    residuals = T_interval - poly_lin(t_interval)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((T_interval - np.mean(T_interval))**2)
    r2_lin = 1 - (ss_res / ss_tot)
    
    t_fine = np.linspace(t_interval.min()-80, t_interval.max()+80, 100)
    
    # Crear ufloat para la pendiente con su incertidumbre
    pendiente_ufloat = ufloat(popt[0], perr[0])
    
    # Preparar diccionario para resultados lineales
    dict_lin = {
        'pendiente': pendiente_ufloat,
        'ordenada': ufloat(popt[1], perr[1]),
        'r2': r2_lin,
        't_interval': t_interval,
        'T_interval': T_interval,
        'funcion': poly_lin,
        'ecuacion': f"({popt[0]:.3f}±{perr[0]:.3f})t + ({popt[1]:.3f}±{perr[1]:.3f})",
        'rango_x': x,
        'AL_t': t_fine,
        'AL_T': poly_lin(t_fine),
        'covarianza': pcov
    }
    
    # Crear figura 
    fig, ax = plt.subplots(figsize=(12,6), constrained_layout=True)
    ax.plot(t, T, '.-', label=label)
    
    # Plotear ajustes con el rango extendido que definiste
    ax.plot(t_fine, poly_lin(t_fine), '-', c='tab:green', lw=2, 
            label=f'Ajuste lineal: {dict_lin["ecuacion"]} (R²={r2_lin:.3f})')

    ax.axhspan(Tcentral-x, Tcentral+x, 0, 1, color='tab:red', alpha=0.3, 
               label='T$_{eq}\pm\Delta T$ ='+ f' {Tcentral:.1f} $\pm$ {x} ºC')
    
    ax.set_xlabel('t (s)')
    ax.set_ylabel('T (°C)')
    ax.grid()
    ax.legend()
    
    if guardar:
        plt.savefig(f'{label}_ajuste_lineal.png', dpi=300)
    plt.show()

    # Imprimir resultados (manteniendo tu formato)
    print("\nResultados del ajuste lineal:")
    print(f"Pendiente: {dict_lin['pendiente']} °C/s")
    print(f"Ordenada: {dict_lin['ordenada']} °C")
    print(f"Coeficiente R²: {dict_lin['r2']:.5f}")
    
    
    return dict_lin
#%%# 2 mar 26 
# en estas medidas no anote el horario de campo ON y campo OFF
# calculo los rates con ajuste lineal  
#300
t_NE_300_050=t_NE_300_050[:440]

T_NE_300_050=T_NE_300_050[:440]
T_interm_NE_300_150 = (T_NE_300_150.max()+T_NE_300_150.min())/2
T_interm_NE_300_100 = (T_NE_300_100.max()+T_NE_300_100.min())/2

res_NE_300_150 = ajustes_lineal_T_arbitraria(T_interm_NE_300_150, t_NE_300_150, T_NE_300_150,'NE 300_150', x=5, guardar=True)
res_NE_300_100 = ajustes_lineal_T_arbitraria(T_interm_NE_300_100, t_NE_300_100, T_NE_300_100,'NE 300_100', x=4, guardar=True)
res_NE_300_050 = ajustes_lineal_T_arbitraria(27.2, t_NE_300_050, T_NE_300_050,'NE 300_050', x=2, guardar=True)
res_NE_300_050_bis = ajustes_lineal_T_arbitraria(33, t_NE_300_050, T_NE_300_050,'NE 300_050_bis', x=1, guardar=True)
#%212
# recorto para evitar problemas de ajuste 
t_NE_212_150=t_NE_212_150[:240]
T_NE_212_150=T_NE_212_150[:240]

t_NE_212_100=t_NE_212_100[:340]
T_NE_212_100=T_NE_212_100[:340]

t_NE_212_050=t_NE_212_050[:1000]
T_NE_212_050=T_NE_212_050[:1000]

T_interm_NE_212_150 = (T_NE_212_150.max()+T_NE_212_150.min())/2
T_interm_NE_212_100 = (T_NE_212_100.max()+T_NE_212_100.min())/2

res_NE_212_150 = ajustes_lineal_T_arbitraria(T_interm_NE_212_150, t_NE_212_150, T_NE_212_150,'NE 212_150', x=5, guardar=True)
res_NE_212_100 = ajustes_lineal_T_arbitraria(T_interm_NE_212_100, t_NE_212_100, T_NE_212_100,'NE 212_100', x=5.5, guardar=True)
res_NE_212_050 = ajustes_lineal_T_arbitraria(25, t_NE_212_050, T_NE_212_050,'NE 212_050', x=1, guardar=True)
res_NE_212_050_bis = ajustes_lineal_T_arbitraria(29.2, t_NE_212_050, T_NE_212_050,'NE 212_050_bis', x=0.75, guardar=True)
#% 135
t_NE_135_150=t_NE_135_150[:400]
T_NE_135_150=T_NE_135_150[:400]
T_interm_NE_135_150 = (T_NE_135_150.max()+T_NE_135_150.min())/2
res_NE_135_150 = ajustes_lineal_T_arbitraria(T_interm_NE_135_150, t_NE_135_150, T_NE_135_150,'NE 135_150', x=4, guardar=True)


#%%
frecs=[300,300,300,300,212,212,212,212,135]
campos=[57,38,20,20,57,38,20,20,57]
pendientes=[res_NE_300_150['pendiente'],res_NE_300_100['pendiente'],res_NE_300_050['pendiente'],res_NE_300_050_bis['pendiente'],
            res_NE_212_150['pendiente'],res_NE_212_100['pendiente'],res_NE_212_050['pendiente'],res_NE_212_050_bis['pendiente'],
            res_NE_135_150['pendiente']]
concentracion=ufloat(9,1) #mg/ml

print('frecuencia (kHz)', 'campo (kA/m)', 'pendiente (°C/s)', 'CSAR (W/g)')
for i,p in enumerate(pendientes):
    csar=p*4.186e3/concentracion
    print(f'{frecs[i]:^14.0f}',f'{campos[i]:^14.0f}',f'{pendientes[i]:^14.1uS}',f'{csar:8.1uS} W/g')


