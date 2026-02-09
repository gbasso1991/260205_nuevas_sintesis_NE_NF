#%%Comparador resultados de medidas ESAR sintesis de Elisa 
# Medidas a 300 kHz y 57 kA/m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import chardet
import re
import os
from uncertainties import ufloat
#%% Funciones
def plot_ciclos_promedio(directorio):
    # Buscar recursivamente todos los archivos que coincidan con el patrón
    archivos = glob(os.path.join(directorio, '**', '*ciclo_promedio*.txt'), recursive=True)

    if not archivos:
        print(f"No se encontraron archivos '*ciclo_promedio.txt' en {directorio} o sus subdirectorios")
        return
    fig,ax=plt.subplots(figsize=(8, 6),constrained_layout=True)
    for archivo in archivos:
        try:
            # Leer los metadatos (primeras líneas que comienzan con #)
            metadatos = {}
            with open(archivo, 'r') as f:
                for linea in f:
                    if not linea.startswith('#'):
                        break
                    if '=' in linea:
                        clave, valor = linea.split('=', 1)
                        clave = clave.replace('#', '').strip()
                        metadatos[clave] = valor.strip()

            # Leer los datos numéricos
            datos = np.loadtxt(archivo, skiprows=9)  # Saltar las 8 líneas de encabezado/metadatos

            tiempo = datos[:, 0]
            campo = datos[:, 3]  # Campo en kA/m
            magnetizacion = datos[:, 4]  # Magnetización en A/m

            # Crear etiqueta para la leyenda
            nombre_base = os.path.split(archivo)[-1].split('_')[1]
            #os.path.basename(os.path.dirname(archivo))  # Nombre del subdirectorio
            etiqueta = f"{nombre_base}"

            # Graficar

            ax.plot(campo, magnetizacion, label=etiqueta)

        except Exception as e:
            print(f"Error procesando archivo {archivo}: {str(e)}")
            continue

    plt.xlabel('H (kA/m)')
    plt.ylabel('M (A/m)')
    plt.title(f'Comparación de ciclos de histéresis {os.path.split(directorio)[-1]}')
    plt.grid(True)
    plt.legend()  # Leyenda fuera del gráfico
    plt.savefig('comparativa_ciclos_'+os.path.split(directorio)[-1]+'.png',dpi=300)
    plt.show()
#%% LECTOR RESULTADOS
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))
                    
                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor
                    
                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata

#%% Localizo ciclos y resultados
ciclos_NF=glob(('**/*ciclo_promedio*'),recursive=True)
ciclos_NF.sort()
labels=['NF']

#%% comparo los ciclos promedio
plot_ciclos_promedio(os.getcwd())

#%% SAR y tau 

res_NF = glob('**/*resultados*',recursive=True)
res=[res_NF]
for r in res:
    r.sort()
    
SAR_NF,err_SAR_NF,tau_NF,err_tau_NF,hc_NF, err_hc_NF = [],[],[],[],[],[]
SAR_dil,err_SAR_dil,tau_dil,err_tau_dil,hc_dil, err_hc_dil = [],[],[],[],[],[] 
    
for C in res_NF:
    meta, _,_,_,_,_,_,_,_,_,_,_,SAR,tau,_= lector_resultados(C)
    SAR_NF.append(meta['SAR_W/g'].n)
    err_SAR_NF.append(meta['SAR_W/g'].s)
    tau_NF.append(meta['tau_ns'].n)
    err_tau_NF.append(meta['tau_ns'].s)
    hc_NF.append(meta['Hc_kA/m'].n)
    err_hc_NF.append(meta['Hc_kA/m'].s)   
# for C in res_dil:
#     meta, _,_,_,_,_,_,_,_,_,_,_,SAR,tau,_= lector_resultados(C)
#     SAR_dil.append(meta['SAR_W/g'].n)
#     err_SAR_dil.append(meta['SAR_W/g'].s)
#     tau_dil.append(meta['tau_ns'].n)
#     err_tau_dil.append(meta['tau_ns'].s)
#     hc_dil.append(meta['Hc_kA/m'].n)
#     err_hc_dil.append(meta['Hc_kA/m'].s)
    
    
#%% SAR# 
fig, a = plt.subplots(nrows=1, figsize=(7,5), constrained_layout=True)
categories = ['NF']

for i, (sarC, errC) in enumerate(zip([SAR_NF],
                    [err_SAR_NF])):
    x_pos = [i+1]*len(sarC)  # Posición X fija para cada categoría
    a.errorbar(x_pos, sarC, yerr=errC, fmt='.', 
                 label=categories[i], 
                 capsize=5, linestyle='None')

a.set_title('Muestras - 300 kHz - 57 kA/m', loc='left')
a.set_xlabel('Categorías')
a.set_ylabel('SAR (W/g)')
a.legend(ncol=2, loc='upper right')
a.grid(True, axis='y', linestyle='--')

# CORRECCIÓN: CoNFigurar correctamente las etiquetas del eje X
a.set_xticks([1, 2])  # Posiciones donde quieres las etiquetas
a.set_xticklabels(categories)  # Etiquetas correspondientes

plt.savefig('comparativa_SAR_NF_dil.png', dpi=300)
plt.show()


#%% Hc
fig, a = plt.subplots(nrows=1, figsize=(7,5), constrained_layout=True)
categories = ['conc', 'dil']

for i, (hc, err_hc) in enumerate(zip([hc_conc, hc_dil],
                    [err_hc_conc, err_hc_dil])):
    x_pos = [i+1]*len(hc)  # Posición X fija para cada categoría
    a.errorbar(x_pos, hc, yerr=err_hc, fmt='.', 
                 label=categories[i], 
                 capsize=5, linestyle='None')

a.set_title('Muestras - 300 kHz - 57 kA/m', loc='left')
a.set_xlabel('Categorías')
a.set_ylabel('Hc (kA/m)')
a.legend(ncol=2, loc='upper right')
a.grid(True, axis='y', linestyle='--')

# CORRECCIÓN: Coconcigurar correctamente las etiquetas del eje X
a.set_xticks([1, 2])  # Posiciones donde quieres las etiquetas
a.set_xticklabels(categories)  # Etiquetas correspondientes

plt.savefig('comparativa_Hc_conc_dil.png', dpi=300)
plt.show()
#%% SAR y tau - Versión mejorada para arrays completos

res_conc = glob('conc/**/*resultados*', recursive=True)
res_dil = glob('dil/**/*resultados*', recursive=True)
res = [res_conc, res_dil]

for r in res:
    r.sort()

# Listas para almacenar los arrays completos de SAR
SAR_arrays_conc, SAR_arrays_dil = [], []
Hc_arrays_conc, Hc_arrays_dil = [], []

# Extraer todos los arrays de SAR para cada categoría
for C in res_conc:
    meta, _, _, _, _, Hc, _, _, _, _, _, _, SAR, tau, _ = lector_resultados(C)
    SAR_arrays_conc.append(SAR)
    Hc_arrays_conc.append(Hc)
for C in res_dil:
    meta, _, _, _, _, Hc, _, _, _, _, _, _, SAR, tau, _ = lector_resultados(C)
    SAR_arrays_dil.append(SAR)
    Hc_arrays_dil.append(Hc)

#%% Gráfico de SAR vs índice (2 filas, 1 columna)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 7), constrained_layout=True,sharex=True)
categorias = ['conc', 'dil']
colores = ['tab:blue', 'tab:red']

# Función para graficar cada categoría
def graficar_sar_vs_indice(ax, sar_arrays, categoria, color, idx):
    for i, sar_array in enumerate(sar_arrays):
        indices = np.arange(len(sar_array))  # Índices del array
        ax.plot(indices, sar_array, 'o-', alpha=0.7, linewidth=1, markersize=4,
                label=f'{categoria} {i+1}', color=color)
    
    ax.set_title(f'{categoria} - SAR vs Índice', loc='left')

    ax.set_ylabel('SAR (W/g)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Ajustar límites del eje Y para mejor visualización
    # if sar_arrays:  #conc_dil hay datos
    #     all_values = np.concatenate(sar_arrays)
    #     y_min = np.min(all_values) * 0.98
    #     y_max = np.max(all_values) * 1.02
    #     ax.set_ylim(y_min, y_max)
axes[1].set_xlabel('Índice')

# Graficar cada categoría en su propio subplot
graficar_sar_vs_indice(axes[0], SAR_arrays_conc, 'conc', 'tab:blue', 0)
graficar_sar_vs_indice(axes[1], SAR_arrays_dil, 'dil', 'tab:red', 1)

plt.suptitle('Comparativa SAR vs Índice - 300 kHz - 57 kA/m', fontsize=14)
plt.savefig('comparativa_SAR_conc_dil.png', dpi=300)
plt.show()


#%% Gráfico de Hc vs índice (1 fila, 1 columna - ambas categorías juntas)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), constrained_layout=True)
colores = ['tab:blue', 'tab:red']
categorias = ['conc', 'dil']

# Graficar conc
for i, hc_array in enumerate(Hc_arrays_conc):
    indices = np.arange(len(hc_array))
    ax.plot(indices, hc_array, 's-', alpha=0.7, linewidth=1, markersize=4,
            label=f'conc - Muestra {i+1}', color='tab:blue')

# Graficar dil
for i, hc_array in enumerate(Hc_arrays_dil):
    indices = np.arange(len(hc_array))
    ax.plot(indices, hc_array, 'o-', alpha=0.7, linewidth=1, markersize=4,
            label=f'dil - Muestra {i+1}', color='tab:red')

ax.set_title('Comparativa Campo Coercitivo vs Índice', loc='left')
ax.set_xlabel('Índice')
ax.set_ylabel('Hc (kA/m)')
ax.legend(loc='upper right', ncol=2)
ax.grid(True, linestyle='--', alpha=0.7)

plt.savefig('comparativa_Hc_conc_dil_combinado.png', dpi=300)
plt.show()
#%% Comparo los ciclos promedio conc
_,_,_,H_conc_Am,M_conc_Am,_ = lector_ciclos(ciclos_conc[0])
_,_,_,H_dil_Am,M_dil_Am,_ = lector_ciclos(ciclos_dil[0])


fig, (a,b)= plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True,sharex=True,sharey=True)

a.set_title('Concentrado')
for i,c in enumerate(ciclos_conc):
    _,_,_,H_conc_Am,M_conc_Am,_ = lector_ciclos(c)
    a.plot(H_conc_Am/1000,M_conc_Am,label=i+1)

b.set_title('diluido')
for i,d in enumerate(ciclos_dil):
    _,_,_,H_dil_Am,M_dil_Am,_ = lector_ciclos(d)
    b.plot(H_dil_Am/1000,M_dil_Am,label=i+1)
# b.plot(H_dil_Am/1000,M_dil_Am,label='dil')



for i in (a,b):
    i.grid(True)
    i.legend()
    i.set_xlabel('H (kA/m)')
a.set_ylabel('M (A/m)')

plt.suptitle('Comparativa ciclos promedio\n300 kHz - 57 kA/m', fontsize=14)

plt.savefig('comparativa_ciclos_promedio_conc_dil.png', dpi=300)
plt.show()
#%% idem Normalizado a max valor
_,_,_,H_conc_Am,M_conc_Am,_ = lector_ciclos(ciclos_conc[0])
_,_,_,H_dil_Am,M_dil_Am,_ = lector_ciclos(ciclos_dil[0])

_,_,_,H_7A_Am,M_7A_Am,_ = lector_ciclos('data_7A_3Z/300kHz_150dA_100Mss_bobN17A00_ciclo_promedio_H_M.txt')
_,_,_,H_3Z_Am,M_3Z_Am,_ = lector_ciclos('data_7A_3Z/300kHz_150dA_100Mss_bobN13Z00_ciclo_promedio_H_M.txt')

fig, (a,b)= plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True,sharex=True,sharey=True)

a.set_title('7A vs conc (cristal grande)')
a.plot(H_7A_Am/1000,M_7A_Am/max(M_7A_Am),label='7A')
a.plot(H_conc_Am/1000,M_conc_Am/max(M_conc_Am),label='conc')

b.set_title('3Z vs dil (cristal chico)')
b.plot(H_3Z_Am/1000,M_3Z_Am/max(M_3Z_Am),label='3Z')
b.plot(H_dil_Am/1000,M_dil_Am/max(M_F2_Am),label='F2')



for i in (a,b):
    i.grid(True)
    i.legend()
    i.set_xlabel('H (kA/m)')
a.set_ylabel('M/M$_{max}$')

plt.suptitle('Comparativa ciclos promedio normalizados\n300 kHz - 57 kA/m', fontsize=14)

plt.savefig('comparativa_ciclos_promedio_NF_F2_normalizados.png', dpi=300)
plt.show()