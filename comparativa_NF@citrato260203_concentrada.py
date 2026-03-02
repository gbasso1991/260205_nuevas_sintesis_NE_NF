#%%
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
#%%
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
#%% Extraigo valores de las tablas de resultados
resultados_135 = glob("NF@citrato - solvotermal - concentrada/NF_135/**/**/Analisis_*/*resultados.txt")
resultados_135.sort()

tau_135_050,tau_135_075,tau_135_100,tau_135_125,tau_135_150 = [],[],[],[],[]
SAR_135_050,SAR_135_075,SAR_135_100,SAR_135_125,SAR_135_150 = [],[],[],[],[]
Hc_135_050,Hc_135_075,Hc_135_100,Hc_135_125,Hc_135_150 = [],[],[],[],[]
Mr_135_050,Mr_135_075,Mr_135_100,Mr_135_125,Mr_135_150 = [],[],[],[],[]
for f in resultados_135:
    if '050dA' in f:
        meta, _, _,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_135_050.append(meta['tau_ns'])
        SAR_135_050.append(meta['SAR_W/g'])
        Hc_135_050.append(meta['Hc_kA/m'])
        Mr_135_050.append(meta['Mr_A/m'])
    elif '075dA' in f:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_135_075.append(meta['tau_ns'])
        SAR_135_075.append(meta['SAR_W/g'])
        Hc_135_075.append(meta['Hc_kA/m'])
        Mr_135_075.append(meta['Mr_A/m'])
    elif '100dA' in f:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_135_100.append(meta['tau_ns'])
        SAR_135_100.append(meta['SAR_W/g'])
        Hc_135_100.append(meta['Hc_kA/m'])
        Mr_135_100.append(meta['Mr_A/m'])
    elif '125dA' in f:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_135_125.append(meta['tau_ns'])
        SAR_135_125.append(meta['SAR_W/g'])
        Hc_135_125.append(meta['Hc_kA/m'])
        Mr_135_125.append(meta['Mr_A/m'])
    elif '150dA' in f:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_135_150.append(meta['tau_ns'])
        SAR_135_150.append(meta['SAR_W/g'])
        Hc_135_150.append(meta['Hc_kA/m'])
        Mr_135_150.append(meta['Mr_A/m'])

#%% 212 kHz 
resultados_212 = glob("NF@citrato - solvotermal - concentrada/NF_212/**/**/Analisis_*/*resultados.txt")
resultados_212.sort()    

tau_212_050,tau_212_075,tau_212_100,tau_212_125,tau_212_150 = [],[],[],[],[]
SAR_212_050,SAR_212_075,SAR_212_100,SAR_212_125,SAR_212_150 = [],[],[],[],[]
Hc_212_050,Hc_212_075,Hc_212_100,Hc_212_125,Hc_212_150 = [],[],[],[],[]
Mr_212_050,Mr_212_075,Mr_212_100,Mr_212_125,Mr_212_150 = [],[],[],[],[]
for f in resultados_212:
    if '050dA' in f:
        meta, _, _,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_212_050.append(meta['tau_ns'])
        SAR_212_050.append(meta['SAR_W/g'])
        Hc_212_050.append(meta['Hc_kA/m'])
        Mr_212_050.append(meta['Mr_A/m'])
    elif '075dA' in f:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_212_075.append(meta['tau_ns'])
        SAR_212_075.append(meta['SAR_W/g'])
        Hc_212_075.append(meta['Hc_kA/m'])
        Mr_212_075.append(meta['Mr_A/m'])
    elif '100dA' in f:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_212_100.append(meta['tau_ns'])
        SAR_212_100.append(meta['SAR_W/g'])
        Hc_212_100.append(meta['Hc_kA/m'])
        Mr_212_100.append(meta['Mr_A/m'])
    elif '125dA' in f:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_212_125.append(meta['tau_ns'])
        SAR_212_125.append(meta['SAR_W/g'])
        Hc_212_125.append(meta['Hc_kA/m'])
        Mr_212_125.append(meta['Mr_A/m'])
    elif '150dA' in f:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_212_150.append(meta['tau_ns'])
        SAR_212_150.append(meta['SAR_W/g'])
        Hc_212_150.append(meta['Hc_kA/m'])
        Mr_212_150.append(meta['Mr_A/m'])

#%%300 kHz
resultados_300 = glob("NF@citrato - solvotermal - concentrada/NF_300/**/**/Analisis_*/*resultados.txt")
resultados_300.sort()

tau_300_050,tau_300_075,tau_300_100,tau_300_125,tau_300_150 = [],[],[],[],[]
SAR_300_050,SAR_300_075,SAR_300_100,SAR_300_125,SAR_300_150 = [],[],[],[],[]
Hc_300_050,Hc_300_075,Hc_300_100,Hc_300_125,Hc_300_150 = [],[],[],[],[]
Mr_300_050,Mr_300_075,Mr_300_100,Mr_300_125,Mr_300_150 = [],[],[],[],[]

for f in resultados_300:
    if '050dA' in f:
        meta, _, _,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_300_050.append(meta['tau_ns'])
        SAR_300_050.append(meta['SAR_W/g'])
        Hc_300_050.append(meta['Hc_kA/m'])
        Mr_300_050.append(meta['Mr_A/m'])
    elif '075dA' in f:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_300_075.append(meta['tau_ns'])
        SAR_300_075.append(meta['SAR_W/g'])
        Hc_300_075.append(meta['Hc_kA/m'])
        Mr_300_075.append(meta['Mr_A/m'])
    elif '100dA' in f:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_300_100.append(meta['tau_ns'])
        SAR_300_100.append(meta['SAR_W/g'])
        Hc_300_100.append(meta['Hc_kA/m'])
        Mr_300_100.append(meta['Mr_A/m'])
    elif '125dA' in f:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)        
        tau_300_125.append(meta['tau_ns'])
        SAR_300_125.append(meta['SAR_W/g'])
        Hc_300_125.append(meta['Hc_kA/m'])
        Mr_300_125.append(meta['Mr_A/m'])

    elif '150dA' in f:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(f)
        tau_300_150.append(meta['tau_ns'])
        SAR_300_150.append(meta['SAR_W/g'])
        Hc_300_150.append(meta['Hc_kA/m'])
        Mr_300_150.append(meta['Mr_A/m'])

#%% Promedio con test de chi**2 para verificar consistencia entre medidas
def promedio_ponderado_con_consistencia(array_ufloats):
    """
    Calcula el promedio ponderado por varianza inversa de mediciones con ufloat,
    verificando la consistencia mediante test de χ².
    
    Criterios de decisión:
    - χ²/ν ≤ 1.5: CONSISTENTE → usa incertidumbre instrumental (ponderada)
    - 1.5 < χ²/ν ≤ 2.0: LEVE INCONSISTENCIA → usa incertidumbre por dispersión observada
    - χ²/ν > 2.0: INCONSISTENTE → usa la mayor de ambas incertidumbres
    
    Parameters
    ----------
    array_ufloats : list of ufloat
        Lista de mediciones con sus incertidumbres.
        
    Returns
    -------
    ufloat
        Resultado con incertidumbre apropiada según consistencia.
    float
        χ² reducido (χ²/ν) calculado.
    
    Notes
    -----
    χ²/ν = Σ[(x_i - x_ponderado)/σ_i]² / (n-1)
    ν = n-1 grados de libertad (para n mediciones)
    """
    # Extraer valores e incertezas
    valores = unumpy.nominal_values(array_ufloats)
    incertezas = unumpy.std_devs(array_ufloats)
    
    # 1. Calcular promedio ponderado estándar
    pesos = 1.0 / (incertezas ** 2)
    valor_ponderado = np.sum(pesos * valores) / np.sum(pesos)
    incert_ponderada = 1.0 / np.sqrt(np.sum(pesos))
    
    # 2. Calcular dispersión entre medidas
    n = len(valores)
    if n > 1:
        dispersion_muestral = np.std(valores, ddof=1)
        incert_dispersion = dispersion_muestral / np.sqrt(n)
    else:
        incert_dispersion = incert_ponderada
    
    # 3. Test de χ² para verificar consistencia
    chi2 = np.sum(((valores - valor_ponderado) / incertezas) ** 2)
    grados_libertad = n - 1
    chi2_reducido = chi2 / grados_libertad if grados_libertad > 0 else 1
    
    # 4. Aplicar criterios de decisión CORREGIDOS
    print('-'*50)
    print(array_ufloats)
    print(f"  χ²/ν = {chi2_reducido:.2f} (ν={grados_libertad})")
    print(f"  Incert. instrumental: {incert_ponderada:.3f}")
    print(f"  Incert. dispersión:   {incert_dispersion:.3f}")
    
    if chi2_reducido > 2.0:  # INCONSISTENTE
        print(f"  ⚠️  INCONSISTENTE (χ²/ν > 2.0)")
        incert_final = max(incert_ponderada, incert_dispersion)
        
    elif chi2_reducido > 1.5:  # LEVE INCONSISTENCIA
        print(f"  ⚠️  Leve inconsistencia (1.5 < χ²/ν ≤ 2.0)")
        incert_final = incert_dispersion  # Usar dispersión observada
        
    else:  # CONSISTENTE (χ²/ν ≤ 1.5)
        print(f"  ✅  Consistente (χ²/ν ≤ 1.5)")
        incert_final = incert_ponderada  # Usar incertidumbre instrumental
    
    # Mostrar resultado formateado
    resultado = ufloat(valor_ponderado, incert_final)
    print(f"  Resultado: {resultado:.1uS}")

    
    return resultado
#%% Ejecuto y armo dframe
datos = {
    135: {
        20: {"tau": tau_135_050, "sar": SAR_135_050, "hc": Hc_135_050, "mr": Mr_135_050},
        29: {"tau": tau_135_075, "sar": SAR_135_075, "hc": Hc_135_075, "mr": Mr_135_075},
        38: {"tau": tau_135_100, "sar": SAR_135_100, "hc": Hc_135_100, "mr": Mr_135_100},
        47: {"tau": tau_135_125, "sar": SAR_135_125, "hc": Hc_135_125, "mr": Mr_135_125},
        57: {"tau": tau_135_150, "sar": SAR_135_150, "hc": Hc_135_150, "mr": Mr_135_150},
    },
    212: {
        20: {"tau": tau_212_050, "sar": SAR_212_050, "hc": Hc_212_050, "mr": Mr_212_050},
        29: {"tau": tau_212_075, "sar": SAR_212_075, "hc": Hc_212_075, "mr": Mr_212_075},
        38: {"tau": tau_212_100, "sar": SAR_212_100, "hc": Hc_212_100, "mr": Mr_212_100},
        47: {"tau": tau_212_125, "sar": SAR_212_125, "hc": Hc_212_125, "mr": Mr_212_125},
        57: {"tau": tau_212_150, "sar": SAR_212_150, "hc": Hc_212_150, "mr": Mr_212_150},
    },
    300: {
        20: {"tau": tau_300_050, "sar": SAR_300_050, "hc": Hc_300_050, "mr": Mr_300_050},
        29: {"tau": tau_300_075, "sar": SAR_300_075, "hc": Hc_300_075, "mr": Mr_300_075},
        38: {"tau": tau_300_100, "sar": SAR_300_100, "hc": Hc_300_100, "mr": Mr_300_100},
        47: {"tau": tau_300_125, "sar": SAR_300_125, "hc": Hc_300_125, "mr": Mr_300_125},
        57: {"tau": tau_300_150, "sar": SAR_300_150, "hc": Hc_300_150, "mr": Mr_300_150},
    }
}
filas = []

for frecuencia, campos in datos.items():
    for campo_kAm, magnitudes in campos.items():
        
        fila = {
            "frecuencia_kHz": frecuencia,
            "campo_kA/m": campo_kAm
        }
        
        for nombre_mag, lista_ufloats in magnitudes.items():
            
            if len(lista_ufloats) == 0:
                fila[nombre_mag] = None
                continue
            
            resultado = promedio_ponderado_con_consistencia(lista_ufloats)
            fila[nombre_mag] = resultado
        
        filas.append(fila)

df = pd.DataFrame(filas)
df.sort_values(["frecuencia_kHz", "campo_kA/m"], inplace=True)
df.reset_index(drop=True, inplace=True)

df

# %% Graficos
fig1, (ax,ax2) = plt.subplots(2,1,figsize=(10,5),constrained_layout=True)

for f, sub in df.groupby("frecuencia_kHz"):
    sub = sub.sort_values("campo_kA/m")
    
    ax.errorbar(sub["campo_kA/m"],sub["tau"].apply(lambda x: x.n),yerr=sub["tau"].apply(lambda x: x.s),
        fmt='o-', capsize=4,
        label=f"{f}")
    ax2.errorbar(sub["campo_kA/m"],sub["sar"].apply(lambda x: x.n),yerr=sub["sar"].apply(lambda x: x.s),
        fmt='s-', capsize=4,
        label=f"{f}")

ax.set_title('tau',loc='left')
ax.set_ylabel("τ (ns)")
ax2.set_title('SAR',loc='left')
ax2.set_ylabel("SAR (W/g)")
ax2.set_xlabel("H$_0$ (kA/m)")
    
for a in [ax,ax2]:
    a.set_xticks(sub["campo_kA/m"])
    a.grid()
    a.legend(title="Frecuencia (kHz)",ncol=2)
plt.suptitle('NF@citrato_conc 260203 - 12.6 g/L Fe$_3$O$_4$')    
plt.show()

fig2 , (ax,ax2) = plt.subplots(2,1,figsize=(10,5),constrained_layout=True)

for f, sub in df.groupby("frecuencia_kHz"):
    sub = sub.sort_values("campo_kA/m")

    ax.errorbar(sub["campo_kA/m"],sub["hc"].apply(lambda x: x.n),yerr=sub["hc"].apply(lambda x: x.s),
        fmt='o-', capsize=4,
        label=f"{f}")
    ax2.errorbar(sub["campo_kA/m"],sub["mr"].apply(lambda x: x.n),yerr=sub["sar"].apply(lambda x: x.s),
        fmt='s-', capsize=4,
        label=f"{f}")

    
ax.set_ylabel("Coercitivo (kA/m)")
ax.set_title('Hc',loc='left')
ax2.set_title('Mr',loc='left')
ax2.set_ylabel("Remanencia (A/m)")
ax2.set_xlabel("H$_0$ (kA/m)")

for a in [ax,ax2]:
    a.set_xticks(sub["campo_kA/m"])
    a.grid()
    a.legend(title="Frecuencia (kHz)",ncol=2)

plt.suptitle('NF@citrato_conc 260203 - 12.6 g/L Fe$_3$O$_4$')
plt.show()

# Ahora agrupando por campo y graficando vs frecuencia
fig3, (ax,ax2) = plt.subplots(2,1,figsize=(10,5),constrained_layout=True)

for f, sub in df.groupby("campo_kA/m"):   
    sub = sub.sort_values("frecuencia_kHz")

    ax.errorbar(sub["frecuencia_kHz"],sub["tau"].apply(lambda x: x.n),yerr=sub["tau"].apply(lambda x: x.s),
        fmt='o-', capsize=4,
        label=f"{f}")
    ax2.errorbar(sub["frecuencia_kHz"],sub["sar"].apply(lambda x: x.n),yerr=sub["sar"].apply(lambda x: x.s),
        fmt='s-', capsize=4,
        label=f"{f}")

ax.set_ylabel("τ (ns)")
ax.set_title('tau',loc='left')
ax2.set_title('SAR',loc='left')        
ax2.set_ylabel("SAR (W/g)")
ax2.set_xlabel("Frecuencia (kHz)")

for a in [ax,ax2]:
    a.set_xticks(sub["frecuencia_kHz"])
    a.grid()
    a.legend(title="Frecuencia (kHz)",ncol=3)
plt.suptitle('NF@citrato_conc 260203 - 12.6 g/L Fe$_3$O$_4$')    
plt.show()

#lo mismo para Hc y Mr
fig4 , (ax,ax2) = plt.subplots(2,1,figsize=(10,5),constrained_layout=True)

for f, sub in df.groupby("campo_kA/m"):    
    sub = sub.sort_values("frecuencia_kHz")

    ax.errorbar(sub["frecuencia_kHz"],sub["hc"].apply(lambda x: x.n),yerr=sub["hc"].apply(lambda x: x.s),
        fmt='o-', capsize=4,
        label=f"{f}")
    ax2.errorbar(sub["frecuencia_kHz"],sub["mr"].apply(lambda x: x.n),yerr=sub["sar"].apply(lambda x: x.s),
        fmt='s-', capsize=4,
        label=f"{f}")

ax.set_ylabel("Coercitivo (kA/m)")
ax.set_title('Hc',loc='left')
ax2.set_title('Mr',loc='left')
ax2.set_ylabel("Remanencia (A/m)")
ax2.set_xlabel("Frecuencia (kHz)")

for a in [ax,ax2]:
    a.set_xticks(sub["frecuencia_kHz"])
    a.grid()
    a.legend(title="Frecuencia (kHz)",ncol=3)

plt.suptitle('NF@citrato_conc 260203 - 12.6 g/L Fe$_3$O$_4$')
plt.show()

#%% Guardo figuras
for name,figura in zip(['tau_SAR_vs_H','Hc_Mr_vs_H','tau_SAR_vs_frec','Hc_Mr_vs_frec'],[fig1,fig2,fig3,fig4]):
    figura.savefig(f'NF@citrato - solvotermal - concentrada/{name}.png',dpi=300)
#%% Ploteo los ciclos promedio

ciclos_135 = glob("NF@citrato - solvotermal - concentrada/NF_135/**/**/Analisis_*/*ciclo_promedio_H_M.txt")
ciclos_135.sort()

ciclos_135_050,ciclos_135_075,ciclos_135_100,ciclos_135_125,ciclos_135_150 = [],[],[],[],[]

for c in ciclos_135:
    if '050dA' in c:
        pass
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_135_050.append((H_kAm,M_Am,metadata))
    elif '075dA' in c:
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_135_075.append((H_kAm,M_Am,metadata))
    elif '100dA' in c:
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_135_100.append((H_kAm,M_Am,metadata))
    elif '125dA' in c:
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_135_125.append((H_kAm,M_Am,metadata))
    elif '150dA' in c:
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_135_150.append((H_kAm,M_Am,metadata))

ciclos_212 = glob("NF@citrato - solvotermal - concentrada/NF_212/**/**/Analisis_*/*ciclo_promedio_H_M.txt")
ciclos_212.sort()

ciclos_212_050,ciclos_212_075,ciclos_212_100,ciclos_212_125,ciclos_212_150 = [],[],[],[],[]

for c in ciclos_212:
    if '050dA' in c:
        pass
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_212_050.append((H_kAm,M_Am,metadata))
    elif '075dA' in c:
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_212_075.append((H_kAm,M_Am,metadata))
    elif '100dA' in c:
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_212_100.append((H_kAm,M_Am,metadata))
    elif '125dA' in c:
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_212_125.append((H_kAm,M_Am,metadata))
    elif '150dA' in c:
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_212_150.append((H_kAm,M_Am,metadata))

ciclos_300 = glob("NF@citrato - solvotermal - concentrada/NF_300/**/**/Analisis_*/*ciclo_promedio_H_M.txt")
ciclos_300.sort() 
ciclos_300_050,ciclos_300_075,ciclos_300_100,ciclos_300_125,ciclos_300_150 = [],[],[],[],[]

for c in ciclos_300:
    if '050dA' in c:
        pass
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_300_050.append((H_kAm,M_Am,metadata))
    elif '075dA' in c:
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_300_075.append((H_kAm,M_Am,metadata))
    elif '100dA' in c:
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_300_100.append((H_kAm,M_Am,metadata))
    elif '125dA' in c:
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_300_125.append((H_kAm,M_Am,metadata))
    elif '150dA' in c:
        _,_,_,H_kAm,M_Am,metadata = lector_ciclos(c)
        ciclos_300_150.append((H_kAm,M_Am,metadata))

#%% Ploteo ciclos promedio para 135 kHz
fig0, (ax,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(6,18),constrained_layout=True,sharex=True,sharey=True)
for ciclo in ciclos_135_050:
    H_kAm, M_Am, meta = ciclo
    ax.plot(H_kAm/1000,M_Am,label="19 kA/m")

for ciclo in ciclos_135_075:
    H_kAm, M_Am, meta = ciclo
    ax2.plot(H_kAm/1000,M_Am,label="29")

for ciclo in ciclos_135_100:
    H_kAm, M_Am, meta = ciclo
    ax3.plot(H_kAm/1000,M_Am,label="38")

for ciclo in ciclos_135_125:
    H_kAm, M_Am, meta = ciclo
    ax4.plot(H_kAm/1000,M_Am,label="47")

for ciclo in ciclos_135_150:    
    H_kAm, M_Am, meta = ciclo
    ax5.plot(H_kAm/1000,M_Am,label="57")

ax5.set_xlabel("H (kA/m)")
ax.legend(title="H$_0$ (kA/m)")

for a in [ax,ax2,ax3,ax4,ax5]:
    a.set_ylabel("M (A/m)")
    a.grid()
    a.legend(title="H$_0$ (kA/m)",ncol=1)
plt.suptitle('135 kHz - NF@citrato solvotermal - concentrada - 12.6 g/L Fe$_3$O$_4$')
plt.show()
#%% Ploteo ciclos promedio para 212 kHz
for ciclo in ciclos_212_050:
    H_kAm, M_Am, meta = ciclo
    ax.plot(H_kAm/1000,M_Am,label="19 kA/m")

for ciclo in ciclos_212_075:
    H_kAm, M_Am, meta = ciclo
    ax2.plot(H_kAm/1000,M_Am,label="29")

for ciclo in ciclos_212_100:
    H_kAm, M_Am, meta = ciclo
    ax3.plot(H_kAm/1000,M_Am,label="38")

for ciclo in ciclos_212_125:
    H_kAm, M_Am, meta = ciclo
    ax4.plot(H_kAm/1000,M_Am,label="47")

for ciclo in ciclos_212_150:    
    H_kAm, M_Am, meta = ciclo
    ax5.plot(H_kAm/1000,M_Am,label="57")

ax5.set_xlabel("H (kA/m)")
ax.legend(title="H$_0$ (kA/m)")

for a in [ax,ax2,ax3,ax4,ax5]:
    a.set_ylabel("M (A/m)")
    a.grid()
    a.legend(title="H$_0$ (kA/m)",ncol=1)
plt.suptitle('212 kHz - NF@citrato solvotermal - concentrada - 12.6 g/L Fe$_3$O$_4$')
plt.show()
#%% Ploteo ciclos promedio para 300 kHz
fig2, (ax,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(6,18),constrained_layout=True,sharex=True,sharey=True)
for ciclo in ciclos_300_050:
    H_kAm, M_Am, meta = ciclo
    ax.plot(H_kAm/1000,M_Am,label="19 kA/m")

for ciclo in ciclos_300_075:
    H_kAm, M_Am, meta = ciclo
    ax2.plot(H_kAm/1000,M_Am,label="29")

for ciclo in ciclos_300_100:
    H_kAm, M_Am, meta = ciclo
    ax3.plot(H_kAm/1000,M_Am,label="38")

for ciclo in ciclos_300_125:
    H_kAm, M_Am, meta = ciclo
    ax4.plot(H_kAm/1000,M_Am,label="47")

for ciclo in ciclos_300_150:    
    H_kAm, M_Am, meta = ciclo
    ax5.plot(H_kAm/1000,M_Am,label="57")

ax5.set_xlabel("H (kA/m)")
ax.legend(title="H$_0$ (kA/m)")

for a in [ax,ax2,ax3,ax4,ax5]:
    a.set_ylabel("M (A/m)")
    a.grid()
    a.legend(title="H$_0$ (kA/m)",ncol=1)
plt.suptitle('300 kHz - NF@citrato solvotermal - concentrada - 12.6 g/L Fe$_3$O$_4$')
plt.show()  


#%% Comparo 1 ciclo representativo para cada campo
#135 kHz
fig3,ax =plt.subplots(figsize=(6,5),constrained_layout=True)

ax.plot(ciclos_135_050[0][0]/1000,ciclos_135_050[0][1],label="19",zorder=5)
ax.plot(ciclos_135_075[0][0]/1000,ciclos_135_075[0][1],label="29",zorder=4)
ax.plot(ciclos_135_100[0][0]/1000,ciclos_135_100[0][1],label="38",zorder=3)
ax.plot(ciclos_135_125[0][0]/1000,ciclos_135_125[0][1],label="47",zorder=2)
ax.plot(ciclos_135_150[1][0]/1000,ciclos_135_150[1][1],label="57",zorder=1)

ax.set_ylabel("M (A/m)")
ax.grid(zorder=0)
ax.legend(title="H$_0$ (kA/m)",ncol=1)
plt.suptitle('135 kHz - NF@citrato solvotermal - concentrada - 12.6 g/L Fe$_3$O$_4$')
plt.show()
#%%
#212 kHz
fig4,ax =plt.subplots(figsize=(6,5),constrained_layout=True)
ax.plot(ciclos_212_050[0][0]/1000,ciclos_212_050[0][1],label="19",zorder=5)
ax.plot(ciclos_212_075[0][0]/1000,ciclos_212_075[0][1],label="29",zorder=4)
ax.plot(ciclos_212_100[0][0]/1000,ciclos_212_100[0][1],label="38",zorder=3)
ax.plot(ciclos_212_125[0][0]/1000,ciclos_212_125[0][1],label="47",zorder=2)
ax.plot(ciclos_212_150[1][0]/1000,ciclos_212_150[1][1],label="57",zorder=1)

ax.set_ylabel("M (A/m)")
ax.grid(zorder=0)
ax.legend(title="H$_0$ (kA/m)",ncol=1)
plt.suptitle('212 kHz - NF@citrato solvotermal - concentrada - 12.6 g/L Fe$_3$O$_4$')
plt.show()
# %%
# 300 kHz
fig5,ax =plt.subplots(figsize=(6,5),constrained_layout=True)
ax.plot(ciclos_300_050[1][0]/1000,ciclos_300_050[1][1],label="19",zorder=5)
ax.plot(ciclos_300_075[0][0]/1000,ciclos_300_075[0][1],label="29",zorder=4)
ax.plot(ciclos_300_100[1][0]/1000,ciclos_300_100[1][1],label="38",zorder=3)
ax.plot(ciclos_300_125[0][0]/1000,ciclos_300_125[0][1],label="47",zorder=2)
ax.plot(ciclos_300_150[1][0]/1000,ciclos_300_150[1][1],label="57",zorder=1)

ax.set_ylabel("M (A/m)")
ax.grid(zorder=0)
ax.legend(title="H$_0$ (kA/m)",ncol=1)
plt.suptitle('300 kHz - NF@citrato solvotermal - concentrada - 12.6 g/L Fe$_3$O$_4$')
plt.show()
# %% Comparo ciclos representativos para cada campo
#20
fig6,ax =plt.subplots(figsize=(6,5),constrained_layout=True)
ax.plot(ciclos_135_050[0][0]/1000,ciclos_135_050[0][1],label="135 kHz",zorder=3)
ax.plot(ciclos_212_050[0][0]/1000,ciclos_212_050[0][1],label="212 kHz",zorder=2)
ax.plot(ciclos_300_050[1][0]/1000,ciclos_300_050[1][1],label="300 kHz",zorder=1)    

ax.set_ylabel("M (A/m)")
ax.grid(zorder=0)
ax.legend(title="$f$ (kHz)",ncol=1)
ax.set_title("20 kA/m - NF@citrato solvotermal - concentrada - 12.6 g/L Fe$_3$O$_4$")

plt.show()
#%%29
fig7,ax =plt.subplots(figsize=(6,5),constrained_layout=True)
ax.plot(ciclos_135_075[0][0]/1000,ciclos_135_075[0][1],label="135 kHz",zorder=3)
ax.plot(ciclos_212_075[0][0]/1000,ciclos_212_075[0][1],label="212 kHz",zorder=2)
ax.plot(ciclos_300_075[1][0]/1000,ciclos_300_075[1][1],label="300 kHz",zorder=1)    

ax.set_ylabel("M (A/m)")
ax.grid(zorder=0)
ax.legend(title="$f$ (kHz)",ncol=1)
ax.set_title("29 kA/m - NF@citrato solvotermal - concentrada - 12.6 g/L Fe$_3$O$_4$")

plt.show()

#%%38
fig8,ax =plt.subplots(figsize=(6,5),constrained_layout=True)
ax.plot(ciclos_135_100[0][0]/1000,ciclos_135_100[0][1],label="135 kHz",zorder=3)
ax.plot(ciclos_212_100[0][0]/1000,ciclos_212_100[0][1],label="212 kHz",zorder=2)
ax.plot(ciclos_300_100[1][0]/1000,ciclos_300_100[1][1],label="300 kHz",zorder=1)    

ax.set_ylabel("M (A/m)")
ax.grid(zorder=0)
ax.legend(title="$f$ (kHz)",ncol=1)
ax.set_title("38 kA/m - NF@citrato solvotermal - concentrada - 12.6 g/L Fe$_3$O$_4$")

plt.show()
#%% 47
fig9,ax =plt.subplots(figsize=(6,5),constrained_layout=True)
ax.plot(ciclos_135_125[0][0]/1000,ciclos_135_125[0][1],label="135 kHz",zorder=3)
ax.plot(ciclos_212_125[0][0]/1000,ciclos_212_125[0][1],label="212 kHz",zorder=2)
ax.plot(ciclos_300_125[0][0]/1000,ciclos_300_125[0][1],label="300 kHz",zorder=1)    

ax.set_ylabel("M (A/m)")
ax.grid(zorder=0)
ax.legend(title="$f$ (kHz)",ncol=1)
ax.set_title("47 kA/m - NF@citrato solvotermal - concentrada - 12.6 g/L Fe$_3$O$_4$")

plt.show()  

#%%57
fig10,ax =plt.subplots(figsize=(6,5),constrained_layout=True)
ax.plot(ciclos_135_150[1][0]/1000,ciclos_135_150[1][1],label="135 kHz",zorder=3)
ax.plot(ciclos_212_150[1][0]/1000,ciclos_212_150[1][1],label="212 kHz",zorder=2)    
ax.plot(ciclos_300_150[0][0]/1000,ciclos_300_150[0][1],label="300 kHz",zorder=1)

ax.set_ylabel("M (A/m)")
ax.grid(zorder=0)
ax.legend(title="$f$ (kHz)",ncol=1)
ax.set_title("57 kA/m - NF@citrato solvotermal - concentrada - 12.6 g/L Fe$_3$O$_4$")

plt.show()
# %% guardo figuras
for name,figura in zip(['ciclos_135kHz_all',
                        'ciclos_212kHz_all',
                        'ciclos_300kHz_all',
                        'ciclos_135kHz_comparativo',
                        'ciclos_212kHz_comparativo',
                        'ciclos_300kHz_comparativo',
                        'ciclos_20kAm',
                        'ciclos_29kAm',
                        'ciclos_38kAm',
                        'ciclos_47kAm',
                        'ciclos_57kAm'],
                       [fig0,
                        fig1,
                        fig2,
                        fig3,
                        fig4,
                        fig5,
                        fig6,
                        fig7,
                        fig8,
                        fig9,
                        fig10]):
    figura.savefig(f'NF@citrato - solvotermal - concentrada/{name}.png',dpi=300)
#%%
