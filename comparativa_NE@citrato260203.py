#%%
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd

#%% 135 kHz

tau_135_050 = [ufloat(157,4), ufloat(153,8), ufloat(154,6)]
tau_135_075 = [ufloat(129,3), ufloat(139,4), ufloat(129,4)]
tau_135_100 = [ufloat(111,15), ufloat(154,39), ufloat(106,11)]
tau_135_125 = [ufloat(68,9), ufloat(87,26), ufloat(93,11)]    
tau_135_150 = [ufloat(39,13), ufloat(70,14), ufloat(36,18)]

SAR_135_050 = [ufloat(32,1), ufloat(31,2), ufloat(31,1)]
SAR_135_075 = [ufloat(49,1), ufloat(52,1), ufloat(49,1)]
SAR_135_100 = [ufloat(63,8), ufloat(86,21), ufloat(60,6)]
SAR_135_125 = [ufloat(51,7), ufloat(69,20), ufloat(73,9)]
SAR_135_150 = [ufloat(39,13), ufloat(75,14), ufloat(36,18)]

#%% 212 kHz 
tau_212_050 = [ufloat(104, 1), ufloat(106, 1), ufloat(105, 2)]
tau_212_075 = [ufloat(81, 4), ufloat(83, 4), ufloat(78, 6)]
tau_212_100 = [ufloat(63, 3), ufloat(60, 3), ufloat(63, 2)]
tau_212_125 = [ufloat(48, 2), ufloat(51, 2), ufloat(49, 2)]
tau_212_150 = [ufloat(36, 2), ufloat(36, 2), ufloat(36, 2)]

SAR_212_050 = [ufloat(51, 1), ufloat(52, 1), ufloat(53, 1)]
SAR_212_075 = [ufloat(149, 7), ufloat(155, 7), ufloat(142, 11)]
SAR_212_100 = [ufloat(133, 5), ufloat(131, 7), ufloat(138, 5)]
SAR_212_125 = [ufloat(118, 6), ufloat(120, 4), ufloat(115, 5)]
SAR_212_150 = [ufloat(99, 5), ufloat(107, 5), ufloat(94, 5)]

#%% 300 kHz
tau_300_050 = [ufloat(78, 2), ufloat(79, 3), ufloat(73, 4)]
tau_300_075 = [ufloat(63, 1), ufloat(66, 1), ufloat(64, 2)]
tau_300_100 = [ufloat(69, 1), ufloat(47, 1), ufloat(47, 1)]
tau_300_125 = [ufloat(33, 1), ufloat(40, 1), ufloat(35, 1)]
tau_300_150 = [ufloat(32, 1), ufloat(33, 1), ufloat(32, 1), ufloat(30, 1)]

SAR_300_050 = [ufloat(82, 2), ufloat(82, 3), ufloat(77, 4)]
SAR_300_075 = [ufloat(250, 5), ufloat(244, 5), ufloat(218, 6)]
SAR_300_100 = [ ufloat(220, 4), ufloat(216, 3)]
SAR_300_125 = [ufloat(175, 5), ufloat(238, 5), ufloat(191, 5)]
SAR_300_150 = [ufloat(225, 7), ufloat(259, 5), ufloat(243, 7), ufloat(182, 5)]

#%% 

def promedio_ponderado(array_ufloats):
    """
    Calcula el promedio ponderado por varianza inversa para un array de ufloats.
    
    Args:
        array_ufloats: Lista de objetos ufloat (ej: [ufloat(104, 1), ufloat(106, 1)])
    
    Returns:
        ufloat: Promedio ponderado con su incertidumbre
    """
    # Extraer valores y desviaciones estándar
    valores = unumpy.nominal_values(array_ufloats)
    incertezas = unumpy.std_devs(array_ufloats)
    
    # Calcular pesos (inverso de la varianza)
    pesos = 1.0 / (incertezas ** 2)
    
    # Promedio ponderado
    valor_ponderado = np.sum(pesos * valores) / np.sum(pesos)
    
    # Incertidumbre del promedio ponderado
    incertidumbre_ponderada = 1.0 / np.sqrt(np.sum(pesos))
    
    return ufloat(valor_ponderado, incertidumbre_ponderada)

#%% Promedios ponderados
promedios_tau_135 = {
    '050': promedio_ponderado(tau_135_050),
    '075': promedio_ponderado(tau_135_075),
    '100': promedio_ponderado(tau_135_100),
    '125': promedio_ponderado(tau_135_125),
    '150': promedio_ponderado(tau_135_150)
}
promedios_SAR_135 = {
    '050': promedio_ponderado(SAR_135_050),
    '075': promedio_ponderado(SAR_135_075),
    '100': promedio_ponderado(SAR_135_100),     
    '125': promedio_ponderado(SAR_135_125),
    '150': promedio_ponderado(SAR_135_150)
}

promedios_tau_212 = {
    '050': promedio_ponderado(tau_212_050),
    '075': promedio_ponderado(tau_212_075),
    '100': promedio_ponderado(tau_212_100),
    '125': promedio_ponderado(tau_212_125),
    '150': promedio_ponderado(tau_212_150)
}
promedios_SAR_212 = {
    '050': promedio_ponderado(SAR_212_050),
    '075': promedio_ponderado(SAR_212_075),
    '100': promedio_ponderado(SAR_212_100),
    '125': promedio_ponderado(SAR_212_125),
    '150': promedio_ponderado(SAR_212_150)
}

promedios_tau_300 = {
    '050': promedio_ponderado(tau_300_050),
    '075': promedio_ponderado(tau_300_075),
    '100': promedio_ponderado(tau_300_100),
    '125': promedio_ponderado(tau_300_125),
    '150': promedio_ponderado(tau_300_150)      
}   
promedios_SAR_300 = {
    '050': promedio_ponderado(SAR_300_050),
    '075': promedio_ponderado(SAR_300_075),
    '100': promedio_ponderado(SAR_300_100),     
    '125': promedio_ponderado(SAR_300_125),     
    '150': promedio_ponderado(SAR_300_150)
}       
#%%
import numpy as np
from uncertainties import ufloat, unumpy

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

    
    return resultado, chi2_reducido

#%% Ejecuto los promediados y las propagaciones de incertidumbres

promedios_cc_tau_135 = {
    '050': promedio_ponderado_con_consistencia(tau_135_050),
    '075': promedio_ponderado_con_consistencia(tau_135_075),
    '100': promedio_ponderado_con_consistencia(tau_135_100),
    '125': promedio_ponderado_con_consistencia(tau_135_125),
    '150': promedio_ponderado_con_consistencia(tau_135_150)}

promedios_cc_SAR_135 = {
    '050': promedio_ponderado_con_consistencia(SAR_135_050),
    '075': promedio_ponderado_con_consistencia(SAR_135_075),
    '100': promedio_ponderado_con_consistencia(SAR_135_100),     
    '125': promedio_ponderado_con_consistencia(SAR_135_125),
    '150': promedio_ponderado_con_consistencia(SAR_135_150)}

promedios_cc_tau_212 = {
    '050': promedio_ponderado_con_consistencia(tau_212_050),
    '075': promedio_ponderado_con_consistencia(tau_212_075),
    '100': promedio_ponderado_con_consistencia(tau_212_100),
    '125': promedio_ponderado_con_consistencia(tau_212_125),
    '150': promedio_ponderado_con_consistencia(tau_212_150)}
promedios_cc_SAR_212 = {
    '050': promedio_ponderado_con_consistencia(SAR_212_050),
    '075': promedio_ponderado_con_consistencia(SAR_212_075),
    '100': promedio_ponderado_con_consistencia(SAR_212_100),
    '125': promedio_ponderado_con_consistencia(SAR_212_125),
    '150': promedio_ponderado_con_consistencia(SAR_212_150)}

promedios_cc_tau_300 = {
    '050': promedio_ponderado_con_consistencia(tau_300_050),
    '075': promedio_ponderado_con_consistencia(tau_300_075),
    '100': promedio_ponderado_con_consistencia(tau_300_100),
    '125': promedio_ponderado_con_consistencia(tau_300_125),
    '150': promedio_ponderado_con_consistencia(tau_300_150)      }   
promedios_cc_SAR_300 = {
    '050': promedio_ponderado_con_consistencia(SAR_300_050),
    '075': promedio_ponderado_con_consistencia(SAR_300_075),
    '100': promedio_ponderado_con_consistencia(SAR_300_100),     
    '125': promedio_ponderado_con_consistencia(SAR_300_125),     
    '150': promedio_ponderado_con_consistencia(SAR_300_150)}       

# %% PRINT
print("Promedios ponderados de tau para 135 kHz:")
for campo, valor in promedios_cc_tau_135.items():
    print(f"Campo {campo}: {valor[0]:.1uS} ns")

print("\nPromedios ponderados de tau para 212 kHz:")
for campo, valor in promedios_cc_tau_212.items():
    print(f"Campo {campo}: {valor[0]:.1uS} ns") 

print("\nPromedios ponderados de tau para 300 kHz:")
for campo, valor in promedios_cc_tau_300.items():
    print(f"Campo {campo}: {valor[0]:.1uS} ns") 

print('-'*50)

print("\nPromedios ponderados de SAR para 135 kHz:")    
for campo, valor in promedios_cc_SAR_135.items():
    print(f"Campo {campo}: {valor[0]:.1uS} W/g")    

print("\nPromedios ponderados de SAR para 212 kHz:")    
for campo, valor in promedios_cc_SAR_212.items():
    print(f"Campo {campo}: {valor[0]:.1uS} W/g") 

print("\nPromedios ponderados de SAR para 300 kHz:")
for campo, valor in promedios_cc_SAR_300.items():
    print(f"Campo {campo}: {valor[0]:.1uS} W/g")       
#%%
# Crear el DataFrame para plotear
H0_valores = [20, 29, 38, 47, 57]
H0_nombres = ['050', '075', '100', '125', '150']
frecuencias = [135, 212, 300]

# Lista para almacenar filas
filas = []

# Para cada frecuencia y H0
for f in frecuencias:
    for h0_nombre, h0_valor in zip(H0_nombres, H0_valores):
        # Obtener los resultados de tau y SAR para esta combinación
        dicc_tau = globals().get(f'promedios_cc_tau_{f}', {})
        dicc_sar = globals().get(f'promedios_cc_SAR_{f}', {})
        
        if h0_nombre in dicc_tau and h0_nombre in dicc_sar:
            tau_result, tau_chi2 = dicc_tau[h0_nombre]
            sar_result, sar_chi2 = dicc_sar[h0_nombre]
            
            filas.append({
                'frecuencia_kHz': f,
                'H0_kA_m': h0_valor,
                'H0_nombre': h0_nombre,
                'tau': tau_result,
                'SAR': sar_result,
                'tau_valor': tau_result.n,
                'tau_error': tau_result.s,
                'SAR_valor': sar_result.n,
                'SAR_error': sar_result.s,
                'tau_chi2': tau_chi2,
                'SAR_chi2': sar_chi2
            })

# Crear DataFrame
df = pd.DataFrame(filas)

# Ordenar
df = df.sort_values(['frecuencia_kHz', 'H0_kA_m'])

# Mostrar
print("DataFrame completo (15 filas):")
print(df[['frecuencia_kHz', 'H0_kA_m', 'tau_valor', 'tau_error', 'SAR_valor', 'SAR_error']].to_string(index=False))

# Función para extraer datos de gráficos (la que ya tenías)
def get_datos_grafico(df, frecuencia, variable='tau'):
    """Extrae datos para un gráfico específico"""
    mask = df['frecuencia_kHz'] == frecuencia
    datos = df[mask]
    return {
        'H0': datos['H0_kA_m'].values,
        'valores': datos[f'{variable}_valor'].values,
        'errores': datos[f'{variable}_error'].values
    }

# %% Graficos
# tau vs H0 para cada frecuencia
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

for idx, f in enumerate(frecuencias):
    datos = get_datos_grafico(df, f, 'tau')
    
    ax.errorbar(datos['H0'], datos['valores'], yerr=datos['errores'], 
                fmt='o-', label=f'{f}', capsize=4)

ax.set_xlabel('H$_0$ (kA/m)')
ax.set_ylabel('τ (ns)')
ax.set_xticks(H0_valores)
ax.grid()
ax.legend(title='$f$ (kHz)')
ax.set_title(' τ vs Amplitud de campo H$_0$\nNE@citrato 260203 - 9.0 g/L Fe$_3$O$_4$')
plt.savefig('taus_vs_H0_NE_citrato_260203.png', dpi=300)
plt.show()

# sar vs H0 para cada frecuencia
fig2, ax2 = plt.subplots(figsize=(10, 5), constrained_layout=True)

for idx, f in enumerate(frecuencias):
    datos = get_datos_grafico(df, f, 'SAR')
    
    ax2.errorbar(datos['H0'], datos['valores'], yerr=datos['errores'], 
                 fmt='s-', label=f'{f}', capsize=4)
ax2.set_xticks(H0_valores)
ax2.set_xlabel('H$_0$ (kA/m)')
ax2.set_ylabel('SAR (W/g)')
ax2.grid()
ax2.legend(title='$f$ (kHz)')
ax2.set_title('SAR vs Amplitud de campo H$_0$\nNE@citrato 260203 - 9.0 g/L Fe$_3$O$_4$')
plt.savefig('SAR_vs_H0_NE_citrato_260203.png', dpi=300)
plt.show()

#%% graficos tau vs f para cada H0
fig3, axes3 = plt.subplots(figsize=(10, 5), constrained_layout=True)
for idx, h0 in enumerate(H0_valores):
    ax = axes3
    mask = df['H0_kA_m'] == h0
    datos_h0 = df[mask].sort_values('frecuencia_kHz')
    
    ax.errorbar(datos_h0['frecuencia_kHz'], datos_h0['tau_valor'], 
                 yerr=datos_h0['tau_error'], fmt='o-', capsize=5, label=f'{h0}')
ax.set_xticks(frecuencias)
ax.set_xlabel('f (kHz)')
ax.set_ylabel('τ (ns)') 
ax.grid()
ax.legend(title='H$_0$ (kA/m)')
ax.set_title('τ vs f\nNE@citrato 260203 - 9.0 g/L Fe$_3$O$_4$')
plt.savefig('tau_vs_f_NE_citrato_260203.png', dpi=300)
plt.show()

fig4, axes4 = plt.subplots(figsize=(10, 5), constrained_layout=True)
for idx, h0 in enumerate(H0_valores):
    ax = axes4
    mask = df['H0_kA_m'] == h0
    datos_h0 = df[mask].sort_values('frecuencia_kHz')
    
    ax.errorbar(datos_h0['frecuencia_kHz'], datos_h0['SAR_valor'], 
                 yerr=datos_h0['SAR_error'], fmt='s-', capsize=5, label=f'{h0}')
ax.set_xticks(frecuencias)
ax.set_xlabel('f (kHz)')
ax.set_ylabel('SAR (W/g)')
ax.grid()
ax.legend(title='H$_0$ (kA/m)')
ax.set_title('SAR vs f\nNE@citrato 260203 - 9.0 g/L Fe$_3$O$_4$')
plt.savefig('SAR_vs_f_NE_citrato_260203.png', dpi=300)
plt.show()

# Gráficos de SAR vs H0 para cada frecuencia
# for idx, f in enumerate(frecuencias):
#     ax = axes[1, idx]
#     datos = get_datos_grafico(df, f, 'SAR')
    
#     ax.errorbar(datos['H0'], datos['valores'], yerr=datos['errores'],
#                 fmt='s-', capsize=5, capthick=2, linewidth=2, color='red')
#     ax.set_title(f'SAR vs H0 - {f} kHz')
#     ax.set_xlabel('H0 (kA/m)')
#     ax.set_ylabel('SAR (unidades)')
#     ax.grid(True, alpha=0.3)




# %%
