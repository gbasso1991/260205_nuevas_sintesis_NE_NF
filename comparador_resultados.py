#%% 
import os
from glob import glob
from clase_resultados import ResultadosESAR
from uncertainties import ufloat
import matplotlib.pyplot as plt
#%% 1 - NE@citrico
subdirectorios=os.listdir(os.path.join(os.getcwd(),"NE@citrato - coprecipitacion/NE_135"))
subdirectorios.sort()
print(subdirectorios)

#%%
for sd in subdirectorios:
    print(sd)
    directorio_a_analizar = os.path.join(os.getcwd(), "NE@citrato - coprecipitacion/NE_135", sd)

    patron_analisis = os.path.join(directorio_a_analizar, "Analisis_*")
    directorios_analisis = glob(patron_analisis)

    if not directorios_analisis:
        print(f"No se encontraron directorios 'Analisis_' en {directorio_a_analizar}")
        exit()

    directorio_analisis = directorios_analisis[-1]

    try:
        resultados = ResultadosESAR(directorio_analisis)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        exit()

    print(f"Mediciones: {len(resultados.files)}")

    print(f'Concentracion: {resultados.meta["Concentracion g/m^3"]/1000} mg/mL')
    if hasattr(resultados, 'temperatura'):
        temp_min = resultados.temperatura.min()
        temp_max = resultados.temperatura.max()
        print(f"Temperatura: {temp_min:.1f}°C → {temp_max:.1f}°C")

    if hasattr(resultados, 'SAR'):
        print(f"SAR: {ufloat(resultados.SAR.mean(), resultados.SAR.std()):.1uS} W/g")

    if hasattr(resultados, 'tau'):
        print(f"Tau: {ufloat(resultados.tau.mean(), resultados.tau.std()):.2uS} ns")

    if hasattr(resultados, 'Hc'):
        print(f"Hc: {ufloat(resultados.Hc.mean(), resultados.Hc.std()):.1uS} kA/m")

    fig, ax = resultados.plot_ciclos_comparacion(guardar=True)

    fig1, ax1 = resultados.plot_ciclos_comparacion(guardar=True)

    fig2, ax2 = resultados.plot_evolucion_temporal(guardar=True)

    fig3, ax3 = resultados.plot_evolucion_temperatura(guardar=True)

    plt.show()
#%% 2 - NP quitosano
subdirectorios=os.listdir(os.path.join(os.getcwd(),"2"))
subdirectorios.sort()
print(subdirectorios)

for sd in subdirectorios:
    print(sd)
    directorio_a_analizar = os.path.join(os.getcwd(), "2", sd)

    patron_analisis = os.path.join(directorio_a_analizar, "Analisis_*")
    directorios_analisis = glob.glob(patron_analisis)

    if not directorios_analisis:
        print(f"No se encontraron directorios 'Analisis_' en {directorio_a_analizar}")
        exit()

    directorio_analisis = directorios_analisis[-1]

    try:
        resultados = ResultadosESAR(directorio_analisis)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        exit()

    print(f"Mediciones: {len(resultados.files)}")

    print(f'Concentracion: {resultados.meta["Concentracion g/m^3"]/1000} mg/mL')
    if hasattr(resultados, 'temperatura'):
        temp_min = resultados.temperatura.min()
        temp_max = resultados.temperatura.max()
        print(f"Temperatura: {temp_min:.1f}°C → {temp_max:.1f}°C")

    if hasattr(resultados, 'SAR'):
        print(f"SAR: {ufloat(resultados.SAR.mean(), resultados.SAR.std()):.1uS} W/g")

    if hasattr(resultados, 'tau'):
        print(f"Tau: {ufloat(resultados.tau.mean(), resultados.tau.std()):.2uS} ns")

    if hasattr(resultados, 'Hc'):
        print(f"Hc: {ufloat(resultados.Hc.mean(), resultados.Hc.std()):.1uS} kA/m")

    fig, ax = resultados.plot_ciclos_comparacion(guardar=True)

    fig1, ax1 = resultados.plot_ciclos_comparacion(guardar=True)

    fig2, ax2 = resultados.plot_evolucion_temporal(guardar=True)

    fig3, ax3 = resultados.plot_evolucion_temperatura(guardar=True)

    plt.show()      
    
#%% 3 - NP alginato
subdirectorios=os.listdir(os.path.join(os.getcwd(),"3"))
subdirectorios.sort()
print(subdirectorios)

for sd in subdirectorios:   
    print(sd)
    directorio_a_analizar = os.path.join(os.getcwd(), "3", sd)

    patron_analisis = os.path.join(directorio_a_analizar, "Analisis_*")
    directorios_analisis = glob.glob(patron_analisis)

    if not directorios_analisis:
        print(f"No se encontraron directorios 'Analisis_' en {directorio_a_analizar}")
        exit()

    directorio_analisis = directorios_analisis[-1]

    try:
        resultados = ResultadosESAR(directorio_analisis)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        exit()

    print(f"Mediciones: {len(resultados.files)}")

    print(f'Concentracion: {resultados.meta["Concentracion g/m^3"]/1000} mg/mL')
    if hasattr(resultados, 'temperatura'):
        temp_min = resultados.temperatura.min()
        temp_max = resultados.temperatura.max()
        print(f"Temperatura: {temp_min:.1f}°C → {temp_max:.1f}°C")

    if hasattr(resultados, 'SAR'):
        print(f"SAR: {ufloat(resultados.SAR.mean(), resultados.SAR.std()):.1uS} W/g")

    if hasattr(resultados, 'tau'):
        print(f"Tau: {ufloat(resultados.tau.mean(), resultados.tau.std()):.2uS} ns")

    if hasattr(resultados, 'Hc'):
        print(f"Hc: {ufloat(resultados.Hc.mean(), resultados.Hc.std()):.1uS} kA/m")

    fig, ax = resultados.plot_ciclos_comparacion(guardar=True)

    fig1, ax1 = resultados.plot_ciclos_comparacion(guardar=True)

    fig2, ax2 = resultados.plot_evolucion_temporal(guardar=True)

    fig3, ax3 = resultados.plot_evolucion_temperatura(guardar=True)

    plt.show()
#%% Comparativa de ciclos promedio
from lectores import plot_ciclos_promedio, lector_ciclos,lector_resultados

plot_ciclos_promedio(os.path.join(os.getcwd(),"1"))

plot_ciclos_promedio(os.path.join(os.getcwd(),"2"))

plot_ciclos_promedio(os.path.join(os.getcwd(),"3"))   

print('parece haber algun problema con el 2° alginato\nno lo considero para la comparativa final')
#%%
dir1= os.path.join(os.getcwd(),'1','251218_142737','Analisis_20251219_101402')
dir2= os.path.join(os.getcwd(),'2','251218_143840','Analisis_20251219_101755')
dir3= os.path.join(os.getcwd(),'3','251218_144635','Analisis_20251219_102350')
ciclo_promedio_1 = os.path.join(dir1,'300kHz_150dA_100Mss_bobN1P100_ciclo_promedio_H_M.txt')
ciclo_promedio_2 = os.path.join(dir2,'300kHz_150dA_100Mss_bobN1P200_ciclo_promedio_H_M.txt')
ciclo_promedio_3 = os.path.join(dir3,'300kHz_150dA_100Mss_bobN1P200_ciclo_promedio_H_M.txt')

results1=ResultadosESAR(os.path.join(dir1))
results2=ResultadosESAR(os.path.join(dir2))
results3=ResultadosESAR(os.path.join(dir3))

_,_,_,H1,M1,meta1=lector_ciclos(ciclo_promedio_1)
_,_,_,H2,M2,meta2=lector_ciclos(ciclo_promedio_2)
_,_,_,H3,M3,meta3=lector_ciclos(ciclo_promedio_3)

label1=f'NP citrico\n C: {results1.meta["Concentracion g/m^3"]/1000} g/L Fe$_3$O$_4$\n SAR: {ufloat(results1.SAR.mean(), results1.SAR.std()):.1uS} W/g\n Tau: {ufloat(results1.tau.mean(), results1.tau.std()):.1uS} ns\n' 
label2=f'NP quitosano\n C: {results2.meta["Concentracion g/m^3"]/1000} g/L Fe$_3$O$_4$\n SAR: {ufloat(results2.SAR.mean(), results2.SAR.std()):.1uS} W/g\n Tau: {ufloat(results2.tau.mean(), results2.tau.std()):.1uS} ns\n'
label3=f'NP alginato\n C: {results3.meta["Concentracion g/m^3"]/1000} g/L Fe$_3$O$_4$\n SAR: {ufloat(results3.SAR.mean(), results3.SAR.std()):.1uS} W/g\n Tau: {ufloat(results3.tau.mean(), results3.tau.std()):.1uS} ns'

fig, ax = plt.subplots(figsize=(9,6.75), constrained_layout=True)
ax.plot(H1/1000, M1, label=label1)
ax.plot(H2/1000, M2, label=label2)
ax.plot(H3/1000, M3, label=label3)
ax.set_xlabel('H (kA/m)')
ax.set_ylabel('M (A/m)')
ax.legend(ncol=1) 
ax.grid(True)
ax.set_title('Ciclos promedio - 300 kHz - 57 kA/m')
plt.savefig('comparativa_ciclos_promedio_final.png', dpi=300)
plt.show()



fig2, ax = plt.subplots(figsize=(9,6.75), constrained_layout=True)
ax.plot(H1/1000, M1/(results1.meta["Concentracion g/m^3"]/1000), label=label1)
ax.plot(H2/1000, M2/(results2.meta["Concentracion g/m^3"]/1000), label=label2)
ax.plot(H3/1000, M3/(results3.meta["Concentracion g/m^3"]/1000), label=label3)
ax.set_xlabel('H (kA/m)')
ax.set_ylabel('M/[NPM] (Am²/kg)')
ax.legend(ncol=1) 
ax.grid(True)
ax.set_title('Ciclos promedio - 300 kHz - 57 kA/m')
plt.savefig('comparativa_ciclos_normalizados_promedio_final.png', dpi=300)
plt.show()
# %%
