#%% nueva_clase_resultados.py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from uncertainties import ufloat, unumpy
from lectores import lector_resultados, lector_ciclos

class AnalizadorESAR:
    """
    Clase para analizar resultados ESAR agrupados por frecuencia y corriente.
    
    Estructura esperada:
    directorio_base/
    ├── frecuencia_135/
    │   ├── campo_050/
    │   │   ├── Analisis_20240101_rep1/
    │   │   ├── Analisis_20240101_rep2/
    │   │   └── Analisis_20240101_rep3/
    │   ├── campo_075/
    │   └── ...
    ├── frecuencia_212/
    └── frecuencia_300/
    """
    
    def __init__(self, directorio_base):
        """
        Inicializa el analizador con el directorio base.
        
        Parámetros:
        -----------
        directorio_base : str
            Directorio que contiene las carpetas de frecuencia
        """
        self.directorio_base = Path(directorio_base).resolve()
        
        if not self.directorio_base.exists():
            raise FileNotFoundError(f"✗ Directorio no encontrado: {self.directorio_base}")
        
        print(f"\n{'='*60}")
        print(f"ANALIZADOR ESAR - CARGA INICIAL")
        print(f"{'='*60}")
        print(f"Directorio base: {self.directorio_base}")
        
        # 1. Encontrar todas las carpetas de frecuencia
        self._encontrar_frecuencias()
        
        # 2. Para cada frecuencia, encontrar las corrientes
        self._analizar_estructura()
        
    def _encontrar_frecuencias(self):
        """Encuentra todas las carpetas de frecuencia en el directorio base."""
        self.frecuencias = {}
        
        # Buscar carpetas que contengan 'frecuencia' o 'freq' en el nombre
        patrones = ['*frecuencia*', '*freq*', 'F*Hz*']
        
        for patron in patrones:
            carpetas = list(self.directorio_base.glob(patron))
            for carpeta in carpetas:
                if carpeta.is_dir():
                    # Extraer valor de frecuencia del nombre
                    freq_val = self._extraer_frecuencia(carpeta.name)
                    if freq_val:
                        self.frecuencias[freq_val] = {
                            'ruta': carpeta,
                            'nombre': carpeta.name
                        }
        
        # Si no encontramos con patrones, buscar cualquier carpeta
        if not self.frecuencias:
            carpetas = [d for d in self.directorio_base.iterdir() if d.is_dir()]
            for carpeta in carpetas:
                freq_val = self._extraer_frecuencia(carpeta.name)
                if freq_val:
                    self.frecuencias[freq_val] = {
                        'ruta': carpeta,
                        'nombre': carpeta.name
                    }
        
        print(f"✓ Frecuencias encontradas: {len(self.frecuencias)}")
        for freq in sorted(self.frecuencias.keys()):
            print(f"  - {freq} Hz: {self.frecuencias[freq]['nombre']}")
    
    def _extraer_frecuencia(self, nombre):
        """Extrae el valor numérico de frecuencia del nombre de carpeta."""
        import re
        # Buscar números en el nombre (pueden ser 135, 212, 300, etc.)
        numeros = re.findall(r'\d+', nombre)
        if numeros:
            # Tomar el número más grande (asumiendo que es la frecuencia en Hz/kHz)
            nums = [int(n) for n in numeros]
            # Filtrar números razonables para frecuencia (50-1000 Hz)
            freqs = [n for n in nums if 50 <= n <= 1000]
            if freqs:
                return max(freqs)  # Devolver la frecuencia más alta encontrada
        return None
    
    def _extraer_corriente(self, nombre):
        """Extrae el valor numérico de corriente del nombre de carpeta."""
        import re
        # Buscar números de 3 dígitos (050, 075, 100, etc.)
        numeros = re.findall(r'\b\d{3}\b', nombre)
        if numeros:
            # Convertir a entero (050 -> 50, 100 -> 100)
            return int(numeros[0])
        
        # Buscar cualquier número
        numeros = re.findall(r'\d+', nombre)
        if numeros:
            return int(numeros[0])
        
        return None
    
    def _analizar_estructura(self):
        """Analiza la estructura de directorios para cada frecuencia."""
        self.estructura = {}
        
        for freq, info in self.frecuencias.items():
            print(f"\nAnalizando frecuencia {freq} Hz...")
            
            # Buscar carpetas de corriente/campo
            corrientes_encontradas = {}
            
            # Patrones para buscar carpetas de corriente
            patrones_corriente = [
                '*campo*', '*corriente*', '*current*',
                '*050*', '*075*', '*100*', '*125*', '*150*'
            ]
            
            for patron in patrones_corriente:
                carpetas = list(info['ruta'].glob(patron))
                for carpeta in carpetas:
                    if carpeta.is_dir():
                        corriente_val = self._extraer_corriente(carpeta.name)
                        if corriente_val:
                            corrientes_encontradas[corriente_val] = {
                                'ruta': carpeta,
                                'nombre': carpeta.name,
                                'repeticiones': self._encontrar_repeticiones(carpeta)
                            }
            
            # Ordenar por corriente
            corrientes_ordenadas = dict(sorted(corrientes_encontradas.items()))
            
            if corrientes_ordenadas:
                self.estructura[freq] = corrientes_ordenadas
                print(f"  ✓ Corrientes encontradas: {list(corrientes_ordenadas.keys())}")
                for corr, datos in corrientes_ordenadas.items():
                    print(f"    - {corr}: {len(datos['repeticiones'])} repeticiones")
            else:
                print(f"  ✗ No se encontraron corrientes para frecuencia {freq} Hz")
    
    def _encontrar_repeticiones(self, directorio_corriente):
        """Encuentra todas las repeticiones (Analisis_*) en un directorio de corriente."""
        repeticiones = []
        
        # Buscar directorios que comiencen con 'Analisis_'
        analisis_dirs = list(directorio_corriente.glob('Analisis_*'))
        
        for dir_analisis in analisis_dirs:
            if dir_analisis.is_dir():
                # Verificar que tenga el archivo resultados.txt
                resultados_files = list(dir_analisis.glob('*resultados.txt'))
                if resultados_files:
                    repeticiones.append({
                        'ruta': dir_analisis,
                        'resultados': resultados_files[0],
                        'ciclos_dir': dir_analisis / 'ciclos_H_M'
                    })
        
        return repeticiones
    
    def cargar_frecuencia(self, frecuencia):
        """
        Carga y procesa todos los datos para una frecuencia específica.
        
        Parámetros:
        -----------
        frecuencia : int
            Frecuencia a analizar (ej: 135, 212, 300)
        
        Retorna:
        --------
        datos_procesados : dict
            Diccionario con datos promediados por corriente
        """
        if frecuencia not in self.estructura:
            print(f"✗ Frecuencia {frecuencia} Hz no encontrada")
            print(f"  Frecuencias disponibles: {list(self.estructura.keys())}")
            return None
        
        print(f"\n{'='*60}")
        print(f"PROCESANDO FRECUENCIA {frecuencia} Hz")
        print(f"{'='*60}")
        
        datos_frecuencia = {
            'frecuencia': frecuencia,
            'corrientes': {},
            'metadata': {}
        }
        
        # Procesar cada corriente
        for corriente, info in self.estructura[frecuencia].items():
            print(f"\nCorriente {corriente}:")
            
            if not info['repeticiones']:
                print(f"  ✗ No hay repeticiones disponibles")
                continue
            
            # Cargar todas las repeticiones para esta corriente
            repeticiones_data = []
            
            for i, rep in enumerate(info['repeticiones']):
                print(f"  Repetición {i+1}: {rep['ruta'].name}")
                
                try:
                    # Cargar datos de resultados.txt
                    (meta, files, time, temperatura, Mr, Hc, campo_max, mag_max,
                     xi_M_0, frecuencia_fund, magnitud_fund, dphi_fem, SAR, tau, N) = lector_resultados(str(rep['resultados']))
                    
                    # Almacenar datos
                    rep_data = {
                        'ruta': rep['ruta'],
                        'meta': meta,
                        'SAR': SAR,
                        'Hc': Hc,
                        'tau': tau,
                        'Mr': Mr,
                        'campo_max': campo_max,
                        'mag_max': mag_max,
                        'temperatura': temperatura,
                        'time': time,
                        'files': files,
                        'indice': i
                    }
                    
                    repeticiones_data.append(rep_data)
                    print(f"    ✓ Datos cargados: {len(SAR)} puntos")
                    
                except Exception as e:
                    print(f"    ✗ Error cargando repetición: {e}")
                    continue
            
            if not repeticiones_data:
                print(f"  ✗ No se pudieron cargar repeticiones para corriente {corriente}")
                continue
            
            # Promediar las repeticiones
            datos_promediados = self._promediar_repeticiones(repeticiones_data)
            datos_frecuencia['corrientes'][corriente] = datos_promediados
            
            print(f"  ✓ Promediado: {len(repeticiones_data)} repeticiones")
            print(f"    SAR promedio: {datos_promediados['SAR_promedio']:.3f} ± {datos_promediados['SAR_error']:.3f} W/g")
        
        return datos_frecuencia
    
    def _promediar_repeticiones(self, repeticiones_data):
        """
        Promedia múltiples repeticiones usando incertezas.
        
        Parámetros:
        -----------
        repeticiones_data : list
            Lista de diccionarios con datos de cada repetición
        
        Retorna:
        --------
        datos_promediados : dict
            Datos promediados con incertezas
        """
        # Suponemos que todas las repeticiones tienen el mismo número de puntos
        n_puntos = len(repeticiones_data[0]['SAR'])
        n_repeticiones = len(repeticiones_data)
        
        # Inicializar arrays para promedios
        SAR_promedio = np.zeros(n_puntos)
        SAR_error = np.zeros(n_puntos)
        Hc_promedio = np.zeros(n_puntos)
        Hc_error = np.zeros(n_puntos)
        tau_promedio = np.zeros(n_puntos)
        tau_error = np.zeros(n_puntos)
        Mr_promedio = np.zeros(n_puntos)
        Mr_error = np.zeros(n_puntos)
        
        # Para cada punto en el tiempo, promediar entre repeticiones
        for i in range(n_puntos):
            # Recolectar valores de todas las repeticiones para este punto
            sar_vals = []
            hc_vals = []
            tau_vals = []
            mr_vals = []
            
            for rep in repeticiones_data:
                if i < len(rep['SAR']):
                    sar_vals.append(rep['SAR'][i])
                    hc_vals.append(rep['Hc'][i])
                    tau_vals.append(rep['tau'][i])
                    mr_vals.append(rep['Mr'][i])
            
            # Calcular promedio y error estándar
            if sar_vals:
                SAR_promedio[i] = np.mean(sar_vals)
                SAR_error[i] = np.std(sar_vals) / np.sqrt(len(sar_vals))
            
            if hc_vals:
                Hc_promedio[i] = np.mean(hc_vals)
                Hc_error[i] = np.std(hc_vals) / np.sqrt(len(hc_vals))
            
            if tau_vals:
                tau_promedio[i] = np.mean(tau_vals)
                tau_error[i] = np.std(tau_vals) / np.sqrt(len(tau_vals))
            
            if mr_vals:
                Mr_promedio[i] = np.mean(mr_vals)
                Mr_error[i] = np.std(mr_vals) / np.sqrt(len(mr_vals))
        
        # También promediamos otras cantidades escalares
        campo_max_promedio = np.mean([rep['campo_max'][0] for rep in repeticiones_data if len(rep['campo_max']) > 0])
        mag_max_promedio = np.mean([rep['mag_max'][0] for rep in repeticiones_data if len(rep['mag_max']) > 0])
        
        # Temperatura promedio (puede ser constante)
        temp_vals = []
        for rep in repeticiones_data:
            if len(rep['temperatura']) > 0:
                temp_vals.extend(rep['temperatura'])
        
        temperatura_promedio = np.mean(temp_vals) if temp_vals else np.nan
        temperatura_std = np.std(temp_vals) if temp_vals else np.nan
        
        # Metadata de la primera repetición
        meta_promedio = repeticiones_data[0]['meta'].copy()
        
        return {
            'repeticiones': repeticiones_data,
            'n_repeticiones': n_repeticiones,
            'SAR_promedio': SAR_promedio,
            'SAR_error': SAR_error,
            'Hc_promedio': Hc_promedio,
            'Hc_error': Hc_error,
            'tau_promedio': tau_promedio,
            'tau_error': tau_error,
            'Mr_promedio': Mr_promedio,
            'Mr_error': Mr_error,
            'campo_max_promedio': campo_max_promedio,
            'mag_max_promedio': mag_max_promedio,
            'temperatura_promedio': temperatura_promedio,
            'temperatura_std': temperatura_std,
            'meta': meta_promedio,
            'indices_temporales': repeticiones_data[0]['time'] if n_repeticiones > 0 else None
        }
    
    def plot_frecuencia(self, frecuencia, parametro='SAR', figsize=(12, 8), guardar=False):
        """
        Grafica un parámetro vs corriente para una frecuencia específica.
        
        Parámetros:
        -----------
        frecuencia : int
            Frecuencia a graficar
        parametro : str
            Parámetro a graficar ('SAR', 'Hc', 'tau', 'Mr')
        figsize : tuple
            Tamaño de la figura
        guardar : bool o str
            Si es True, guarda el gráfico
        """
        if frecuencia not in self.estructura:
            print(f"✗ Frecuencia {frecuencia} Hz no encontrada")
            return None
        
        # Cargar datos si no están ya cargados
        if not hasattr(self, 'datos_cargados') or frecuencia not in self.datos_cargados:
            datos_freq = self.cargar_frecuencia(frecuencia)
            if not datos_freq:
                return None
            if not hasattr(self, 'datos_cargados'):
                self.datos_cargados = {}
            self.datos_cargados[frecuencia] = datos_freq
        
        datos_freq = self.datos_cargados[frecuencia]
        
        # Preparar datos para graficar
        corrientes = sorted(datos_freq['corrientes'].keys())
        
        if not corrientes:
            print(f"✗ No hay datos para frecuencia {frecuencia} Hz")
            return None
        
        valores = []
        errores = []
        
        for corriente in corrientes:
            datos_corr = datos_freq['corrientes'][corriente]
            
            # Tomar el valor promedio del parámetro
            if parametro == 'SAR':
                vals = datos_corr['SAR_promedio']
            elif parametro == 'Hc':
                vals = datos_corr['Hc_promedio']
            elif parametro == 'tau':
                vals = datos_corr['tau_promedio']
            elif parametro == 'Mr':
                vals = datos_corr['Mr_promedio']
            else:
                print(f"✗ Parámetro '{parametro}' no reconocido")
                return None
            
            # Promediar sobre el tiempo (si hay múltiples puntos)
            valor_promedio = np.mean(vals) if len(vals) > 0 else np.nan
            
            # Calcular error combinado
            if parametro == 'SAR':
                error_vals = datos_corr['SAR_error']
            elif parametro == 'Hc':
                error_vals = datos_corr['Hc_error']
            elif parametro == 'tau':
                error_vals = datos_corr['tau_error']
            elif parametro == 'Mr':
                error_vals = datos_corr['Mr_error']
            
            error_promedio = np.sqrt(np.sum(error_vals**2)) / len(error_vals) if len(error_vals) > 0 else 0
            
            valores.append(valor_promedio)
            errores.append(error_promedio)
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convertir corrientes a valores de campo si es necesario
        # (Aquí asumimos que la corriente es proporcional al campo)
        ejex = corrientes
        
        ax.errorbar(ejex, valores, yerr=errores, fmt='o-', 
                   capsize=5, capthick=2, linewidth=2, markersize=8,
                   label=f'{frecuencia} Hz')
        
        # Configurar gráfico
        ax.set_xlabel('Corriente', fontsize=12, fontweight='bold')
        ax.set_ylabel(self._get_parametro_nombre(parametro), fontsize=12, fontweight='bold')
        ax.set_title(f'{self._get_parametro_nombre(parametro)} vs Corriente - {frecuencia} Hz', 
                    fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Guardar si se solicita
        if guardar:
            nombre_base = f"{parametro}_vs_corriente_{frecuencia}Hz"
            if isinstance(guardar, str):
                nombre_archivo = guardar
            else:
                nombre_archivo = f"{nombre_base}.png"
            
            ruta_guardado = self.directorio_base / nombre_archivo
            fig.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado en: {ruta_guardado}")
        
        return fig, ax
    
    def _get_parametro_nombre(self, parametro):
        """Devuelve el nombre completo del parámetro."""
        nombres = {
            'SAR': 'SAR (W/g)',
            'Hc': 'Campo Coercitivo (kA/m)',
            'tau': 'τ (ns)',
            'Mr': 'Magnetización Remanente (A/m)'
        }
        return nombres.get(parametro, parametro)
    
    def analizar_todas_frecuencias(self):
        """Analiza y procesa todas las frecuencias encontradas."""
        print(f"\n{'='*60}")
        print(f"ANALIZANDO TODAS LAS FRECUENCIAS")
        print(f"{'='*60}")
        
        resultados_totales = {}
        
        for frecuencia in self.estructura.keys():
            print(f"\nProcesando frecuencia {frecuencia} Hz...")
            datos_freq = self.cargar_frecuencia(frecuencia)
            if datos_freq:
                resultados_totales[frecuencia] = datos_freq
        
        self.datos_cargados = resultados_totales
        return resultados_totales
    
    def plot_comparacion_frecuencias(self, parametro='SAR', figsize=(12, 8), guardar=False):
        """
        Compara un parámetro entre diferentes frecuencias.
        
        Parámetros:
        -----------
        parametro : str
            Parámetro a comparar
        figsize : tuple
            Tamaño de la figura
        guardar : bool o str
            Si es True, guarda el gráfico
        """
        if not hasattr(self, 'datos_cargados'):
            print("⚠ Primero debe cargar los datos. Ejecute analizar_todas_frecuencias()")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for frecuencia, datos_freq in self.datos_cargados.items():
            corrientes = sorted(datos_freq['corrientes'].keys())
            valores = []
            errores = []
            
            for corriente in corrientes:
                datos_corr = datos_freq['corrientes'][corriente]
                
                if parametro == 'SAR':
                    vals = datos_corr['SAR_promedio']
                    errs = datos_corr['SAR_error']
                elif parametro == 'Hc':
                    vals = datos_corr['Hc_promedio']
                    errs = datos_corr['Hc_error']
                elif parametro == 'tau':
                    vals = datos_corr['tau_promedio']
                    errs = datos_corr['tau_error']
                elif parametro == 'Mr':
                    vals = datos_corr['Mr_promedio']
                    errs = datos_corr['Mr_error']
                else:
                    continue
                
                valor_promedio = np.mean(vals) if len(vals) > 0 else np.nan
                error_promedio = np.sqrt(np.sum(errs**2)) / len(errs) if len(errs) > 0 else 0
                
                valores.append(valor_promedio)
                errores.append(error_promedio)
            
            # Graficar esta frecuencia
            ax.errorbar(corrientes, valores, yerr=errores, fmt='o-',
                       capsize=5, capthick=2, linewidth=2, markersize=8,
                       label=f'{frecuencia} Hz')
        
        # Configurar gráfico
        ax.set_xlabel('Corriente', fontsize=12, fontweight='bold')
        ax.set_ylabel(self._get_parametro_nombre(parametro), fontsize=12, fontweight='bold')
        ax.set_title(f'Comparación de {self._get_parametro_nombre(parametro)} por Frecuencia', 
                    fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, title='Frecuencia')
        
        # Guardar si se solicita
        if guardar:
            nombre_base = f"comparacion_{parametro}_todas_frecuencias"
            if isinstance(guardar, str):
                nombre_archivo = guardar
            else:
                nombre_archivo = f"{nombre_base}.png"
            
            ruta_guardado = self.directorio_base / nombre_archivo
            fig.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado en: {ruta_guardado}")
        
        return fig, ax

