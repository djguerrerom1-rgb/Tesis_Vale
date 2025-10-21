import numpy as np
import pandas as pd
import os
import joblib
from src.angles.knee_angles import procesar_datos_rodilla
from src.angles.hip_angles import procesar_datos_cadera
from src.angles.ankle_angles import procesar_datos_tobillo
from src.angles.pelvis_angles import procesar_datos_pelvis
from src.angles.trunk_angles import procesar_datos_torso
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

#####  PRIMERO VAMOS A CREAR LOS ARCHIVOS NECESARIOS PARA EL MODELO ML   ######
def encontrar_giros(tiempo, senal, picos, fs=120, seg_busqueda=1, umbral=75):
    base_picos = {}
    for pico_idx in picos:
        idx_ini = max(0, pico_idx - int(seg_busqueda*fs))
        idx_fin = pico_idx
        tramo = senal[idx_ini:idx_fin]
        grad = np.gradient(tramo)
        candidatos = np.where(tramo < umbral)[0]
        if len(candidatos) > 0:
            idx_grad = np.argmax(grad[candidatos])
            idx_base_antes = idx_ini + candidatos[idx_grad]
        else:
            idx_base_antes = idx_ini + np.argmax(grad)
        tiempo_base_antes = tiempo[idx_base_antes]
        idx_post_ini = pico_idx
        idx_post_fin = min(len(senal), pico_idx + int(seg_busqueda*fs))
        tramo_post = senal[idx_post_ini:idx_post_fin]
        grad_post = np.gradient(tramo_post)
        candidatos_post = np.where(tramo_post < umbral)[0]
        if len(candidatos_post) > 0:
            idx_grad_post = np.argmin(grad_post[candidatos_post])
            idx_base_post = idx_post_ini + candidatos_post[idx_grad_post]
        else:
            idx_base_post = idx_post_ini + np.argmin(grad_post)
        tiempo_base_post = tiempo[idx_base_post]
        base_picos[pico_idx] = (tiempo_base_antes, tiempo_base_post)
    return base_picos

def detectar_tramos_quietos(signal, fs, umbral_deriv=0.2, min_duracion=1.0):
    deriv = np.abs(np.diff(signal, prepend=signal[0]))
    quiet_mask = deriv < umbral_deriv
    min_samples = int(min_duracion * fs)
    tramos = []
    en_tramo = False
    for i in range(len(quiet_mask)):
        if quiet_mask[i] and not en_tramo:
            start = i
            en_tramo = True
        elif not quiet_mask[i] and en_tramo:
            end = i
            if (end - start) >= min_samples:
                tramos.append((start, end))
            en_tramo = False
    if en_tramo and (len(quiet_mask) - start) >= min_samples:
        tramos.append((start, len(quiet_mask)))
    return tramos

def poner_tramos_a_cero(signal, tramos):
    sig = signal.copy()
    for ini, fin in tramos:
        sig[ini:fin] = 0
    return sig


def cargar_datos_3dma(directorio_3dma, nombre_archivo_3dma):
    path_3dma = buscar_archivo(directorio_3dma, nombre_archivo_3dma)
    return pd.read_csv(path_3dma, sep=';', decimal=',', encoding='utf-8')

def buscar_archivo(directorio, nombre_parcial):
    """Busca un archivo en el directorio que contenga una palabra clave."""
    for archivo in os.listdir(directorio):
        if nombre_parcial in archivo and archivo.endswith('.csv'):
            return os.path.join(directorio, archivo)
    raise FileNotFoundError(f"No se encontró el archivo para {nombre_parcial}")

def generar_excel_ml(resultados, nombre_sujeto, numero_captura, ruta_guardado):
    """
    Genera un archivo Excel para el modelo ML, usando los datos en el diccionario `resultados`.
    Devuelve el DataFrame preparado para el modelo ML y la ruta.
    """
    totalData = pd.DataFrame()
    totalData["F/E Right Knee"] = resultados["rodilla_der_f_e"]
    totalData["F/E Left Knee"] = resultados["rodilla_izq_f_e"]
    totalData["Right Hip Flexion/Extension"] = resultados["cadera_der_f_e"]
    totalData["Left Hip Flexion/Extension"] = resultados["cadera_izq_f_e"]
    totalData["RelacionCaderaDerecha"] = totalData["F/E Right Knee"] - totalData["Right Hip Flexion/Extension"]
    totalData["RelacionCaderaIzquierda"] = totalData["F/E Left Knee"] - totalData["Left Hip Flexion/Extension"]

    # Etiquetas binarias
    RelacionCaderaDerecha = (totalData["RelacionCaderaDerecha"] > 0).astype(int)
    RelacionCaderaIzquierda = (totalData["RelacionCaderaIzquierda"] > 0).astype(int)

    cambioSentidoIzquierda = [1, 1]
    cambioSentidoDerecha = [1, 1]
    for i in range(2, len(totalData)-2):
        cond_izq = all(totalData["F/E Left Knee"][i+j] <= totalData["F/E Left Knee"][i] for j in [-2, -1, 1, 2])
        cambioSentidoIzquierda.append(2 if cond_izq else 1)
        cond_der = all(totalData["F/E Right Knee"][i+j] <= totalData["F/E Right Knee"][i] for j in [-2, -1, 1, 2])
        cambioSentidoDerecha.append(2 if cond_der else 1)
    cambioSentidoIzquierda += [1, 1]
    cambioSentidoDerecha += [1, 1]

    for i in range(len(totalData)-2):
        if cambioSentidoIzquierda[i] == 2:
            for j in [-2, -1, 0, 1, 2]:
                idx = i + j
                if 0 <= idx < len(cambioSentidoIzquierda):
                    cambioSentidoIzquierda[idx] = 0
        if cambioSentidoDerecha[i] == 2:
            for j in [-2, -1, 0, 1, 2]:
                idx = i + j
                if 0 <= idx < len(cambioSentidoDerecha):
                    cambioSentidoDerecha[idx] = 0

    totalData["RelacionCaderaDerecha"] = RelacionCaderaDerecha
    totalData["RelacionCaderaIzquierda"] = RelacionCaderaIzquierda
    totalData["cambioSentidoIzquierda"] = cambioSentidoIzquierda
    totalData["cambioSentidoDerecha"] = cambioSentidoDerecha

    # Guarda en Excel
    output_path = os.path.join(ruta_guardado, f"{nombre_sujeto}_captura{numero_captura}_gait_ml.xlsx")
    totalData.to_excel(output_path, index=False)
    print(f"✅ Archivo Excel para modelo guardado en: {output_path}")

    return totalData, output_path

def generar_excel_ml_3dma(ruta_3dma, archivos_columnas, nombre_sujeto, numero_captura, ruta_guardado, modelo_path):
    carpeta_3dma = os.path.join(ruta_guardado, "3DMA")
    os.makedirs(carpeta_3dma, exist_ok=True)

    datos = {}
    for archivo, cols in archivos_columnas.items():
        df = cargar_datos_3dma(ruta_3dma, archivo)
        for key, col in cols.items():
            datos[key] = df[col].values

    totalData = pd.DataFrame()
    totalData["F/E Right Knee"] = datos["F/E Right Knee"]
    totalData["F/E Left Knee"] = datos["F/E Left Knee"]
    totalData["Right Hip Flexion/Extension"] = datos["Right Hip Flexion/Extension"]
    totalData["Left Hip Flexion/Extension"] = datos["Left Hip Flexion/Extension"]
    totalData["RelacionCaderaDerecha"] = totalData["F/E Right Knee"] - totalData["Right Hip Flexion/Extension"]
    totalData["RelacionCaderaIzquierda"] = totalData["F/E Left Knee"] - totalData["Left Hip Flexion/Extension"]

    RelacionCaderaDerecha = (totalData["RelacionCaderaDerecha"] > 0).astype(int)
    RelacionCaderaIzquierda = (totalData["RelacionCaderaIzquierda"] > 0).astype(int)

    cambioSentidoIzquierda = [1, 1]
    cambioSentidoDerecha = [1, 1]
    for i in range(2, len(totalData)-2):
        cond_izq = all(totalData["F/E Left Knee"][i+j] <= totalData["F/E Left Knee"][i] for j in [-2, -1, 1, 2])
        cambioSentidoIzquierda.append(2 if cond_izq else 1)
        cond_der = all(totalData["F/E Right Knee"][i+j] <= totalData["F/E Right Knee"][i] for j in [-2, -1, 1, 2])
        cambioSentidoDerecha.append(2 if cond_der else 1)
    cambioSentidoIzquierda += [1, 1]
    cambioSentidoDerecha += [1, 1]

    for i in range(len(totalData)-2):
        if cambioSentidoIzquierda[i] == 2:
            for j in [-2, -1, 0, 1, 2]:
                idx = i + j
                if 0 <= idx < len(cambioSentidoIzquierda):
                    cambioSentidoIzquierda[idx] = 0
        if cambioSentidoDerecha[i] == 2:
            for j in [-2, -1, 0, 1, 2]:
                idx = i + j
                if 0 <= idx < len(cambioSentidoDerecha):
                    cambioSentidoDerecha[idx] = 0

    totalData["RelacionCaderaDerecha"] = RelacionCaderaDerecha
    totalData["RelacionCaderaIzquierda"] = RelacionCaderaIzquierda
    totalData["cambioSentidoIzquierda"] = cambioSentidoIzquierda
    totalData["cambioSentidoDerecha"] = cambioSentidoDerecha

    # Excel solo datos ML
    output_path = os.path.join(carpeta_3dma, f"{nombre_sujeto}_captura{numero_captura}_gait_ml_3dma.xlsx")
    totalData.to_excel(output_path, index=False)
    print(f"✅ Archivo Excel ML (3DMA) guardado en: {output_path}")

    # Excel con predicción
    modelo = joblib.load(modelo_path)
    y_pred = modelo.predict(totalData)
    totalData['Prediccion'] = y_pred
    pred_excel_path = os.path.join(carpeta_3dma, f"{nombre_sujeto}_captura{numero_captura}_gait_ml_con_prediccion_3dma.xlsx")
    totalData.to_excel(pred_excel_path, index=False)
    print(f"✅ Archivo Excel ML (3DMA con predicción) guardado en: {pred_excel_path}")

    return totalData, output_path, pred_excel_path

###### AUTOMATIZACIÓN DEL PROCESO DE GENERACIÓN DE ÁNGULOS ######

# --- AHORA ADAPTA TU main_lb ---
def main_lb(nombre_sujeto, numero_captura):
    # --------- CONFIGURACIÓN ---------
    base_dir = r"C:\Users\Valentina\OneDrive - Universidad de los andes\Documentos\GitHub\IMU-3D-Kinematics\data\SUJETOS"
    carpeta_sujeto = os.path.join(base_dir, nombre_sujeto)
    ruta_xsens = os.path.join(carpeta_sujeto, "XSENS", "XSENS_CUATERNIONES", str(numero_captura))

    # --------- CREA CARPETA RESULTADOS SUJETO ---------
    carpeta_resultados_general = os.path.join(base_dir, "RESULTADOS")
    if not os.path.exists(carpeta_resultados_general):
        os.makedirs(carpeta_resultados_general)
    carpeta_resultados = os.path.join(carpeta_resultados_general, nombre_sujeto)
    if not os.path.exists(carpeta_resultados):
        os.makedirs(carpeta_resultados)

    # --------- PROCESA LOS ÁNGULOS ---------
    lados = ["der", "izq"]
    resultados = {}

    # --- Primero, obtenemos los ángulos y los guardamos ---
    angulos_guardar = {}
    for lado in lados:
        archivo_muslo = f"muslo_{lado}"
        archivo_tibia = f"tibia_{lado}"
        archivo_pie   = f"pie_{lado}"
        archivo_pelvis = "pelvis"

        angulos_rodilla = procesar_datos_rodilla(
            ruta_xsens, archivo_muslo, archivo_tibia, archivo_pelvis, usar_identidad=False
        )
        angulos_guardar[f"rodilla_{lado}_f_e"] = angulos_rodilla[:, 0]
        angulos_guardar[f"rodilla_{lado}_rot_int_ext"] = angulos_rodilla[:, 1]
        angulos_guardar[f"rodilla_{lado}_abd_add"] = angulos_rodilla[:, 2]

        angulos_cadera, giroscopio_cadera = procesar_datos_cadera(
            ruta_xsens, archivo_muslo, archivo_pelvis, usar_identidad=False
        )
        angulos_guardar[f"cadera_{lado}_f_e"] = angulos_cadera[:, 0]
        angulos_guardar[f"cadera_{lado}_abd_add"] = angulos_cadera[:, 1]
        angulos_guardar[f"cadera_{lado}_rot_int_ext"] = angulos_cadera[:, 2]

        angulos_tobillo = procesar_datos_tobillo(
            ruta_xsens, archivo_tibia, archivo_pie, usar_identidad=False
        )
        angulos_guardar[f"tobillo_{lado}_f_e"] = angulos_tobillo

    angulos_pelvis = procesar_datos_pelvis(ruta_xsens, "pelvis", usar_identidad=False)
    angulos_guardar["pelvic_tilt"] = angulos_pelvis[:, 0]
    angulos_guardar["pelvic_obliq"] = angulos_pelvis[:, 1]

    #angulos_torso = procesar_datos_torso(ruta_xsens, "torso", usar_identidad=False)
    #angulos_guardar["trunk_tilt"] = angulos_torso[:, 0]
    #angulos_guardar["trunk_obliq"] = angulos_torso[:, 1]

    N = len(angulos_pelvis)
    frecuencia = 120  # Hz
    tiempo = np.arange(N) / frecuencia
    angulos_guardar["tiempo"] = tiempo

    # --- Aquí va la parte nueva: limpiar giros/inicios en las señales de cadera/rodilla ---
    # Si tienes el archivo del giroscopio de pelvis (como en tus scripts de análisis):
    ruta_pelvis = os.path.join(ruta_xsens, "pelvis.csv")
    if os.path.exists(ruta_pelvis):
        df_pelvis = pd.read_csv(ruta_pelvis)
        giroscopio = df_pelvis['Gyr_X'].values
        t_g = np.arange(len(giroscopio)) / frecuencia
        sigma = 6
        giroscopio_suav = gaussian_filter1d(giroscopio, sigma=sigma)
        picos_giro, _ = find_peaks(giroscopio_suav, prominence=150)
        base_picos_giro = encontrar_giros(t_g, giroscopio_suav, picos_giro, fs=frecuencia, seg_busqueda=1, umbral=75)
        tramos_giro = []
        for _, (t_ini, t_fin) in base_picos_giro.items():
            idx_ini = int(t_ini * frecuencia)
            idx_fin = int(t_fin * frecuencia)
            tramos_giro.append((idx_ini, idx_fin))

        # Opcional: tramos quietos (puedes usar solo giros si prefieres)
        tramos_quietos = detectar_tramos_quietos(angulos_guardar["cadera_der_f_e"], fs=frecuencia, umbral_deriv=0.2, min_duracion=1.0)
        tramos_total = tramos_giro + tramos_quietos

        # --- Limpia solo las señales relevantes ---
        for clave in ["cadera_der_f_e", "cadera_izq_f_e", "rodilla_der_f_e", "rodilla_izq_f_e"]:
            if clave in angulos_guardar:
                angulos_guardar[clave] = poner_tramos_a_cero(angulos_guardar[clave], tramos_total)

    # --- AJUSTE: RECORTA TODOS LOS ARRAYS AL MISMO LARGO ---
    columnas = ["tiempo"] + [k for k in angulos_guardar if k != "tiempo"]
    min_len = min(len(angulos_guardar[col]) for col in columnas)
    for col in columnas:
        angulos_guardar[col] = angulos_guardar[col][:min_len]

    # --- CREA EL DATAFRAME Y GUARDA ---
    nombre_csv = f"{nombre_sujeto}_captura{numero_captura}.csv"
    path_csv = os.path.join(carpeta_resultados, nombre_csv)
    df = pd.DataFrame({col: angulos_guardar[col] for col in columnas})
    df.to_csv(path_csv, index=False)
    print(f"\nArchivo CSV guardado: {path_csv}")

    # --- GENERA EL EXCEL PARA ML ---
    totalData, path_excel = generar_excel_ml(angulos_guardar, nombre_sujeto, numero_captura, carpeta_resultados)

    # --- PREDICCIÓN USANDO MODELO .pkl ---
    modelo_path = r'C:\Users\Valentina\OneDrive - Universidad de los andes\Documentos\GitHub\IMU-3D-Kinematics\src\angles\modelo_entrenadoSegmentarCicloDeMarcha.pkl'
    modelo = joblib.load(modelo_path)
    y_pred = modelo.predict(totalData)

    # --- GUARDA LA PREDICCIÓN JUNTO AL EXCEL (agrega columna) ---
    pred_excel_path = os.path.join(carpeta_resultados, f"{nombre_sujeto}_captura{numero_captura}_gait_ml_con_prediccion.xlsx")
    totalData['Prediccion'] = y_pred
    totalData.to_excel(pred_excel_path, index=False)
    print(f"✅ Archivo Excel con predicción guardado en: {pred_excel_path}")

    # --- RESTO IGUAL ---
    print(f"\nTodo listo para {nombre_sujeto} captura {numero_captura}. Los archivos están en:\n{carpeta_resultados}")

    ##### PARA LOS DATOS DE 3DMA SIMULTANEAMENTE
    ruta_3dma = os.path.join(carpeta_sujeto, "3DMA", str(numero_captura))
    archivos_columnas_3dma = {
        "knee angles.csv": {
            "F/E Right Knee": "Right Knee Flexion/Extension::Y",
            "F/E Left Knee": "Left Knee Flexion/Extension::Y"
        },
        "hip angles.csv": {
            "Right Hip Flexion/Extension": "Right Hip Flexion/Extension::Y",
            "Left Hip Flexion/Extension": "Left Hip Flexion/Extension::Y"
        }
    }
    modelo_path = r'C:\Users\Valentina\OneDrive - Universidad de los andes\Documentos\GitHub\IMU-3D-Kinematics\src\angles\modelo_entrenadoSegmentarCicloDeMarcha.pkl'
    generar_excel_ml_3dma(ruta_3dma, archivos_columnas_3dma, nombre_sujeto, numero_captura, carpeta_resultados, modelo_path)

    return df, totalData

# ------------ Cambia tus listas aquí ----------------
lista_sujetos = [
    "Amid_Delgado",
    "Camila_Grazziani",
    "Christian_Cifuentes",
    "Daniela_Guerrero",
    "David_Solorzano",
    "Isabel_Bejarano",
    "Johann_Roa",
    "Jose_Tavera",
    "Juan_Salgado",
    "Laura_Escobedo",
    "Laura_Urrea",
    "Laura_Vidal",
    "Liss_Rios",
    "Santiago_Flores",
    "Valentina_Pinzon",
    "Veronica_Almeida"
]

capturas_por_sujeto = {
    "Amid_Delgado": [3],
    "Camila_Grazziani": [2,3,4],
    "Christian_Cifuentes": [1,2],
    "Daniela_Guerrero": [2],
    "David_Solorzano": [1,2],
    "Isabel_Bejarano": [6],
    "Johann_Roa": [2,3],
    "Jose_Tavera": [1,2],
    "Juan_Salgado": [2],
    "Laura_Escobedo": [1,2, 3],
    "Laura_Urrea": [2,3],
    "Laura_Vidal": [3],
    "Liss_Rios": [5],
    "Santiago_Flores": [1,2],
    "Valentina_Pinzon": [5],
    "Veronica_Almeida": [5]
}

def procesar_todo():
    for nombre_sujeto in lista_sujetos:
        for numero_captura in capturas_por_sujeto[nombre_sujeto]:
            try:
                print(f"\n=== Procesando {nombre_sujeto} captura {numero_captura} ===")
                main_lb(nombre_sujeto, numero_captura)
            except Exception as e:
                print(f"Error procesando {nombre_sujeto} captura {numero_captura}: {e}")

if __name__ == "__main__":
    main_lb("CICLISMO", "bici1")
    #procesar_todo()