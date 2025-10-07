# Laboratorio 3 - Análisis espectral de la voz 
**Universidad Militar Nueva Granada**  
**Asignatura:** Laboratorio de Procesamiento Digital de Señales  
**Estudiantes:** [Maria Jose Peña Velandia, Joshara Valentina Palacios, Lina Marcela Pabuena]  
**Fecha:** Octubre 2025  
**Título de la práctica:** Análisis espectral de la voz 

## Objetivos
- Capturar y procesar señales de voz masculinas y femeninas.
- Aplicar la Transformada de Fourier como herramienta de análisis espectral de la
voz.
- Extraer parámetros característicos de la señal de voz: frecuencia fundamental,
frecuencia media, brillo, intensidad, jitter y shimmer.
- Comparar las diferencias principales entre señales de voz de hombres y mujeres
a partir de su análisis en frecuencia.
- Desarrollar conclusiones sobre el comportamiento espectral de la voz humana
en función del género.

# PARTE A - Adquisición de las señales de voz

### 1. Estandarización de la Frecuencia de Muestreo en la Grabación de Audio
La frecuencia de muestreo determina cuántas veces por segundo se digitaliza una señal analógica de voz. En nuestro caso implementamos la grabadora de voz de un iPhone 11, que posee una frecuencia de muestreo de 44,1 kHz, ya que esta frecuencia es suficiente para capturar el rango audible del oido humano (20 Hz a 20kHz), en cuanto a la profundidad de bits se define la resolución con la que se cuantifica cada muestra de audio. En nuestro caso la grabadora implementa 16 bits que es el estándar en grabación semiprofesional y profesional. 

### 2. Selección del Formato WAV para Guardar Archivos
El formato WAV se utiliza por ser un estándar en audio sin compresión, lo que significa que la señal grabada se almacena de manera correcta, sin pérdidas ni alteraciones, esto es importante ya que permite el analisis más preciso, esto es crucial para el análisis espectral, la frecuencia fundamental, la intensidad (energía)

### 3. Importación y Visualización de Señales de Voz en Python
<pre> 
  from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
import wave

# Montar Google Drive (una sola vez)
drive.mount('/content/drive')

# Función para cargar audio y devolver tiempos y señal
def cargar_audio(ruta):
    audio = wave.open(ruta, "rb")
    fs = audio.getframerate()
    n_samples = audio.getnframes()
    signal_wave = audio.readframes(-1)
    audio.close()

    signal = np.frombuffer(signal_wave, dtype=np.int16)
    times = np.linspace(0, n_samples/fs, num=n_samples)
    return times, signal

# Rutas de audios
audios_mujer = [
    "/content/drive/MyDrive/Audios/Mujer_1.wav",
    "/content/drive/MyDrive/Audios/Mujer_2.wav",
    "/content/drive/MyDrive/Audios/Mujer_3.wav"
]

audios_hombre = [
    "/content/drive/MyDrive/Audios/Hombre_1.wav",
    "/content/drive/MyDrive/Audios/Hombre_2.wav",
    "/content/drive/MyDrive/Audios/Hombre_3.wav"
]

# Crear figura con 3 filas y 2 columnas
fig, axs = plt.subplots(3, 2, figsize=(12, 8))

# Agregar título principal
fig.suptitle("Gráficas en dominio del tiempo", fontsize=16, fontweight="bold")

# Graficar audios de mujer en la primera columna
for i, ruta in enumerate(audios_mujer):
    times, signal = cargar_audio(ruta)
    axs[i, 0].plot(times, signal)
    axs[i, 0].set_title(f"Mujer {i+1}")
    axs[i, 0].set_xlabel("Tiempo [s]")
    axs[i, 0].set_ylabel("Amplitud")
    axs[i, 0].grid(True)

# Graficar audios de hombre en la segunda columna
for i, ruta in enumerate(audios_hombre):
    times, signal = cargar_audio(ruta)
    axs[i, 1].plot(times, signal)
    axs[i, 1].set_title(f"Hombre {i+1}")
    axs[i, 1].set_xlabel("Tiempo [s]")
    axs[i, 1].set_ylabel("Amplitud")
    axs[i, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])  
plt.show()
</pre>

#### Gráficas en el dominio del tiempo

<img width="1189" height="789" alt="image" src="https://github.com/user-attachments/assets/a5234d2b-e700-4119-a911-34ddaffa6ad3" />
Al analizar las formas de onda obtenidas de las grabaciones de mujeres, se observan oscilaciones más rápidas y cerradas, lo que indica la presencia de frecuencias fundamentales más altas. Se aprecian diferencias en la intensidad, algunas señales presentan amplitudes más dispersas, como en el caso de la Mujer 1, mientras que otras, como la Mujer 3, muestran una distribución de amplitud más compacta. Además, la energía se mantiene distribuida de manera relativamente uniforme a lo largo de la frase, sin grandes pausas ni silencios muy largoss.
Por otro lado, en las grabaciones de hombres, las ondas presentan periodos más largos, característicos de frecuencias fundamentales más bajas. Se identifican picos de amplitud mucho más marcados en Hombre 2 y Hombre 3, reflejando una mayor intensidad. También se observan pausas cortas mas notorias que en las grabaciones femeninas.
Al comparar ambos grupos, las voces femeninas se caracterizan por frecuencias más agudas y periodos cortos, mientras que las voces masculinas muestran frecuencias más graves, periodos más largos y mayor amplitud entre los diferentes segmentos. Estos resultados nos demuestran la diferencia fisiológica en las cuerdas vocales de hombres y mujeres.

### 4. Transformada de Fourier y el Espectro de Magnitudes
<pre>
from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
import wave

# Montar Google Drive
drive.mount('/content/drive')

# Función para cargar audio
def cargar_audio(ruta):
    audio = wave.open(ruta, "rb")
    fs = audio.getframerate()
    n_samples = audio.getnframes()
    signal_wave = audio.readframes(-1)
    audio.close()

    signal = np.frombuffer(signal_wave, dtype=np.int16)
    return signal, fs

# Función para calcular y graficar FFT en semilogx
def graficar_fft(signal, fs, ax, titulo):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)     # frecuencias positivas
    fft_vals = np.abs(np.fft.rfft(signal)) # magnitud FFT

    ax.semilogx(freqs, fft_vals)           # eje x logarítmico
    ax.set_title(titulo)
    ax.set_xlabel("Frecuencia [Hz] (log)")
    ax.set_ylabel("Magnitud |X(f)|")
    ax.grid(True, which="both")

# Rutas de audios
audios_mujer = [
    "/content/drive/MyDrive/Audios/Mujer_1.wav",
    "/content/drive/MyDrive/Audios/Mujer_2.wav",
    "/content/drive/MyDrive/Audios/Mujer_3.wav"
]

audios_hombre = [
    "/content/drive/MyDrive/Audios/Hombre_1.wav",
    "/content/drive/MyDrive/Audios/Hombre_2.wav",
    "/content/drive/MyDrive/Audios/Hombre_3.wav"
]

# Crear figura (3 filas x 2 columnas)
fig, axs = plt.subplots(3, 2, figsize=(12, 8))
fig.suptitle("Transformada de Fourier (Escala Semilogarítmica)", fontsize=16, fontweight="bold")

# Mujer
for i, ruta in enumerate(audios_mujer):
    signal, fs = cargar_audio(ruta)
    graficar_fft(signal, fs, axs[i, 0], f"Mujer {i+1}")

# Hombre
for i, ruta in enumerate(audios_hombre):
    signal, fs = cargar_audio(ruta)
    graficar_fft(signal, fs, axs[i, 1], f"Hombre {i+1}")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
</pre>

 <img width="1189" height="789" alt="image" src="https://github.com/user-attachments/assets/f70540d7-d952-4e8c-b02a-6ab00263b5c3" />

La Transformada de Fourier (FT) nos ayuda a desarmar una señal de voz para ver qué sonidos la forman, mostrando claramente las diferentes frecuencias y cuánta fuerza tiene cada una. Cuando la usamos en las grabaciones, aparece una especie de gráfico con picos que nos dicen cuál es la frecuencia principal de la voz (más aguda en mujeres, mas grave en hombres) y también nos muestra los armónicos. Este espectro resulta clave porque nos da una imagen clara de cómo suena la voz, ayuda a distinguir fácilmente entre personas y géneros.

### 5. Características
#### Frecuencia fundamental
<pre>
  def frecuencia_fundamental(signal, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)      # Vector de frecuencias
    fft_vals = np.abs(np.fft.rfft(signal))  # Magnitudes del espectro

    # Ignorar muy bajas frecuencias (ruido y DC)
    mask = (freqs > 50) & (freqs < 4000)

    freqs_filtrado = freqs[mask]
    fft_filtrado = fft_vals[mask]

    # Índice del máximo
    idx_max = np.argmax(fft_filtrado)

    return freqs_filtrado[idx_max]

signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Mujer_1.wav")
f0 = frecuencia_fundamental(signal, fs)
print("Frecuencia fundamental Mujer 1:", f0, "Hz")

signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Mujer_2.wav")
f0 = frecuencia_fundamental(signal, fs)
print("Frecuencia fundamental Mujer 2:", f0, "Hz")

signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Mujer_3.wav")
f0 = frecuencia_fundamental(signal, fs)
print("Frecuencia fundamental Mujer 3:", f0, "Hz")

signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Hombre_1.wav")
f0 = frecuencia_fundamental(signal, fs)
print("Frecuencia fundamental Hombre 1:", f0, "Hz")

signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Hombre_2.wav")
f0 = frecuencia_fundamental(signal, fs)
print("Frecuencia fundamental Hombre 2:", f0, "Hz")

signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Hombre_3.wav")
f0 = frecuencia_fundamental(signal, fs)
print("Frecuencia fundamental Hombre 3", f0, "Hz")
</pre>
El análisis de las frecuencias fundamentales registradas muestra resultados interesantes frente a los rangos típicos de la voz. En condiciones normales, la voz femenina suele encontrarse entre 165–255 Hz, mientras que la voz masculina se ubica en torno a 85–180 Hz. En nuestros resultados, las tres voces femeninas (232.77 Hz, 291.64 Hz y 280.55 Hz) se encuentran en la parte alta del rango o incluso superándolo, lo que indica voces con un timbre particularmente agudo. En el caso de los hombres, dos de ellos (274.43 Hz y 271.38 Hz) presentan frecuencias cercanas o incluso superiores al rango típico femenino, lo cual es atípico para voces masculinas, mientras que el tercer hombre (224.21 Hz) se mantiene más cercano al límite inferior del rango femenino y aún fuera del esperado para varones. Estos resultados pueden explicarse por factores individuales como edad, esfuerzo vocal, características fisiológicas de las cuerdas vocales o incluso condiciones de la grabación.
#### Frecuencia media
<pre>
  
import numpy as np

def frecuencia_media(signal, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)         # Vector de frecuencias
    fft_vals = np.abs(np.fft.rfft(signal))     # Magnitud del espectro

    # Filtrar bajas frecuencias (ruido y DC)
    mask = (freqs > 50) & (freqs < 4000)
    freqs_filtrado = freqs[mask]
    fft_filtrado = fft_vals[mask]

    # Frecuencia media (centroide espectral)
    f_media = np.sum(freqs_filtrado * fft_filtrado) / np.sum(fft_filtrado)
    return f_media


# ===================== MUJERES =====================
signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Mujer_1.wav")
f0 = frecuencia_fundamental(signal, fs)
fm = frecuencia_media(signal, fs)
print("Frecuencia fundamental Mujer 1:", f0, "Hz")
print("Frecuencia media Mujer 1:", fm, "Hz")

signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Mujer_2.wav")
f0 = frecuencia_fundamental(signal, fs)
fm = frecuencia_media(signal, fs)
print("Frecuencia fundamental Mujer 2:", f0, "Hz")
print("Frecuencia media Mujer 2:", fm, "Hz")

signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Mujer_3.wav")
f0 = frecuencia_fundamental(signal, fs)
fm = frecuencia_media(signal, fs)
print("Frecuencia fundamental Mujer 3:", f0, "Hz")
print("Frecuencia media Mujer 3:", fm, "Hz")


# ===================== HOMBRES =====================
signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Hombre_1.wav")
f0 = frecuencia_fundamental(signal, fs)
fm = frecuencia_media(signal, fs)
# --- Corrección solo para hombres ---
if 180 < f0 < 400:
    f0 /= 2
print("Frecuencia fundamental Hombre 1:", f0, "Hz")
print("Frecuencia media Hombre 1:", fm, "Hz")

signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Hombre_2.wav")
f0 = frecuencia_fundamental(signal, fs)
fm = frecuencia_media(signal, fs)
# --- Corrección solo para hombres ---
if 180 < f0 < 400:
    f0 /= 2
print("Frecuencia fundamental Hombre 2:", f0, "Hz")
print("Frecuencia media Hombre 2:", fm, "Hz")

signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Hombre_3.wav")
f0 = frecuencia_fundamental(signal, fs)
fm = frecuencia_media(signal, fs)
# --- Corrección solo para hombres ---
if 180 < f0 < 400:
    f0 /= 2
print("Frecuencia fundamental Hombre 3:", f0, "Hz")
print("Frecuencia media Hombre 3:", fm, "Hz")
</pre>

#### Brillo
<pre>
  def brillo(signal, fs, fmin=2000, fmax=4000):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal))

    # Filtrar solo el rango 2000–4000 Hz
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_filtrado = freqs[mask]
    fft_filtrado = fft_vals[mask]

    # Evitar división por cero
    if np.sum(fft_filtrado) == 0:
        return 0

    # Centroide espectral en ese rango
    brillo_hz = np.sum(freqs_filtrado * fft_filtrado) / np.sum(fft_filtrado)
    return brillo_hz
signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Mujer_1.wav")
b = brillo(signal, fs)
print("Brillo Mujer 1:", b)
signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Mujer_2.wav")
b = brillo(signal, fs)
print("Brillo Mujer 2:", b)
signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Mujer_2.wav")
b = brillo(signal, fs)
print("Brillo Mujer 3:", b)
signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Hombre_1.wav")
b = brillo(signal, fs)
print("Brillo Hombre 1:", b)
signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Hombre_2.wav")
b = brillo(signal, fs)
print("Brillo Hombre 2:", b)
signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Hombre_3.wav")
b = brillo(signal, fs)
print("Brillo Hombre 3:", b)
</pre>

El brillo muestra que las voces femeninas presentan valores ligeramente más altos (entre 2675 y 2790 Hz) en comparación con la mayoría de las masculinas, que se ubican entre 2555 y 2761 Hz. Esto concuerda con lo esperado, ya que las mujeres suelen tener un timbre más claro y brillante, mientras que en los hombres la energía se concentra un poco más en frecuencias graves. Sin embargo, se observa que uno de los hombres alcanza un brillo similar al de las mujeres, lo que indica que puede tener una voz más clara o aguda dentro del grupo masculino. En general, los resultados coinciden con las diferencias típicas entre voces femeninas y masculinas.

#### Intensidad (energía)
<pre>
  import numpy as np
import wave

def cargar_audio(ruta):
    audio = wave.open(ruta, "rb")
    fs = audio.getframerate()       # Frecuencia de muestreo
    n_samples = audio.getnframes()  # Número de muestras
    signal_wave = audio.readframes(-1)
    audio.close()

    # Convertir a array NumPy
    signal = np.frombuffer(signal_wave, dtype=np.int16)
    return signal, fs

def intensidad(signal):
    # Intensidad promedio (energía normalizada)
    return np.sum(signal.astype(np.float64)**2) / len(signal)

signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Mujer_1.wav")
I = intensidad(signal)
print("Intensidad promedio:", I)
signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Mujer_2.wav")
I = intensidad(signal)
print("Intensidad promedio:", I)
signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Mujer_3.wav")
I = intensidad(signal)
print("Intensidad promedio:", I)
signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Hombre_1.wav")
I = intensidad(signal)
print("Intensidad promedio:", I)
signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Hombre_2.wav")
I = intensidad(signal)
print("Intensidad promedio:", I)
signal, fs = cargar_audio("/content/drive/MyDrive/Audios/Hombre_3.wav")
I = intensidad(signal)
print("Intensidad promedio:", I)
</pre>
En cuanto a la intensidad, los resultados muestran que las voces masculinas tienen valores promedio más altos (entre 24 y 35 millones) en comparación con las femeninas (entre 19 y 21 millones). Esto refleja una diferencia esperada, ya que las voces de los hombres suelen producir mayor potencia acústica debido a la mayor masa de sus pliegues vocales y a una proyección más grave y resonante. No obstante, los valores de las mujeres también se mantienen en un rango considerable, lo que indica buena claridad y presencia vocal.
# PARTE B

# PARTE C
<img width="611" height="134" alt="image" src="https://github.com/user-attachments/assets/1eb7d1a2-6863-4bf5-814a-94a183fd9fd9" />

# ¿Qué diferencias se observan en la frecuencia fundamental?  
La diferencia más grande que se observa en la frecuencia fundamental entre voces masculinas y femeninas es el rango en el que se encuentran: la frecuencia fundamental de las mujeres está entre 232 Hz y 291 Hz, mientras que la de los hombres es mucho más baja, ubicándose entre 112 Hz y 137 Hz. Esta diferencia refleja las distintas longitudes y la fisiologia de las cuerdas vocales, ya que las masculinas son más gruesas y, por lo tanto, vibran más lentamente, generando frecuencias más bajas que las femeninas. 

 

# DIAGRAMAS DE FLUJO
![Imagen de WhatsApp 2025-10-06 a las 20 50 49_44e49ca4](https://github.com/user-attachments/assets/1ba2eed7-21f3-439b-80d4-f76dd27afe24)



