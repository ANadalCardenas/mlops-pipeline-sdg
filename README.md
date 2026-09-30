# MLOps SDG Pipeline

Este proyecto entrena un modelo que adivina a qué grupo (A, B, C o D) pertenece cada fila de una tabla de datos. Además, lo hace de forma automática y controlada: cada vez que alguien propone un cambio, el sistema entrena un modelo nuevo, lo compara con el que ya se usa y solo lo pone en uso si es mejor.

---

## 1. Qué se adivina

Los datos están en `data/v1/sdg.csv`: una tabla de 500 filas. La última columna, `Target`, es lo que se quiere adivinar. Tiene 4 valores posibles:

| Grupo | Filas | % del total |
|---|---|---|
| A | 164 | 33% |
| B | 155 | 31% |
| C | 157 | 31% |
| D | 24 | 5% (hay muy pocas) |


---

## 2. Estudio de los datos

Antes de entrenar, se han revisado las 31 columnas (`Timestamp` y de `F1` a `F30`) para decidir cuáles usar.

### La calidad de los datos es buena

- No hay filas repetidas.
- No hay valores extraños o exagerados.
- Casi todas las columnas tienen solo un 1% de valores vacíos.

### Observación importante

Parece ser que la columna `F22` es la única que ayuda a adivinar el grupo. Cuanto más alto es su valor, más "alta" es la letra:

| Si `F22` vale más o menos... | ...casi siempre es el grupo |
|---|---|
| 10 | A |
| 19 | B |
| 31 | C |
| 40 | D |

No es perfecto: entre grupos vecinos (A y B, B y C, C y D) hay casos que se mezclan. Pero nunca se confunde A con D, porque están muy lejos.

Parece que las otras columnas no tienen relación con el grupo. 

### Columnas que se han decidido quitar:

| Columna | Por qué se quita |
|---|---|
| `Timestamp` (fecha y hora) | El mes coincide casi siempre con el grupo (enero = A, abril = B, julio = C, octubre = D). Parece un efecto de cómo se crearon los datos, no algo real. Si se usara, el modelo parecería perfecto en las pruebas y fallaría en la vida real |
| `F17` | Está vacía en el 90% de las filas. No se puede aprender de una columna casi vacía |
| `F4`, `F7`, `F8`, `F11`, `F13` | Son nombres de profesiones ("Nurse", "Pilot"...), con más de 100 valores distintos. Se decide quitarlas porque no tienen relación con el grupo y solo añaden ruido |

### Columnas que se usan:

El modelo usa las otras 24 columnas: `F22` y otras 23 que, aunque no ayudan, tampoco dan problemas claros. Son estas:

| Columnas | Qué contienen |
|---|---|
| `F22` | La importante |
| `F1`, `F2`, `F14`, `F16`, `F19` | Estados: `success`, `warning`, `error` o `unknown` |
| `F3`, `F6`, `F9`, `F12`, `F18`, `F21`, `F24`, `F27` | Números, casi siempre 0 |
| `F5`, `F10`, `F15`, `F20`, `F25`, `F30` | Números que se parecen mucho entre sí (son casi la misma columna repetida) |
| `F23`, `F26`, `F28`, `F29` | Números sin relación con el grupo |

Por qué no se ha decidido usar solo la `F22`, si es la única útil: con solo `F22` el modelo acierta un poco más, pero depender de una sola columna es arriesgado. Si un día `F22` llega vacía o cambia su forma de medirse, el modelo dejaría de funcionar. Antes de dar ese paso hay que saber qué es `F22` y si siempre estará disponible.

### Cuánto acierta el modelo

| Columnas usadas | Aciertos |
|---|---|
| Todas (menos `Timestamp`) | 67 de cada 100 |
| Las 24 que se usan ahora | 72 de cada 100 |
| Solo `F22` | 77 de cada 100 |

El grupo D es el punto débil: como hay tan pocas filas, el modelo casi nunca lo acierta.


## 3. Cómo se mide si el modelo es bueno

| Medida | Qué significa |
|---|---|
| Accuracy | De cada 100 filas, cuántas acierta |
| F1 | Una nota que combina "cuántas acierta" y "cuántas se le escapan". Se calcula para cada grupo y se hace la media, así el grupo D cuenta igual que los demás aunque tenga menos filas |
| ROC AUC | Si el modelo está "seguro" cuando acierta. De 0,5 (adivina al azar) a 1 (perfecto) |
| Recall D | De las filas que son D, cuántas detecta. Se muestra aparte para que no quede escondido |

Un modelo nuevo solo se considera mejor si su F1 y su ROC AUC son iguales o más altos que los del modelo actual.

---

## 4. Cómo funciona el proceso automático

1. Alguien propone un cambio en GitHub (un *pull request*).
2. GitHub entrena dos modelos: uno con el cambio y otro con la versión actual.
3. Los compara y escribe un comentario en el *pull request* con una tabla de resultados.
4. Si el modelo nuevo es peor, la comprobación sale en rojo: no se debe aprobar el cambio.
5. Si es mejor, se marca como candidato (Staging).
6. Cuando se aprueba el cambio, el candidato pasa a ser el modelo en uso (Production) automáticamente.

Dónde se guarda cada cosa:

| Servicio | Qué guarda |
|---|---|
| GitHub | El código |
| Cloudflare R2 | Los datos (con DVC, que guarda cada versión sin borrar las anteriores) |
| DagsHub | Los modelos entrenados, sus resultados y cuál está en uso |

Versiones de los datos: `v1` son los datos originales. `v2` es una versión creada para simular que llegan datos nuevos (más filas, más grupo D y pequeños cambios en `F22` y `F1`).

---

## 5. Cómo ejecutarlo

Todos los comandos se ejecutan desde la carpeta del proyecto.

### Preparar el proyecto (solo la primera vez)

```bash
git clone git@github.com:ANadalCardenas/mlops-pipeline-sdg.git
cd mlops-pipeline-sdg

# Crear el entorno de Python e instalar lo necesario
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Poner las claves de Cloudflare R2 y descargar los datos
dvc remote modify storage --local access_key_id ACCESS_KEY_ID_DE_R2
dvc remote modify storage --local secret_access_key SECRET_ACCESS_KEY_DE_R2
dvc pull
```

Para usar DagsHub, hay que crear un fichero `.env` en la carpeta del proyecto con estas tres líneas:

```
MLFLOW_TRACKING_URI=https://dagshub.com/aina.nadal/mlops-pipeline-sdg.mlflow
MLFLOW_TRACKING_USERNAME=usuario_de_dagshub
MLFLOW_TRACKING_PASSWORD=token_de_dagshub
```

### Comprobar que todo funciona

```bash
source .venv/bin/activate
pytest
```

### Entrenar un modelo en local

No toca DagsHub ni el modelo en uso: los resultados se quedan en local.

```bash
source .venv/bin/activate
python pipelines/orchestration.py --data-version v1 --experiment-name sdg-local --run-name prueba --output-dir reports/local
```

Los resultados quedan en `reports/local/`: las medidas en `train_metrics.json` y las gráficas en `figures/`.

### Hacer una predicción

Usa el modelo que está en uso (Production) en DagsHub. Necesita el fichero `.env` y Docker.

```bash
# Crear la imagen (solo la primera vez, o si cambia el código)
docker build -f Dockerfile.inference -t sdg-inference .

# Hacer una predicción
mkdir -p predictions
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp --env-file .env \
  -v "$(pwd)/predictions:/app/predictions" \
  sdg-inference \
  --input '{"F1": "error", "F2": "success", "F3": 0.0, "F5": 0.3418, "F6": 0.0, "F9": 0.0, "F10": 0.5333, "F12": 0.0, "F14": "success", "F15": 0.7687, "F16": "success", "F18": 0.0, "F19": "unknown", "F20": null, "F21": 0.0, "F22": 17.7918, "F23": -73.3180, "F24": 0.0, "F25": 0.2662, "F26": 103.2730, "F27": 0.0, "F28": 108.1766, "F29": -98.5651, "F30": 0.1691}'
```

El resultado sale por pantalla (`"prediction": "B"`) y se guarda en `predictions/predictions.db`.



### Ver las predicciones guardadas

```bash
source .venv/bin/activate
python -c "import pandas as pd; print(pd.read_sql('SELECT * FROM predictions', 'sqlite:///predictions/predictions.db'))"
```

---

## 6. Carpetas del proyecto

| Carpeta | Qué hay |
|---|---|
| `src/` | El código: cargar datos, preparar columnas, entrenar, evaluar y predecir |
| `pipelines/orchestration.py` | El programa que ejecuta todo el entrenamiento |
| `tests/` | Las pruebas automáticas |
| `data/` | Los punteros a los datos (`v1.dvc`, `v2.dvc`) y el script que crea `v2` |
| `.github/workflows/` | Los procesos automáticos de GitHub |
| `Dockerfile.inference` | Las instrucciones para crear la imagen de predicción |
