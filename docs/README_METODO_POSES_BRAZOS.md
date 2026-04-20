# Metodo de Poses de Brazos

Este documento describe como el sistema identifica poses de brazos, como usa proporcion aurea para consistencia de tamano y como entrena en curriculum (primero poses que mejor coinciden, luego las mas dificiles).

## 0) Modelo oficial de IA para pose

Para identificacion de pose en Human Shape Lab, la documentacion oficial del proyecto define:

- Modelo de pose base: Google Pose Landmarker (MediaPipe Tasks).
- Modelo propio complementario: refinador entrenable de keypoints (`keypoint_refiner_v1`).

Convivencia de ambos modelos:

- `pose_backend=landmarker` usa Google para la pose base.
- `apply_refiner=true` aplica el modelo propio encima de la base.
- `apply_refiner=false` deja la salida base (Google puro).
- `pose_backend=legacy` permanece disponible por compatibilidad/fallback.

## 1) Objetivo

Mejorar reconocimiento y ajuste de brazos en fotos de cuerpo completo, con estas reglas:

- Evaluar por seccion (brazo, antebrazo, mano) en cada iteracion.
- Corregir solo secciones con error alto.
- Mantener consistencia antropometrica con prior de proporcion aurea.
- Si longitud de brazos no encaja con proporcion aurea y el codo esta cerrado, tratarlo como brazo doblado (no estirar forzadamente).

## 2) Poses detectadas en modelos guardados

Analisis automatico sobre muestras guardadas con keypoints (24 muestras):

- 6: `left_down_bent | right_down_bent`
- 5: `left_down_straight | right_down_straight`
- 3: `left_down_straight | right_down_bent`
- 2: `left_up_bent | right_up_bent`
- 2: `left_front_bent | right_side_bent`
- 1: `left_down_straight | right_front_bent`
- 1: `left_down_bent | right_down_straight`
- 1: `left_front_bent | right_down_bent`
- 1: `left_down_bent | right_side_bent`
- 1: `left_side_bent | right_side_bent`
- 1: `left_front_bent | right_front_bent`

Estas clases se usan para construir prototipos de pose de brazos.

## 3) Deteccion de pose de brazos

Para cada lado (`left`, `right`) se calculan:

- Angulo de codo (hombro-codo-muneca).
- Orientacion aproximada de la muneca respecto al hombro (`up`, `side`, `front`, `down`).
- Estado del brazo: `straight` si angulo >= 150 grados, si no `bent`.

Etiqueta final por lado:

- `left_up_bent`, `left_down_straight`, etc.

Etiqueta de par:

- `left_* | right_*`.

## 4) Prior de brazos (mejora de reconocimiento)

Se aplica despues del refinador principal:

1. Busca el prototipo de brazos mas cercano (promedio de keypoints de brazo por clase).
2. Mezcla suavemente (`arm_pose_alpha`) la prediccion con ese prototipo.
3. Verifica consistencia de longitud de brazo con proporcion aurea:
   - `torso_h = hip_center.y - neck_base.y`
   - `upper_expected = torso_h / phi`
   - `lower_expected = upper_expected / phi`
4. Si hay desajuste fuerte de longitud y codo doblado (`angulo < 150`), se marca como doblado y no se estira.
5. Si el brazo esta recto y muy fuera de longitud esperada, se normaliza suavemente.

## 5) Entrenamiento por curriculum de poses

Para acelerar convergencia y robustez:

- Se calcula error inicial de brazos por muestra.
- Se ordenan muestras de menor a mayor error (mejor coincidencia primero).
- En cada epoca se usa una fraccion creciente de muestras (`curriculum_start_fraction` -> 1.0).
- Esto permite entrenar primero poses mas alineadas y luego incorporar poses mas dificiles.

## 6) Parametros API relevantes

Endpoint: `POST /human-shape-lab/keypoints/iterate`

- `update_only_erroneous_sections`: corrige solo secciones con error alto.
- `section_error_threshold`: umbral de error por seccion.
- `enable_golden_ratio_prior`: activa prior global de proporcion.
- `golden_ratio_alpha`: fuerza del prior global.
- `enable_arm_pose_prior`: activa prior especifico de brazos.
- `arm_pose_alpha`: fuerza del prior de brazos.
- `enable_arm_pose_curriculum`: activa curriculum por poses.
- `curriculum_start_fraction`: fraccion inicial de muestras en curriculum.

## 7) Flujo recomendado

1. Guardar snapshot objetivo con keypoints corregidos.
2. Ejecutar iteraciones cortas (1-3 epocas) con curriculum activo.
3. Revisar `sections_updated_counts` y `section_improvement`.
4. Corregir manualmente solo peores partes en UI.
5. Repetir.

## 8) Criterio de exito

- Reduccion estable de `mean_l2` global.
- Mejora por seccion en `left/right_arm`, `left/right_forearm`, `left/right_hand`.
- Menor variabilidad de tamano de brazos entre fotos comparables.
