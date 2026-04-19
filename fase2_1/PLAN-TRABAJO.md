# Plan de Trabajo - Fase 2.1

## Objetivo

Construir un MVP de Virtual Try-On para imagen estatica que sea funcional, evaluable y entrenable con herramientas gratuitas.

---

## Duracion estimada

4 semanas (iteracion corta con entregables semanales)

---

## Entregables finales de la 2.1

1. Endpoint API para try-on de imagen.
2. Pipeline base en core/tryon con pose, alineacion y overlay.
3. Base de entrenamiento para segmentacion y ajuste geometrico.
4. Script de evaluacion basico con metricas de calidad visual.
5. Documento de resultados y decisiones para pasar a 2.2.

---

## Cronograma

## Semana 1 - Base funcional

Objetivo:

- Dejar pipeline minimo ejecutable de imagen estatica.

Tareas:

1. [x] Crear modulos core/tryon (pose, align, overlay, pipeline).
2. [x] Definir schema de entrada/salida para inferencia.
3. [x] Exponer endpoint try-on en API.
4. [x] Preparar dataset interno de validacion (20 a 50 imagenes).
5. [x] Exponer endpoint batch con reporte estable (`/fase2_1/tryon/batch`).
6. [x] Agregar test de endpoint run y batch.

Criterios de aceptacion:

1. [x] El endpoint devuelve imagen de salida para un caso baseline.
2. [x] No rompe endpoints actuales del proyecto.
3. [x] Batch produce reporte JSON con `summary` y `results`.

---

## Semana 2 - Segmentacion y oclusion

Objetivo:

- Mejorar realismo con mascara y composicion.

Tareas:

1. [~] Integrar segmentacion open source preentrenada.
2. [x] Agregar mascara de oclusion simple (brazos/torso).
3. Mejorar overlay con alpha y orden de capas.
4. [x] Definir metrica visual interna por checklist.

Progreso actual de inicio Semana 2:

- [x] Contrato de segmentacion incorporado en pipeline (`parsing.py` baseline).
- [x] Metadata de segmentacion propagada en salida de `run_tryon`.
- [x] Funcion de overlay con oclusion por mascara incorporada y testeada.
- [x] Checklist de calidad visual versionado en `fase2_1/config/quality_checklist.json`.

Criterios de aceptacion:

1. Menos artefactos visuales vs baseline semana 1.
2. Mejora visible en casos con brazos cruzados o parciales.

---

## Semana 3 - Entrenamiento inicial

Objetivo:

- Empezar entrenamiento real de componentes.

Tareas:

1. Pipeline de datos para fine-tuning de segmentacion.
2. Estructura de entrenamiento de ajuste geometrico (fit_regression).
3. Registro de experimentos con MLflow.
4. Primer experimento entrenado reproducible.

Criterios de aceptacion:

1. Al menos 1 corrida completa de entrenamiento registrada.
2. Mejora medible en metrica acordada.

---

## Semana 4 - Estabilizacion y cierre

Objetivo:

- Consolidar MVP 2.1 y preparar 2.2.

Tareas:

1. Hardening de API y manejo de errores.
2. Benchmark de tiempo por imagen.
3. Documentar limites conocidos y backlog 2.2.
4. Demo interna con casos representativos.

Progreso actual Semana 4:

- [x] Migracion de startup API principal a `lifespan` (sin `on_event` deprecado).
- [x] Script de benchmark de latencia por imagen (`fase2_1/scripts/benchmark_latency.py`).
- [x] Documentacion de limites conocidos y flujo de fallback en README.

Criterios de aceptacion:

1. Tiempo por imagen <= 2.5s en baseline CPU.
2. [x] Tasa de casos visualmente validos >= 90 por ciento en set interno (mecanismo de medicion implementado).

---

## Backlog tecnico 2.1

1. [x] Agregar tests para core/tryon.
2. [x] Versionar configuraciones de inferencia.
3. [x] Añadir selector de categoria de prenda para ajustes por tipo.
4. [x] Definir formato canonico de landmarks y transforms.

---

## Riesgos y mitigacion

1. Riesgo: variacion alta de poses no frontales.
   - Mitigacion: set de validacion con poses diversas y reglas de fallback.

2. Riesgo: composicion poco realista por falta de oclusion.
   - Mitigacion: priorizar segmentacion y capas antes de video.

3. Riesgo: sobrecarga por modelos pesados.
   - Mitigacion: baseline liviano + perfiles de inferencia.

---

## Definicion de terminado (DoD)

La Fase 2.1 se considera terminada cuando:

1. El pipeline funciona de punta a punta en imagen estatica.
2. Hay metrica de calidad y latencia reportada.
3. Existe al menos un componente entrenado/fine-tuned.
4. El sistema queda listo para iniciar 2.2 (batch/video).
