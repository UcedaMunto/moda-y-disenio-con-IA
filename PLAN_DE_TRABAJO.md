# Plan de trabajo - Fabric2Mesh AI

## 1) Entendimiento del objetivo
Objetivo principal del proyecto:
- Vestir un modelo 3D con texturas de tela generadas o refinadas desde pocas fotos.
- Entrenar/ajustar un componente de IA para aprender patron/estilo de tela.
- Presentar N variantes de patron al usuario.
- Permitir seleccion humana de cuales variantes se apegan al objetivo y cuales no.

Interpretacion operativa:
- El sistema no genera geometria 3D nueva.
- El sistema toma un mesh base + UV existentes.
- La IA opera en 2D (textura) y Blender aplica la textura al modelo 3D.

## 2) Alcance funcional (MVP)
Incluye:
- Carga de 2 a 10 fotos por tela/proyecto.
- Preproceso basico de imagen (limpieza, recorte, normalizacion de color).
- Generacion de N candidatos de textura (tileable o casi tileable).
- Pantalla o endpoint para seleccionar candidatos validos/no validos.
- Aplicacion automatica de la textura seleccionada al modelo 3D.
- Exportacion de resultado en GLB/OBJ.

No incluye en MVP:
- Simulacion fisica avanzada de tela.
- Retopologia o UV unwrap automatico completo de alta precision.
- Entrenamiento largo de modelos 3D generativos.

## 3) Arquitectura de trabajo propuesta
- Backend API (FastAPI): orquesta flujo, recibe imagenes, devuelve candidatos y export.
- Motor de textura (host AMD/ROCm): genera variantes de patron con diffusion + LoRA.
- Motor 3D (Blender headless): aplica textura, ajusta escala, exporta modelo.
- Infra soporte (Docker): API, DB, MinIO, MLflow.

## 4) Flujo funcional end-to-end
1. Usuario crea un proyecto y sube fotos de referencia.
2. Pipeline preprocesa fotos.
3. Motor IA genera N texturas candidatas.
4. Sistema renderiza previews 2D/3D rapidas.
5. Usuario etiqueta cada candidato: APROBAR o RECHAZAR.
6. Sistema toma aprobadas y aplica a mesh destino.
7. Exporta GLB/OBJ y guarda trazabilidad de seleccion.

## 5) Fases y entregables

### Fase 0 - Base tecnica
Entregables:
- Estructura de carpetas real del repo.
- Entorno local + Docker de servicios funcionando.
- Scripts minimos ejecutables.
Criterio de salida:
- API arriba y responde health check.

### Fase 1 - Texturizado 3D directo (sin IA compleja)
Entregables:
- Script Blender para aplicar textura fija al modelo.
- Export GLB/OBJ validado.
Criterio de salida:
- Modelo exportado con textura visible y escala controlable.

### Fase 2 - Generacion de candidatos de patron
Entregables:
- Script de generacion de N texturas desde referencia.
- Metodo de mejora seamless/tileable.
Criterio de salida:
- N candidatos generados por corrida con calidad aceptable.

### Fase 3 - Seleccion humana y ranking
Entregables:
- Endpoint/UI para etiquetar APROBAR/RECHAZAR.
- Persistencia de decisiones por proyecto.
Criterio de salida:
- Usuario puede elegir rapidamente cuales patrones se apegan y cuales no.

### Fase 4 - Entrenamiento iterativo
Entregables:
- Uso de feedback humano para mejorar siguientes generaciones.
- Versionado de experimentos en MLflow.
Criterio de salida:
- Mejora medible en tasa de aprobacion.

## 6) Metricas de exito
- Tasa de aprobacion de candidatos por lote.
- Tiempo de extremo a extremo por proyecto.
- Porcentaje de texturas sin repeticion visual evidente.
- Numero de iteraciones necesarias hasta seleccion final.

## 7) Riesgos y mitigacion
- Distorsion UV: validar UV del mesh y agregar controles de escala/rotacion.
- Repeticion visible: forzar pipeline seamless y revisar offset tests.
- Deriva de color: normalizacion y control de iluminacion en preproceso.
- Cuello de GPU: separar colas y cache de resultados intermedios.

## 8) Plan de ejecucion inicial (2 semanas)
Semana 1:
- Cerrar definiciones de entrada/salida.
- Implementar pipeline minimo: carga -> textura -> Blender -> export.
- Producir primeras 5-10 muestras de patron por tela.

Semana 2:
- Agregar seleccion APROBAR/RECHAZAR.
- Persistir feedback y versionar corridas.
- Ajustar calidad de seamless y escala en modelo final.

## 9) Definition of Done del MVP
- Se suben fotos de tela.
- El sistema genera N candidatos.
- El usuario selecciona validos/no validos.
- Se aplica al modelo 3D.
- Se exporta GLB/OBJ usable.
