# Backlog tecnico ejecutable

## Bloque 1 - Fase 0 completada
- [x] Estructura base de carpetas creada.
- [x] API FastAPI con endpoints de health, generate, select, feedback y resumen.
- [x] docker-compose con API, Postgres, MinIO y MLflow.
- [x] Scripts base para generate/apply/export/train + script Blender de aplicacion.
- [x] Endpoint para listar candidatos por proyecto.
- [x] Prueba smoke inicial de API.

## Bloque 2 - Fase 1 (texturizado directo)
- [x] Implementar script Blender real para aplicar textura y exportar GLB.
- [x] Definir contrato de entrada del modelo (OBJ/GLB, escala, UV).
- [x] Extender contrato para soportar FBX en preview/select.
- [ ] Agregar pruebas de regresion visual basicas (renders previos).

## Bloque 3 - Fase 2 (IA texturas)
- [x] Integrar diffusion opcional con fallback baseline en host.
- [x] Integrar soporte opcional para pesos LoRA en pipeline diffusion.
- [x] Generar imagenes reales baseline (sin diffusion) en data/processed.
- [x] Implementar postproceso seamless (offset/blend).

## Bloque 4 - Fase 3 (seleccion humana)
- [x] Endpoint para registrar APROBAR/RECHAZAR por candidato.
- [x] Persistencia en Postgres de decisiones y metadatos.
- [x] Preview rapido 2D para seleccion.
- [x] Preview 3D opcional para seleccion.
- [x] Resumen de feedback por proyecto (conteos + score promedio).
- [x] UI con catalogo automatico de telas y modelos importados.

## Bloque 5 - Fase 4 (mejora iterativa)
- [x] Ranking de candidatos basado en feedback acumulado.
- [x] Tracking opcional en MLflow para eventos de generacion, seleccion y feedback.
- [x] Metricas: tasa de aprobacion y resumen operativo por proyecto.
- [x] Metricas: tiempo por lote y reprocesos.

## Decisiones pendientes (bloqueantes)
- Definicion precisa de entrenar: LoRA, ranking o ambas.
- N de candidatos por defecto.
- Formato principal del modelo (OBJ/GLB/FBX).
- Nivel de PBR para MVP (solo albedo o mapas adicionales).
- Si UI de seleccion entra en primera entrega.
