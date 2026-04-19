# Fase 2.1 Config

Configuraciones versionadas de inferencia y experimentos de Fase 2.1.

Archivo base:

- `inference.yaml`: parametros baseline del pipeline de try-on.

Uso sugerido:

1. Duplicar `inference.yaml` por experimento (`inference-exp01.yaml`).
2. Registrar el nombre del archivo en MLflow o en el reporte de evaluacion batch.
