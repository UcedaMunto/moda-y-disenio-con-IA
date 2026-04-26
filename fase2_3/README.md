# Fase 2.3 - Toolkit OpenTryOn

Toolkit aislado para preparar una integracion de OpenTryOn orientada a previsualizacion de prendas.

Objetivos de esta carpeta:

- mantener la integracion fuera del resto del repo
- validar compatibilidad local con GPU AMD antes de instalar modelos pesados
- preparar un bootstrap reproducible del repositorio `tryonlabs/opentryon`
- dejar una ruta segura para modos API/CPU cuando no exista soporte ROCm utilizable

## Estructura

- `config/`: descubrimiento de rutas y ajustes locales
- `core/`: logica de compatibilidad AMD/ROCm y bootstrap
- `docs/`: notas operativas de compatibilidad
- `scripts/`: CLIs para preflight y bootstrap

## Uso rapido

1. Validar el host actual:

```bash
python fase2_3/scripts/preflight_amd.py
```

2. Ver el plan de bootstrap sin tocar nada:

```bash
python fase2_3/scripts/bootstrap_opentryon.py --dry-run
```

3. Clonar OpenTryOn dentro de esta fase:

```bash
python fase2_3/scripts/bootstrap_opentryon.py --clone
```

## Ruta minima API/CPU (opcion recomendada en este host)

1. Preparar entorno minimo para API cloud (sin modelos locales GPU):

```bash
bash fase2_3/scripts/setup_opentryon_api_cpu.sh
```

2. Arrancar servidor API de OpenTryOn en local:

```bash
bash fase2_3/scripts/run_opentryon_api_cpu.sh
```

3. Comprobar salud del servicio:

```bash
curl -s http://127.0.0.1:8011/health
```

Respuesta esperada:

```json
{"status":"healthy"}
```

## Criterio de compatibilidad

OpenTryOn documenta inferencia local con GPU para rutas CUDA. En este toolkit se considera que AMD esta lista para uso local solo si se cumplen todos estos puntos:

- se detecta GPU AMD en el host
- existe runtime ROCm (`rocminfo` o `hipconfig`)
- el entorno Python puede importar `torch`
- `torch` expone backend HIP/ROCm utilizable

Si alguno falla, el toolkit recomienda operar en modo `api_cpu_only`.