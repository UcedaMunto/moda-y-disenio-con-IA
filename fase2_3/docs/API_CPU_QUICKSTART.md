# Quickstart API/CPU para OpenTryOn (fase2_3)

Esta guia deja OpenTryOn arrancable en modo API/CPU sobre este host AMD, sin depender de ROCm/CUDA.

## 1) Clonar OpenTryOn dentro de fase2_3

```bash
.venv/bin/python fase2_3/scripts/bootstrap_opentryon.py --clone
```

## 2) Crear entorno minimo e instalar dependencias API

```bash
bash fase2_3/scripts/setup_opentryon_api_cpu.sh
```

El script crea el entorno en:

- `fase2_3/vendor/.venv_api_cpu`

## 3) Levantar servidor local

```bash
bash fase2_3/scripts/run_opentryon_api_cpu.sh
```

Servidor:

- `http://127.0.0.1:8011`

## 4) Probar endpoint de salud

```bash
curl -s http://127.0.0.1:8011/health
```

Respuesta esperada:

```json
{"status":"healthy"}
```

## Notas

- Este flujo no activa modelos locales de inferencia GPU.
- El endpoint `/api/v1/virtual-tryon` requiere credenciales cloud en `.env` dentro de `fase2_3/vendor/opentryon`.
- Para este host, el modo objetivo sigue siendo `api_cpu_only` hasta validar ROCm/HIP con `preflight_amd.py`.
