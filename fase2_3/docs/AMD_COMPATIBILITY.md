# Compatibilidad AMD para OpenTryOn en esta maquina

## Resumen

La documentacion oficial de OpenTryOn deja la inferencia local de GPU orientada a CUDA. En este host se detecta una GPU AMD integrada, pero no existe un runtime ROCm utilizable ni un entorno PyTorch listo para HIP/ROCm.

Resultado operativo actual:

- `local_gpu_ready = false`
- `recommended_mode = api_cpu_only`

## Evidencia local comprobada

- GPU detectada: `AMD/ATI Cezanne [Radeon Vega Series]`
- `rocminfo`: no disponible
- `hipconfig`: no disponible
- `.venv/bin/python`: existe, pero sin `torch` instalado

## Implicacion practica

En esta maquina, hoy podemos preparar OpenTryOn de dos formas realistas:

1. usar modos API del proyecto
2. usar CPU para componentes ligeros o de preparacion

No es razonable prometer los modelos locales documentados por OpenTryOn hasta instalar y validar:

- ROCm en el host
- PyTorch compilado para ROCm/HIP
- backend GPU visible desde `torch`

## Comandos de verificacion

```bash
python fase2_3/scripts/preflight_amd.py
python fase2_3/scripts/preflight_amd.py --json
python fase2_3/scripts/bootstrap_opentryon.py --dry-run
```

## Criterio de cambio de estado

Solo se debe cambiar a una recomendacion `local_gpu_or_api` si el preflight devuelve simultaneamente:

- GPU AMD detectada
- ROCm runtime presente
- `torch` instalado
- `torch.version.hip` no vacio
- `torch.cuda.is_available() == True`