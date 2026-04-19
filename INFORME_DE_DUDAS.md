# Informe de dudas y preguntas de definicion

## Estado de entendimiento
El objetivo general esta claro:
- Aprender patrones de tela desde imagenes.
- Generar multiples candidatos.
- Permitir seleccion humana de cuales se apegan al objetivo.
- Aplicar la textura elegida sobre un modelo 3D y exportar.

## Dudas clave a resolver

1. Que significa exactamente entrenar al modelo en este proyecto?
- Opcion A: entrenar/ajustar LoRA para generar mejores texturas 2D.
- Opcion B: entrenar un ranking con feedback APROBAR/RECHAZAR.
- Opcion C: ambas.

2. Cual sera el valor de N por defecto para candidatos de patron?
- Ejemplo: 4, 8, 12 o variable por proyecto.

3. Seleccion humana: como se captura?
- Solo APROBAR/RECHAZAR.
- O tambien score 1-5 y comentarios.

4. Salida visual para elegir candidatos:
- Solo preview 2D de textura.
- Preview 2D + render rapido sobre modelo 3D.

5. Tipo de modelo 3D objetivo:
- Avatar humano completo, prenda separada, o ambos.
- Formato base principal: OBJ, FBX o GLB.

6. Necesitas mapas PBR en MVP?
- Solo albedo/diffuse.
- O tambien normal, roughness, metallic.

7. Restricciones de tiempo por corrida:
- Tiempo maximo aceptable para generar N candidatos.

8. Requisitos de resolucion de textura:
- 1024, 2048, 4096.

9. Calidad minima esperada:
- Criterios objetivos para decir se apega o no se apega.

10. Prioridad de interfaz:
- API primero.
- UI web minima para seleccionar patrones desde el inicio.

## Supuestos de trabajo (si no se define aun)
- N inicial = 8 candidatos.
- Seleccion binaria APROBAR/RECHAZAR.
- MVP con mapa albedo 2K.
- Input mesh en OBJ/GLB con UV validas.
- API primero y UI minima en segunda iteracion.

## Decision requerida para arrancar sin bloqueo
Para comenzar implementacion real sin retrabajo, se necesita tu confirmacion de:
- Definicion de entrenar.
- Valor inicial de N.
- Formato 3D principal.
- Nivel de mapas PBR para MVP.
- Si incluimos UI de seleccion desde la primera entrega.
