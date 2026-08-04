# Contexto de migración — TotalSegmentator / máscaras corporales PMPD

Última actualización: 2026-08-04 (Europe/Madrid).

## Snapshot Git

- Repositorio: https://github.com/EnriqueGMEG/TotalSegmentator.git.
- Rama: master.
- Commit base: fd73a64 (origin/master), paquete 2.17.0.
- El commit de migración añade workflows/pmpd_body_mask/ y este documento.
- No se ha hecho push. Entorno Conda, pesos, dataset, máscaras e informes quedan fuera de Git.

## Objetivo

Generar máscaras binarias del cuerpo para CT abdominales venosos de PMPD_v2, excluyendo fondo y camilla. Junto a cada images se crea body_mask; cada salida conserva nombre, shape, affine, qform y sform del CT.

Solo se seleccionan cohortes con images/ cuyo nombre no contiene arterial. Los CT originales no se modifican.

## Auditoría

Raíz original: /local/radiomics/PMPD_v2_data.

Se validaron 782 NIfTI venosos sin errores:

| Cohorte | Casos |
|---|---:|
| Clinica Universidad de Navarra | 10 |
| DPCG | 183 |
| HMSanchinarro | 77 |
| HUCA | 21 |
| Hospital Universitario Miguel Servet | 12 |
| Pangen_OG | 111 |
| RUM | 144 |
| ZZU | 224 |
| Total | 782 |

Se excluyeron 524 estudios de ZZU_arterial, HMSanchinarro_arterial, Pangen_OG_arterial y DPCG_arterial.

## Máscara

La tarea body contiene:

- label 1: body_trunc
- label 2: body_extremities

El runner pide multilabel y convierte label>0 a uint8. Equivale a body.nii.gz, unión de body_trunc y body_extremities. Usa remove_small_blobs=200.

Antes del reemplazo atómico valida:

- shape idéntica;
- affine con tolerancia 1e-4;
- valores 0/1;
- foreground no vacío y fracción plausible;
- header, qform y sform del CT preservados.

El modelo body separa el contorno corporal del fondo/camilla. Se generaron montajes QC de zzu_008 y deben revisarse otra vez tras mover los datos.

## Scripts versionados

- workflows/pmpd_body_mask/segment_venous_body.py: inventario, caso individual, lote reanudable en dos GPU, binarización, validación, logs e informes.
- workflows/pmpd_body_mask/make_qc_montage.py: montaje axial/coronal/sagital con overlay.

Una salida se salta solo si supera validación completa. El temporal de un caso interrumpido se elimina al reintentarlo.

## Estado congelado

El lote se detuvo para migrar. No queda ningún proceso TotalSegmentator activo.

| Cohorte | Válidas |
|---|---:|
| Clinica Universidad de Navarra | 10/10 |
| DPCG | 159/183 |
| ZZU | 1/224 |
| Resto | 0 |
| Total | 170/782 |

Quedan 612 pendientes.

Temporal interrumpido no válido:

/local/radiomics/PMPD_v2_data/DPCG/body_mask/.DPCG_175.nii.gz.multilabel.tmp.nii.gz

No hace falta borrarlo: el runner eliminará solo sus temporales conocidos y repetirá DPCG_175.

batch_status.jsonl solo registra los primeros 36 resultados porque el coordinador terminó antes que dos workers huérfanos. No usarlo como contador. La validación de los NIfTI finales es la fuente de verdad.

## Estado fuera de Git

| Ruta | Tamaño | Acción |
|---|---:|---|
| /local/elopezl/envs/TotalSegmentator | 6.4 GB | Recrear |
| /local/elopezl/totalsegmentator_runtime | 242 MB | Copiar opcionalmente |
| runtime/weights | 233 MB | Copiar o descargar |
| PMPD_v2_data/*/body_mask | 47 MB | Transferir con el dataset |
| PMPD_v2_data | externo | Migrar aparte |

Mover solo los tres repositorios no mueve las 170 máscaras. Copiar body_mask con PMPD_v2_data y verificar checksums.

## Entorno

- Python 3.11.15
- TotalSegmentator 2.17.0 editable
- PyTorch 2.13.0+cu130
- nnUNetv2 2.8.1
- nibabel 5.4.2
- Driver NVIDIA 580.159.03
- Dos RTX 5000 Ada de 32.760 MiB

Recrear:

~~~bash
conda create -y -p /RUTA_NUEVA/envs/TotalSegmentator python=3.11 pip
/RUTA_NUEVA/envs/TotalSegmentator/bin/python -m pip install -e /RUTA_NUEVA/TotalSegmentator
~~~

Descargar solo body:

~~~bash
export TOTALSEG_HOME_DIR=/RUTA_RUNTIME/setup
export TOTALSEG_WEIGHTS_PATH=/RUTA_RUNTIME/weights
/RUTA_NUEVA/envs/TotalSegmentator/bin/totalseg_download_weights -t body
~~~

No reutilizar un entorno copiado entre workstations.

## Reanudación

1. Transferir PMPD_v2_data con body_mask.
2. Recrear entorno e instalar este checkout.
3. Descargar/copiar Dataset299_body_1559subj.
4. Ejecutar inventario; debe mostrar 782 venosos.
5. Revisar una máscara existente y un caso nuevo con QC.
6. Confirmar memoria GPU y ausencia de interferencia con trabajos ajenos.
7. Ejecutar dentro de tmux/screen o supervisor persistente.
8. Validar 782/782 y ausencia de temporales.

Inventario:

~~~bash
/RUTA_NUEVA/envs/TotalSegmentator/bin/python \
  workflows/pmpd_body_mask/segment_venous_body.py \
  --dataset /RUTA_NUEVA/PMPD_v2_data \
  --runtime /RUTA_RUNTIME \
  --weights /RUTA_RUNTIME/weights \
  --inventory
~~~

Prueba:

~~~bash
/RUTA_NUEVA/envs/TotalSegmentator/bin/python \
  workflows/pmpd_body_mask/segment_venous_body.py \
  --dataset /RUTA_NUEVA/PMPD_v2_data \
  --runtime /RUTA_RUNTIME \
  --weights /RUTA_RUNTIME/weights \
  --case /RUTA_NUEVA/PMPD_v2_data/ZZU/images/zzu_008.nii.gz \
  --gpu 0
~~~

Reanudar con dos GPU:

~~~bash
/RUTA_NUEVA/envs/TotalSegmentator/bin/python \
  workflows/pmpd_body_mask/segment_venous_body.py \
  --dataset /RUTA_NUEVA/PMPD_v2_data \
  --runtime /RUTA_RUNTIME \
  --weights /RUTA_RUNTIME/weights \
  --gpus 0 1 \
  --all
~~~

Se saltarán las 170 válidas y se repartirán 612 pendientes.

QC:

~~~bash
/RUTA_NUEVA/envs/TotalSegmentator/bin/python \
  workflows/pmpd_body_mask/make_qc_montage.py \
  /RUTA_NUEVA/PMPD_v2_data/ZZU/images/zzu_008.nii.gz \
  /RUTA_RUNTIME/qc/zzu_008_after.png \
  --mask /RUTA_NUEVA/PMPD_v2_data/ZZU/body_mask/zzu_008.nii.gz
~~~

## Precauciones

- Usar tmux/screen para no dejar workers huérfanos.
- No contar .*.tmp.nii.gz como máscaras finales.
- No usar cohortes arteriales.
- No sobrescribir una máscara final sin validarla.
- No copiar credenciales al repo.
- Comparar conteos y checksums antes de retirar la workstation antigua.
