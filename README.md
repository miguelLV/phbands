# phbands

Solver de estructuras de bandas para cristales fononicos bidimensionales con
cilindros en una red cuadrada. Incluye polarizaciones in-plane y antiplana,
busqueda global mediante determinante y recuperacion local mediante SVD.

El directorio `cycles/` conserva las etapas de validacion que reproducen los
resultados de Au/Be y Pb/epoxy discutidos por Mei et al., Phys. Rev. B 67,
245107 (2003).

## Instalacion

En PowerShell, desde la raiz del repositorio:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Si `py` no esta disponible, use cualquier instalacion de Python 3.10 o
superior para crear el entorno.

## Ejecutar un caso personalizado

El comando principal para exploracion es:

```powershell
python examples\run_custom_bands.py --help
```

Ejemplo in-plane usando los parametros Au/Be predeterminados:

```powershell
python examples\run_custom_bands.py `
  --polarization inplane `
  --name Au-Be-prueba `
  --omega-max 0.5
```

Ejemplo antiplano modificando velocidades y fraccion de llenado:

```powershell
python examples\run_custom_bands.py `
  --polarization antiplane `
  --name exploracion-velocidades `
  --host-rho 1.85 --host-cl 13.11 --host-ct 9.1042 `
  --inclusion-rho 19.5 --inclusion-cl 3.36 --inclusion-ct 1.2399 `
  --filling-fraction 0.12
```

Ejemplo de cavidad antiplana exacta (traccion nula):

```powershell
python examples\run_custom_bands.py `
  --polarization antiplane `
  --inclusion-type cavity `
  --name cavidad-en-caucho `
  --host-rho 0.920 --host-cl 0.04 --host-ct 0.03 `
  --filling-fraction 0.10 `
  --omega-max 1.3 `
  --n-scan 520
```

Para una cavidad no se usan `inclusion-rho`, `inclusion-cl` ni
`inclusion-ct`. La cavidad in-plane aun no esta implementada.

Para certificar individualmente las raices de una cavidad antiplana, agregue
`--certify`. El ciclo 8 escanea con `m-max`, y valida cada candidato con
`m-max`, `m-max+1` y `m-max+2`, ademas de los cortes `h-primary`, `h-check`
y `h-pole`. Produce una figura solo con bandas certificadas y otra que separa
soluciones certificadas, inciertas y rechazadas. Por ejemplo:

```powershell
python examples\run_custom_bands.py `
  --polarization antiplane --inclusion-type cavity --certify `
  --name cavidad-caucho-certificada `
  --host-rho 0.920 --host-cl 0.04 --host-ct 0.03 `
  --filling-fraction 0.70 --omega-max 1.3 --n-scan 520 --m-max 2
```

Las unidades deben ser consistentes. Los valores predeterminados usan densidad
en `g/cm^3` y velocidades en `km/s`, como en el articulo de MEI. El solver
calcula internamente `mu = rho*ct^2` y, para el problema in-plane,
`lambda = rho*(cl^2 - 2*ct^2)`.

Cada ejecucion crea una carpeta dentro de `runs/` con figura, CSV de bandas,
diagnosticos de candidatos, ventanas activadas y un manifiesto JSON. `runs/`
no se versiona para evitar subir exploraciones locales accidentalmente.

## Parametros numericos importantes

- `--n-scan`: resolucion del escaneo global. Aumentarlo reduce omisiones pero
  eleva el tiempo aproximadamente de forma lineal.
- `--k-per-segment`: resolucion del camino M-Gamma-X-M.
- `--m-max`: corte angular cilindrico. MEI usa 2 y verifica puntos con 4.
- `--h-primary`, `--h-check`, `--h-pole`: cortes de la suma de red usados para
  calculo y validacion convergente.
- `--radius-ratio` o `--filling-fraction`: geometria del cilindro. Son opciones
  mutuamente excluyentes.

Las frecuencias graficadas estan normalizadas como
`omega*a/(2*pi*ct_host)`.

## Validacion

```powershell
python -m pytest -q
```

Los ciclos 5 y 7 contienen las corridas in-plane y antiplana de baja energia
mas completas, respectivamente.

## Capa pre-deformada Mooney-Rivlin (exacta en psi)

`phbands.scatterer.MooneyRivlinPredeformedScatterer` calcula el `T_n` de una
cavidad o inclusion solida de radio `a` rodeada por una capa de la matriz
`a < r < b` pre-deformada azimutalmente (perfil de equilibrio Mooney-Rivlin,
giro `psi` del borde interno, `beta = C2/(C1+C2)`). No trunca en `psi`: integra
la ecuacion incremental de Ogden reducida a una ecuacion real. Con `beta = 0`
(neo-Hookean) o `beta = 1` la pre-deformacion no cambia `T_n`.

```powershell
python examples\run_azimuthal_bands.py --name mr-cavidad `
  --model mooney-rivlin --beta 0.5 --psi 0.5
```
