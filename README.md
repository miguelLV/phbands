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

Ejemplo antiplano modificando razones de Poisson y fraccion de llenado:

```powershell
python examples\run_custom_bands.py `
  --polarization antiplane `
  --name exploracion-nu `
  --host-rho 1.85 --host-E 377.3 --host-nu 0.03 `
  --inclusion-rho 19.5 --inclusion-E 78.5 --inclusion-nu 0.42 `
  --filling-fraction 0.12
```

Las unidades deben ser consistentes. Usar densidad en `g/cm^3` y modulo de
Young en `GPa` produce velocidades en `km/s`, como en el articulo de MEI.
El programa convierte `(rho, E, nu)` a

```text
mu = E / (2*(1 + nu))
lambda = E*nu / ((1 + nu)*(1 - 2*nu))
ct = sqrt(mu/rho)
cl = sqrt((lambda + 2*mu)/rho)
```

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
