# HANDOFF — phbands: solver de estructura de bandas MST (cristales fonónicos 2D)

Lee esto completo antes de tocar código. Este proyecto viene de una sesión de
diseño + implementación en claude.ai con Miguel (magíster en física, U. de
Chile). El objetivo es reconstruir desde cero, con mejor arquitectura, el
solver de estructura de bandas de su tesis — el código original de la tesis
"perdía soluciones constantemente" y era difícil de configurar.

## Contexto físico

Sistema: ondas elásticas **antiplanas** (polarización fuera del plano,
ecuación de Helmholtz escalar) en un medio elástico 2D con **cavidades
cilíndricas** dispuestas en una **red cuadrada** periódica. Se resuelve vía
**teoría de dispersión múltiple (MST)**: cada cavidad tiene una respuesta
local `T` (matriz de dispersión), y la periodicidad de la red se codifica en
una **suma de red** `S` (tipo Ewald/KKR). El problema de bandas es
`det(I - S·T) = 0`, un problema de autovalores **no lineal** en `ω` para cada
vector de Bloch `k` — no es una diagonalización matricial simple, y por eso
es fácil "perder" raíces si el método numérico no es cuidadoso.

## Fuentes (en `docs/`)

- `Tesis_MLetelier.pdf` — tesis de magíster completa ("Metamateriales
  elásticos no-lineales"). Contiene la derivación original, PERO tiene
  errores matemáticos identificados y corregidos en el segundo documento.
- `Modelo_fisico_matematico_canonico_v0_1.docx` — **documento de auditoría
  que es la fuente de verdad matemática**. Corrigió errores específicos de
  la tesis (ver más abajo). Todo el código debe seguir ESTE documento, no la
  tesis directamente, cuando difieran.

### Correcciones clave que el documento canónico ya hizo a la tesis

Estas son probablemente la causa raíz de que el código viejo perdiera
soluciones:

1. **Ec. (2.83) de la tesis**: evaluaba el Hankel con `|r_p|` (punto local)
   en vez de la distancia entre centros `R_qp`, y omitía la fase angular de
   Graf `e^{-i(m-n)Θ_qp}`. La suma de red correcta es (Sec. 10 del docx):
   `S_mn(κ,K) = Σ_{R≠0} e^{iK·R} H^{(1)}_{m-n}(κR) e^{-i(m-n)Θ_R}`
2. **Ec. (2.82)**: contracción de índices modales ambigua (`m''` sin suma
   explícita). Corregido a `(ST - I)a = 0` con suma explícita sobre todos
   los índices internos.
3. Convención temporal fijada a `e^{-iωt}` (⇒ `H_n^{(1)}` es la base
   *saliente*; `J_n` es la base *regular*, NO "entrante" — contiene
   componentes de ambos sentidos).

## Alcance de v0.1 (ya decidido con Miguel, ver docx Sec. 1.2/2.1/25.1)

- Polarización estrictamente antiplana.
- Dispersor: **cavidad cilíndrica libre** (tracción nula) — el único `T`
  auditado en v0.1: `T_n^cav(ω) = -J_n'(κR) / H_n^{(1)'}(κR)`.
- Red cuadrada, sin pre-esfuerzo/pre-deformación.
- Extensiones futuras (fuera de v0.1, NO implementar todavía salvo que
  Miguel lo pida): `T_inclusión`, `T_predeformado`. Deben conectar a la
  MISMA interfaz `b = T·a` sin tocar `lattice.py`/`lattice_sums.py`/
  `secular.py`. Esta separación T-S es un principio de diseño explícito del
  propio documento canónico (Sec. 11), no una preferencia mía — respétala.

## Arquitectura actual (ya implementada, ver `src/phbands/`)

```
src/phbands/
├── lattice.py          # red directa/reciproca cuadrada, camino Γ-X-M-Γ
├── basis.py             # J_n, H_n^(1) y derivadas (via scipy.special)
├── media.py              # HostMedium: mu, rho, c_t, kappa(omega)
├── lattice_sums.py       # direct_sum (lento, referencia) + chin_accelerated
│                          # + S_matrix (vectorizada, produccion)
├── scatterer/
│   ├── base.py            # interfaz abstracta Scatterer (T_n, T_diag, host_kappa)
│   └── cavity.py           # CavityScatterer (T_cavidad)
├── secular.py             # SecularProblem.M(omega,K) = I - S@diag(T)
├── rootfinding/
│   └── scan_bracket.py     # find_roots: escaneo + Brent sobre |det|^2
└── bandstructure.py        # compute_band_structure: barre el camino k
```

Paquete instalable: `pip install -e ".[dev]"` desde la raíz del repo
(`pyproject.toml` ya configurado, src-layout).

## Bugs ya encontrados y corregidos — LÉELO ANTES DE RE-DERIVAR NADA

Al validar `chin_accelerated` (la representación acelerada de Sec. 20 del
docx) contra `direct_sum` (la suma directa exacta de Sec. 10, evaluada con
`Im(κ)>0` para forzar convergencia geométrica), encontré y corregí 3 bugs.
La fórmula del docx **es correcta**; los bugs eran de transcripción a
Python:

1. **`4j**(l+1)` en Python es `(4j)**(l+1)` = `(4i)^(l+1)`, NO `4·i^(l+1)`.**
   Hay que escribir `4.0 * (1j ** (l+1))`. Esto hacía que la suma explotara
   como `4^l` para `l` grande.
2. **Off-by-one en el orden de Bessel**: usé `order = abs(l)` y luego
   evalué `J_order` cuando la fórmula pide `J_{|l|+1}` en TODOS lados
   (prefactor y suma recíproca). Corregido a `order = abs(l) + 1`.
3. **Forma `0/0` en `Q_h → 0`** (ocurre en `K=Γ=(0,0)`, término `h=(0,0)`):
   el código descartaba el término (lo trataba como NaN→0), pero para
   `l=0` (`order=1`) el límite analítico es `(a/2)/(-κ²)`, no cero. Para
   `order>1` sí es cero. Manejado explícitamente en `chin_accelerated` y en
   `S_matrix`.

**Red de seguridad para cualquier cambio futuro a `lattice_sums.py`:**
usar `validate_against_direct_sum(l, kappa_complex_con_Im_positivo, K,
lattice)` — es lo que atrapó los 3 bugs de arriba. Cualquier modificación a
`chin_accelerated`/`S_matrix` debe re-pasar esta validación (idealmente en
`Γ`, `X`, `M` y un punto genérico, para varios `l` incluyendo negativos)
antes de confiar en resultados nuevos.

## Decisiones de diseño ya tomadas con Miguel (no las reabras sin consultar)

- **Root-finding v0.1**: escaneo + bisección/Brent sobre `|det(M(ω))|²`
  (`rootfinding/scan_bracket.py`). Se explicó a Miguel la alternativa
  (método de Beyn por integración de contorno, que garantiza encontrar
  todas las raíces en una ventana sin depender de resolución de malla) y
  se decidió partir con escaneo+bisección por simplicidad, dejando la
  arquitectura lista para añadir Beyn después detrás de la MISMA interfaz:
  `find_roots(det_func, omega_min, omega_max, config) -> list[float]`.
  Cuando implementes Beyn, hazlo como un módulo nuevo
  (`rootfinding/beyn.py`) con esa misma firma, no reemplaces el existente.
- **Layout**: paquete pip instalable, src-layout (decisión delegada a mí,
  ya implementada).
- **Tests formales (pytest)**: Miguel pidió posponerlos — "primero el
  núcleo funcional". Por ahora la validación es ad-hoc (scripts/prints).
  Si conviertes la validación ad-hoc en pytest, hazlo sin quitarle a
  Miguel la visibilidad de qué se está probando y por qué (docstrings
  claros, no solo asserts mudos).

## Estado de validación al momento del handoff

- ✅ `direct_sum` vs `chin_accelerated`/`S_matrix`: coinciden a `1e-6`–`1e-9`
  en `Γ`, `X`, `M` y puntos genéricos, para `l` de `-8` a `8`.
- ✅ Pipeline end-to-end corre (lattice → lattice_sums → secular →
  rootfinding → bandstructure) y da resultados físicamente razonables
  (rama acústica emergiendo de `Γ`, número creciente de bandas al alejarse
  del origen).
- ⚠️ **Rendimiento**: ~32s para 6 puntos k en un tramo corto Γ→X
  (`m_max=3`, `h_max=30`, `n_scan=300`). Esto es demasiado lento para un
  barrido completo de la Fig. 2.5 con buena resolución. Ya se optimizó una
  vez (de 288s a 32s) cacheando los vectores recíprocos y vectorizando
  `S_matrix` sobre todos los `l` a la vez en lugar de un bucle Python. Hay
  más margen: perfilar antes de optimizar a ciegas; candidatos: bajar
  `h_max` si converge con menos términos, cachear `T_diag`, vectorizar la
  evaluación de `det` sobre varios `omega` a la vez si `numpy.linalg.det`
  lo permite en batch.
- ✅ **Validado cualitativamente contra Fig. 2.5 de la tesis**
  (`examples/reproduce_fig_2_5.py`, figura en
  `examples/fig_2_5_reproduction.png`). Para `ν=0.1` y `ν=0.7`: la rama
  acústica toca `ω̄=0` exactamente en `Γ` (dispersión lineal, como exige la
  física), el número de bandas crece al alejarse de `Γ`, y aparecen los
  cruces/anticruces esperados. El rango de `ω̄` (0 a ~1.2-1.4) y la forma
  general coinciden con el paper.
  - Parámetros calibrados que dan el mismo resultado que `m_max=5,
    h_max=30, n_scan=600` (el punto de partida del handoff anterior) pero
    ~15-20x más rápido: `m_max=3, h_max=10, n_scan=300` (~1-2s/punto k en
    vez de ~30s). Verificado comparando raíces punto a punto en varios `K`
    genéricos y en `Γ`, `X`, `M` para ambos `ν` — coinciden a 3-4
    decimales. Con esto, barrer el camino completo (120 puntos k) toma
    ~3-4.5 min por caso en vez de >1h.
  - ⚠️ **Hallazgo pendiente de investigar**: en el panel `ν=0.7` (llenado
    alto, radio cavidad ≈0.47 con `a=1` — cavidades casi tocándose) hay
    puntos dispersos/aislados cerca de `ω̄≈0.6` y un segmento sospechosamente
    plano en `ω̄≈1.12` en vez de una nube continua. Sospecha principal: el
    escaneo (`n_scan=300`) está perdiendo raíces en resonancias angostas —
    es la limitación ya documentada en `rootfinding/scan_bracket.py`
    (mesh puede fusionar/perder raíces muy cercanas o picos muy agudos de
    `|det|²`), agravada en `ν=0.7` porque cavidades casi tocándose generan
    resonancias tipo Mie muy localizadas (picos angostos en `ω`). NO se
    sospecha bug de fórmula (la suma de red y T ya están validadas por
    separado). Antes de investigar más: subir `n_scan` localmente en esa
    zona del camino y ver si el "hueco" se llena de puntos continuos.

## Próximo paso inmediato: refinar el root-finding para resonancias angostas (ν alto)

Con el pipeline ya validado end-to-end contra Fig. 2.5, el siguiente foco es
resolver el artefacto de arriba: probablemente requiere `n_scan` más alto
(o refinamiento adaptativo) específicamente en zonas donde `|det|²` tiene
picos muy agudos, sin pagar el costo de subir `n_scan` en todo el camino.
Candidatos a evaluar (discutir con Miguel antes de implementar, ver
"Decisiones de diseño" — root-finding es territorio sensible):
- Refinamiento adaptativo: tras el escaneo grueso, hacer un segundo
  escaneo más fino solo en sub-intervalos "sospechosos" (p.ej. donde el
  contraste entre mínimos vecinos es alto).
- Subir `n_scan` global (más caro, pero simple) y ver si resuelve el
  problema — sirve como diagnóstico rápido antes de optimizar.
- Adelantar el método de Beyn (`rootfinding/beyn.py`, ya previsto en el
  diseño) si el escaneo+Brent resulta insuficiente en general para `ν`
  altos.

## Referencia: cómo se reprodujo Fig. 2.5 (ya hecho, para no repetir)

Caso: cavidades cilíndricas en matriz de caucho, red cuadrada.
- `ρ_c = 920 kg/m³`, `C_t0 = 30 m/s` (tesis, Sec. 2.2.1, p. 25)
- Dos razones de llenado: `ν = 0.1` y `ν = 0.7`, con `ν = π r₁²/a²`
- Eje horizontal: camino `X-Γ-M-X` (o `Γ-X-M-Γ`, equivalente)
- Eje vertical: frecuencia adimensional `ω̄ = ωa/(2πC_t0)`

Como todo está nondimensionalizado (`a=1`), conviene trabajar con
`HostMedium(mu=1.0, rho=1.0)` (⇒ `c_t=1`, `κ=ω`) y mapear al graficar
`ω̄ = ω/(2π)` — así no hay que cargar unidades físicas reales, solo la
razón adimensional que se compara contra el eje del paper. Verifica este
mapeo con cuidado antes de confiar en la comparación (fue una discusión en
el chat original, no está 100% probado en código).

Pasos:
1. Construir el sistema (`SquareLattice`, `HostMedium(mu=1,rho=1)`,
   `CavityScatterer(radius=sqrt(nu/pi), host=...)`, `SecularProblem`).
2. Barrer el camino completo `Γ-X-M-Γ` (usar `standard_GXM_path` o
   replicar el orden `X-Γ-M-X` de la tesis si se quiere comparar
   directamente el mismo layout de la Fig. 2.5).
3. Graficar (matplotlib) `ω̄` vs distancia-a-lo-largo-del-camino, como nube
   de puntos (así se grafica en la tesis, sin conectar ramas explícitamente).
4. Comparar visualmente contra Fig. 2.5(a)/(b) de la tesis (p. 25 del PDF,
   `docs/Tesis_MLetelier.pdf`) para `ν=0.1` y `ν=0.7`.
5. Si no coincide: sospechar primero de la normalización/mapeo de
   frecuencia, luego de `m_max`/`h_max` insuficientes, antes de asumir que
   hay otro bug de fórmula (esas ya están bien validadas por separado).

## Fig. 2.4 (polarizacion en el plano, `phbands.inplane`) -- YA REPRODUCIDA

Ademas de Fig. 2.5 (antiplano, v0.1), se implemento el subpaquete
`src/phbands/inplane/` para reproducir Fig. 2.4 de la tesis (ondas L+T
acopladas dispersadas por una inclusion elastica bonded, comparacion
contra Mei et al.). Script: `examples/reproduce_fig_2_4.py`, figura en
`examples/fig_2_4_reproduction.png`.

**Importante para cualquier trabajo futuro en esta area**: este regimen
es fisicamente distinto al antiplano y sigue DIRECTAMENTE la tesis
(Sec. 2.1.4, Anexo B) -- el documento canonico de auditoria NO cubre este
caso (solo antiplano). Trata `phbands/inplane/` como una derivacion
independiente, no como una extension auditada.

Arquitectura (deliberadamente separada del pipeline antiplano, sin tocar
`lattice.py`/`lattice_sums.py`/`secular.py`/`scatterer/`):
- `inplane/media.py`: `InPlaneMedium(rho, c_l, c_t)` -- da mu, lambda,
  kappa_l(omega), kappa_t(omega).
- `inplane/boundary.py`: `transfer_matrix_D(omega, m, radius, host,
  inclusion)` -- matriz T 2x2 por modo (ec. 2.67 + Anexo B B.1a-B.1w),
  acopla l/t DENTRO del mismo modo m (nunca entre modos distintos).
  Incluye `validate_zero_scattering_limit`: chequeo algebraico (D=0 si
  inclusion=anfitrion) que ya paso.
- `inplane/secular.py`: `InPlaneSecularProblem.M(omega,K) = T@S - I` con
  indice combinado (m,alpha). S es bloque-diagonal por POLARIZACION
  (reusa `lattice_sums.S_matrix` sin modificar, una vez con kappa_l y
  otra con kappa_t -- la suma de red NO mezcla polarizaciones, ec. 2.83
  es identica por canal). T es bloque-diagonal por MODO (bloques 2x2 de
  `transfer_matrix_D`).
- Reutiliza sin cambios: `bandstructure.compute_band_structure` y
  `rootfinding.find_roots` (mismo patron `.det(omega,K)`).

**Validacion**: ademas del chequeo algebraico, la comparacion visual
contra Fig. 2.4(a)/(b) es fuerte -- en particular, las DOS ramas
acusticas (una casi plana ~0.18-0.2, otra en V clasica bajando a 0 en
Gamma) que la tesis predice explicitamente (Sec. 2.1.4, "es posible
observar dos ramas acusticas, una para el modo longitudinal y otra para
el modo transversal") aparecen en ambos paneles reproducidos, con
valores de omega_bar en X consistentes con la figura original.

Parametros calibrados: `m_max=2` (igual al usado en la tesis), `h_max=8`,
`n_scan=200` -- ~2s/punto k, ~250s para barrer 120 puntos k por caso.
Verificado contra `m_max=3,h_max=15,n_scan=300`: coincide a 3-4
decimales.

## Estilo de trabajo con Miguel

- Comunicarse en español.
- Es físico, magíster — entiende MST, funciones de Bessel/Hankel, teorema
  de Bloch, teorema de adición de Graf, sumación de Poisson. No hace falta
  explicar conceptos básicos, pero sí vale mostrar el razonamiento de
  debugging (le interesa el "por qué", no solo "ya funciona").
  Cuando algo esté ambiguo (p.ej. una decisión de estrategia numérica),
  prefiere que se le expliquen las opciones con trade-offs antes de
  decidir, no que se le presente una única opción ya elegida sin
  justificación.
- Cuidado con la fatiga de confirmación: prefiere avanzar con trabajo
  concreto (código corriendo, validaciones numéricas) por sobre discusión
  extensa sin resultados.

## Cómo levantar el proyecto

```bash
cd phbands
pip install -e ".[dev]"
python examples/reproduce_fig_2_5.py   # (si ya existe; si no, es el próximo archivo a crear)
```
