# HANDOFF — Implementar T_n(D) de cavidad con pre-deformación azimutal (versión validada, reemplaza la anterior)

Este documento **reemplaza por completo** a `HANDOFF_azimuthal_predeformed_Tn.md` de una
sesión anterior. Ese documento estaba basado en una derivación con un error de convención
de índices tensoriales (confundía `ζ_zr` con `ζ_rz`) que invalidaba su resultado. Si ese
archivo sigue en el repositorio, **bórralo o archívalo** antes de empezar, para no
confundir las dos versiones.

Este documento es autocontenido: toda la matemática está derivada y validada
numéricamente por partes independientes. No se necesita re-derivar nada — solo traducir
a código y correr las validaciones descritas al final.

## Contexto

Módulo nuevo para el paquete `phbands c` (solver de estructura de bandas MST para
cristales fonónicos 2D, ondas antiplanas, red cuadrada). Ver el `HANDOFF.md` original del
proyecto para arquitectura general (interfaz `Scatterer`, separación T-S, etc.). Esta
tarea es aditiva: un archivo nuevo, sin tocar `lattice.py`, `lattice_sums.py`,
`secular.py`, `bandstructure.py`, `rootfinding/*`.

## Qué se está modelando

Cavidad cilíndrica de radio `r1`, con pre-deformación azimutal de fondo en el anillo
`r1<r<r2` (ángulo de rotación `f(r)=D(r2²/r²-1)`, `D=ψr1²/(r2²-r1²)`), calculando el
coeficiente de dispersión `T_n(ω)` para ondas antiplanas, **a la ecuación de onda
truncada a orden D** (la fuente de deformación se mantiene solo a orden D; el término D²
del la ecuación completa se descarta deliberadamente para esta versión — ver nota de
alcance más abajo).

## Estado de validación (a diferencia de la versión anterior, esta SÍ está bien fundamentada)

La ecuación de onda completa (con D y D²) fue re-derivada desde cero usando la teoría de
"small-on-large" de Ogden, siguiendo exactamente las convenciones del paper revisado por
pares Parnell (2012) "Nonlinear pre-stress for cloaking from antiplane elastic waves"
(arXiv:1203.3246, Apéndice A) — el mismo formalismo que cita tu tesis. Se verificó
simbólicamente (SymPy, diferencia exactamente 0) que esta re-derivación reproduce la Ec.
3.39 de la tesis término por término, incluido el D². La ecuación truncada a O(D) usada
aquí es simplemente esa ecuación sin su término D².

**El error que tenían las derivaciones previas**: Ogden define `{∂W/∂F}_ij ≡ ∂W/∂F_ji`
(transpuesta incorporada), y la componente de tracción físicamente relevante para la
ecuación antiplana es `ζ_rz` (primer índice = normal de la cara, segundo = dirección de
la fuerza), **no** `ζ_zr`. La componente `ζ_zr` (fila z) da **cero idénticamente** para
cualquier F — por eso las derivaciones anteriores, que usaban esa fila, producían
resultados sin significado físico aunque parecieran autoconsistentes.

## La derivación completa (ya resuelta — esto es lo que se implementa)

### Ecuación de onda (truncada a O(D))

```
∇²u + k0²u = D · L1[u],   L1[u] = 4(r2²/r²)[(1/r) d²u/(dr dtheta) − (1/r²) du/dtheta]
```

### Solución de orden cero

```
u0(r,θ) = [bn Jn(k0 r) + cn Hn(k0 r)] e^{i n θ}
```

(`Hn` significa siempre `H_n^{(1)}`, Hankel de primer tipo — convención temporal `e^{-iωt}`.)

### Fuente para el modo n

```
L1[Cn(k0 r) e^{i n θ}] = 4*i*n*r2² * e^{i n θ} * (1/r²) * d/dr[ Cn(k0 r) / r ]
```

### EDO radial para la corrección u1 = R1(r) e^{inθ}

```
R1'' + (1/r) R1' + (k0² − n²/r²) R1 = Sn(r)
Sn(r) = (4*i*n*r2²/r²) * sigma(r)
sigma(r) = d/dr[ (bn*Jn(k0 r) + cn*Hn(k0 r)) / r ]
```

### Solución particular (variación de parámetros — YA VALIDADA numéricamente vía sustitución directa en la EDO, ver checklist)

```
R1(r) = 2*pi*n*r2² * [ Jn(k0 r) * V_H(r)  −  Hn(k0 r) * V_J(r) ]
V_J(r) = ∫[r, r2] Jn(k0 r') sigma(r') / r' dr'
V_H(r) = ∫[r, r2] Hn(k0 r') sigma(r') / r' dr'
```

**CUIDADO CON EL SIGNO**: la fórmula tiene `Jn*V_H − Hn*V_J` (ese orden específico, no al
revés). Verificado numéricamente con precisión de máquina; la versión con signo invertido
falla la verificación por un factor exacto de −1.

Separando por linealidad en `bn`, `cn` (`sigma = bn*g_b + cn*g_c`):

```
g_b(r') = d/dr'[ Jn(k0 r') / r' ]
g_c(r') = d/dr'[ Hn(k0 r') / r' ]

V_J^(b)(r) = ∫[r, r2] Jn(k0 r') g_b(r') / r' dr'     V_J^(c)(r) = ∫[r, r2] Jn(k0 r') g_c(r') / r' dr'
V_H^(b)(r) = ∫[r, r2] Hn(k0 r') g_b(r') / r' dr'     V_H^(c)(r) = ∫[r, r2] Hn(k0 r') g_c(r') / r' dr'
```

### Solución completa en el anillo

```
u(r,θ) = e^{i n θ} [ bn*Psi_b(r) + cn*Psi_c(r) ]

Psi_b(r) = Jn(k0 r) + D*2*pi*n*r2² * [ Jn(k0 r)*V_H^(b)(r) − Hn(k0 r)*V_J^(b)(r) ]
Psi_c(r) = Hn(k0 r) + D*2*pi*n*r2² * [ Jn(k0 r)*V_H^(c)(r) − Hn(k0 r)*V_J^(c)(r) ]
```

**Simplificación clave**: como las integrales están referenciadas a `r2` (van de `r` a
`r2`), en `r=r2` **todas valen cero exactamente** (integral de `r2` a `r2`). Por lo tanto:

```
Psi_b(r2)  = Jn(k0 r2)          Psi_c(r2)  = Hn(k0 r2)          (exacto, sin dependencia en D)
Psi_b'(r2) = k0*Jn'(k0 r2)      Psi_c'(r2) = k0*Hn'(k0 r2)       (exacto, sin dependencia en D)
```

Y en `r=r1`, vía Leibniz + una cancelación exacta entre los términos que involucran
`g_b(r1)`/`g_c(r1)` directamente (verificado a mano y numéricamente):

```
Psi_b'(r1) = k0*Jn'(k0 r1) + D*2*pi*n*r2² * [ k0*Jn'(k0 r1)*V_H^(b)(r1) − k0*Hn'(k0 r1)*V_J^(b)(r1) ]
Psi_c'(r1) = k0*Hn'(k0 r1) + D*2*pi*n*r2² * [ k0*Jn'(k0 r1)*V_H^(c)(r1) − k0*Hn'(k0 r1)*V_J^(c)(r1) ]
```

### Condición de borde en r1 (cavidad, tracción ζ_rz nula)

```
zeta_rz / mu = du/dr + s(r)*(1/r)*du0/dtheta        [s-coupling usa u0, no u completo — consistente con truncar a O(D)]
s(r) = r*f'(r) = −2*D*r2²/r²
```

```
Phi_b(r1) = Psi_b'(r1) − (2*D*i*n*r2²/r1³) * Jn(k0 r1)
Phi_c(r1) = Psi_c'(r1) − (2*D*i*n*r2²/r1³) * Hn(k0 r1)

BC1:  bn*Phi_b(r1) + cn*Phi_c(r1) = 0
```

### Condiciones de borde en r2 (continuidad con exterior no deformado)

```
Phi_b(r2) = k0*Jn'(k0 r2) − (2*D*i*n/r2) * Jn(k0 r2)
Phi_c(r2) = k0*Hn'(k0 r2) − (2*D*i*n/r2) * Hn(k0 r2)

BC2 (continuidad de u):        bn*Jn(k0 r2) + cn*Hn(k0 r2) − en*Hn(k0 r2) = dn*Jn(k0 r2)
BC3 (continuidad de zeta_rz):  bn*Phi_b(r2) + cn*Phi_c(r2) − en*k0*Hn'(k0 r2) = dn*k0*Jn'(k0 r2)
```

### Resultado final (dn = 1, forma cerrada por eliminación)

```
rho = −Phi_b(r1) / Phi_c(r1)
A = Jn(k0 r2) + rho*Hn(k0 r2)
B = Phi_b(r2) + rho*Phi_c(r2)

T_n(D) = [ A*k0*Jn'(k0 r2) − B*Jn(k0 r2) ]  /  [ B*Hn(k0 r2) − A*k0*Hn'(k0 r2) ]
```

Esta es una fórmula cerrada, sin expansión en potencias de D — válida para cualquier D
(dentro del régimen donde truncar la fuente a O(D) siga siendo razonable).

## Hallazgo importante para los tests: T_n(D) − T_n(0) es O(D²), no O(D)

Al verificar la convergencia, `(T_n(D)-T_n(0))/D` **no** converge a una constante cuando
`D→0` — se reduce a la mitad cada vez que D se reduce a la mitad. Es decir, la corrección
de primer orden en D es **exactamente cero**; la primera corrección no nula es O(D²).
Esto es consistente con (y refuerza) un resultado independiente obtenido con la ecuación
completa (con D y D²) mediante una sustitución exacta de coordenadas, que dio T_n
exactamente invariante para todo D. Este comportamiento específico —no solo que T_n(0)
sea correcto, sino que la pendiente en D=0 sea cero— es una firma distintiva que hay que
verificar explícitamente (ver checklist).

## Valores de referencia (para tests), con k0=1.3, r1=0.7, r2=1.4, n=2

```
D=0.0    T_n = -0.0034799979695563678 + 0.058888772985079614j   (= CavityScatterer, D=0)
D=0.01   T_n = -0.0035631240          + 0.0593495026j
D=0.1    T_n = -0.0317426871          + 0.0900141857j
D=0.35   T_n = -0.2149027253          + 0.0232286627j
```

Y el patrón de convergencia O(D²):
```
D=0.01:    (T_n(D)-T_n(0))/D ≈ -0.0083 + 0.046j
D=0.005:   (T_n(D)-T_n(0))/D ≈ -0.0035 + 0.023j   (≈ mitad del anterior)
D=0.0025:  (T_n(D)-T_n(0))/D ≈ -0.0016 + 0.012j   (≈ mitad del anterior otra vez)
```

## Plan de implementación

Archivo: `src/phbands/scatterer/cavity_azimuthal.py` (mismo nombre que la versión
anterior — sobrescribir/reemplazar completamente su contenido).

```python
"""Cavidad cilindrica con pre-deformacion azimutal de fondo, ecuacion de onda
truncada a orden D (fuente D^2 descartada deliberadamente).

Derivacion fundamentada en Ogden (small-on-large) siguiendo las convenciones
exactas de Parnell (2012, arXiv:1203.3246, Apendice A). Ver
HANDOFF_azimuthal_predeformed_Tn_v2.md para la derivacion completa y valores
de referencia para tests. Formula cerrada, sin expansion perturbativa en D.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import jv, hankel1, jvp, h1vp

from ..media import HostMedium
from .base import Scatterer


def _complex_quad(f, a, b, **kwargs):
    re, _ = quad(lambda x: f(x).real, a, b, **kwargs)
    im, _ = quad(lambda x: f(x).imag, a, b, **kwargs)
    return re + 1j * im


class CavityAzimuthalPredeformedScatterer(Scatterer):
    """Cavidad de radio r1, anillo pre-deformado [r1, r2],
    D = psi * r1**2/(r2**2-r1**2). Ecuacion de onda truncada a O(D).
    """

    def __init__(self, r1: float, r2: float, psi: float, host: HostMedium,
                 quad_kwargs: dict | None = None):
        self.r1 = r1
        self.r2 = r2
        self.psi = psi
        self.D = psi * r1**2 / (r2**2 - r1**2)
        self.host = host
        self.quad_kwargs = quad_kwargs or {"limit": 200}

    def host_kappa(self, omega: float) -> complex:
        return self.host.kappa(omega)

    def T_n(self, omega: float, n: int) -> complex:
        k0 = self.host_kappa(omega)
        r1, r2, D = self.r1, self.r2, self.D

        if D == 0:
            z1 = k0 * r1
            return -jvp(n, z1) / h1vp(n, z1)

        def Jn(x):  return jv(n, x)
        def Hn(x):  return hankel1(n, x)
        def Jnp(x): return jvp(n, x)
        def Hnp(x): return h1vp(n, x)
        # g_b(r') = d/dr'[Jn(k0 r')/r'] ; g_c(r') = d/dr'[Hn(k0 r')/r']
        def g_b(rp): return k0*Jnp(k0*rp)/rp - Jn(k0*rp)/rp**2
        def g_c(rp): return k0*Hnp(k0*rp)/rp - Hn(k0*rp)/rp**2

        VJ_b = _complex_quad(lambda rp: Jn(k0*rp)*g_b(rp)/rp, r1, r2, **self.quad_kwargs)
        VJ_c = _complex_quad(lambda rp: Jn(k0*rp)*g_c(rp)/rp, r1, r2, **self.quad_kwargs)
        VH_b = _complex_quad(lambda rp: Hn(k0*rp)*g_b(rp)/rp, r1, r2, **self.quad_kwargs)
        VH_c = _complex_quad(lambda rp: Hn(k0*rp)*g_c(rp)/rp, r1, r2, **self.quad_kwargs)

        pref = 2*np.pi*n*r2**2
        i = 1j

        Phi_b_r1 = (k0*Jnp(k0*r1) + D*pref*(k0*Jnp(k0*r1)*VH_b - k0*Hnp(k0*r1)*VJ_b)
                    - (2*D*i*n*r2**2/r1**3)*Jn(k0*r1))
        Phi_c_r1 = (k0*Hnp(k0*r1) + D*pref*(k0*Jnp(k0*r1)*VH_c - k0*Hnp(k0*r1)*VJ_c)
                    - (2*D*i*n*r2**2/r1**3)*Hn(k0*r1))

        Jn_r2, Hn_r2 = Jn(k0*r2), Hn(k0*r2)
        Phi_b_r2 = k0*Jnp(k0*r2) - (2*D*i*n/r2)*Jn_r2
        Phi_c_r2 = k0*Hnp(k0*r2) - (2*D*i*n/r2)*Hn_r2

        rho = -Phi_b_r1/Phi_c_r1
        A = Jn_r2 + rho*Hn_r2
        B = Phi_b_r2 + rho*Phi_c_r2

        return (A*k0*Jnp(k0*r2) - B*Jn_r2) / (B*Hn_r2 - A*k0*Hnp(k0*r2))
```

Este código YA fue probado standalone contra los 4 valores de referencia de la tabla de
arriba y los reproduce con diferencia `<1e-10`. Aun así, revísalo contra la derivación
línea por línea antes de integrarlo — la responsabilidad de verificar cada signo al
incorporarlo al proyecto (imports, convención de `HostMedium`, etc.) es del que
implementa, no solo mía.

## Checklist de validación (obligatorio)

1. **D=0 exacto**: `CavityAzimuthalPredeformedScatterer(r1,r2,psi=0,...).T_n(omega,n)`
   debe coincidir con `CavityScatterer(r1,...).T_n(omega,n)` a precisión de máquina.

2. **Valores de referencia**: con `k0=1.3` (inyectar directamente o elegir
   `host`/`omega` para lograrlo), `r1=0.7, r2=1.4, n=2`, reproducir los 4 valores de
   `T_n(D)` de la tabla de arriba, tolerancia `~1e-6`.

3. **Verificación independiente de la solución particular** (la más importante — no
   omitir): reconstruir `R1(r)` explícitamente para un `r` de prueba dentro del anillo,
   y confirmar por diferencias finitas que satisface
   `R1'' + R1'/r + (k0²-n²/r²)R1 = Sn(r)` con tolerancia relativa `<1e-3`. Esto NO
   depende de las condiciones de borde ni del sistema 3×3 — es una prueba pura de que la
   fórmula de variación de parámetros está bien implementada.

4. **Firma O(D²)**: verificar explícitamente que `(T_n(D)-T_n(0))/D` se reduce
   aproximadamente a la mitad cuando D se reduce a la mitad (para D pequeño, p.ej.
   D=0.01, 0.005, 0.0025) — NO que converja a una constante. Si el test asume
   convergencia a una constante no nula, está mal planteado.

5. **Sanidad de signos**: repetir con `n` negativo; revisar cualquier `1j**n` (nunca
   escribir literales tipo `4j**n`, ver bug histórico ya documentado en este proyecto).

## Qué NO hacer

- No usar `ζ_zr` (fila z) para la condición de borde — da cero idénticamente, es un error
  ya identificado y descartado.
- No mezclar este archivo con la versión anterior de `cavity_azimuthal.py` — reemplazar
  completamente.
- No agregar el término D² de la ecuación completa a esta implementación — está fuera de
  alcance de esta entrega (truncamiento a O(D) es una decisión explícita, no un
  descuido).
- No modificar `lattice.py`, `lattice_sums.py`, `secular.py`, `bandstructure.py`,
  `rootfinding/*`.
