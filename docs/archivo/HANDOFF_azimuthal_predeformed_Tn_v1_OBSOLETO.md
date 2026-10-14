> **OBSOLETO — NO USAR.**
>
> Este documento fue reemplazado por `HANDOFF_azimuthal_predeformed_Tn_v2.md`
> (en la raiz del repositorio). Su derivacion tenia un error de convencion de
> indices tensoriales: usaba la componente de traccion `zeta_zr` (fila z) en vez
> de `zeta_rz`. La fila z da cero identicamente para cualquier F, por lo que el
> resultado carecia de significado fisico aunque pareciera autoconsistente.
>
> Se conserva solo como registro historico. La implementacion vigente en
> `src/phbands/scatterer/cavity_azimuthal.py` NO sigue este documento.

---

# HANDOFF — Implementar T_n de cavidad cilíndrica con pre-deformación azimutal (orden D)

Este documento es autocontenido: trae toda la matemática ya derivada y validada
numéricamente. No hace falta re-derivar nada — solo traducir a código e
implementar los tests de validación descritos abajo.

## Contexto

Este es un módulo nuevo para el paquete `phbands c` (solver de estructura de
bandas MST para cristales fonónicos 2D, ondas antiplanas, red cuadrada). 

**Esta tarea es aditiva**: agrega un nuevo `Scatterer` sin tocar
`lattice.py`, `lattice_sums.py`, `secular.py`, `bandstructure.py`, ni
`rootfinding/*`. Si te encuentras necesitando modificar alguno de esos
archivos, detente — algo está mal entendido.

## Qué se está modelando

Una cavidad cilíndrica de radio `r1`, embebida en un medio elástico con una
**pre-deformación azimutal de fondo** confinada al anillo `r1<r<r2`: el borde
interno fue rotado un ángulo finito `ψ` (imaginar un cilindro rígido que giró
y fue retirado), decayendo a rotación nula en `r2`. Más allá de `r2` el medio
es homogéneo, sin deformar. Se busca el coeficiente de dispersión `T_n(ω)`
para ondas elásticas **antiplanas** (polarización fuera del plano) dispersadas
por esta cavidad, a primer orden en el parámetro de deformación:

```
D = ψ * r1² / (r2² - r1²)
```

**IMPORTANTE — alcance y estado de esta derivación:** la ecuación de onda de
la que parte este cálculo fue dada explícitamente por el usuario como punto de
partida para ESTE ejercicio específico, tratándola como correcta por ahora.
Es una revisión pendiente — el usuario indicó que "en un momento distinto
revisaremos y si es necesario actualizaremos". Por eso:
- Todo el código de este módulo debe dejar muy claro (en docstrings/comentarios)
  que implementa ESTA ecuación específica, citándola explícitamente.
- Debe ser fácil de reemplazar/actualizar después sin arqueología de código —
  aísla toda la matemática específica de esta derivación en este único
  archivo nuevo.
- No mezclar con ninguna otra variante de la ecuación que pueda aparecer en
  el historial del proyecto — usa únicamente las fórmulas de este documento.

## La ecuación de onda (punto de partida, truncada a O(D))

```
∇²u + k₀²u = D · L₁[u],      L₁[u] ≡ 4(r₂²/r²)[ (1/r)∂²u/∂r∂θ − (1/r²)∂u/∂θ ]
```

válida en el anillo `r1<r<r2`. Se descartan explícitamente los términos
`O(D²)` (decisión explícita del usuario para esta iteración).

## Derivación completa (ya resuelta — esto es lo que se implementa)

### T_n^(0) — orden cero (sin deformación)

```
T_n^(0) = -J_n'(k0*r1) / H_n^(1)'(k0*r1)
```

Nota: esto DEBE coincidir exactamente con `CavityScatterer.T_n` ya existente
en `phbands/scatterer/cavity.py` — es el mismo caso sin deformar. Úsalo como
test de sanidad (ver sección de validación).

### T_n^(1) — corrección de orden D

**Paso A — 4 integrales** (sin forma cerrada; cuadratura numérica). Con
`g_b(r') = d/dr'[J_n(k0 r')/r']` y `g_c(r') = d/dr'[H_n^(1)(k0 r')/r']`
(derivadas elementales de Bessel/Hankel vía las relaciones de recurrencia
estándar — usar `scipy.special.jvp`, `h1vp` para las derivadas de J_n, H_n^(1)
respecto de su argumento completo, luego regla del producto):

```
V_n^(b)(r1) = ∫[r1,r2] (1/r') H_n^(1)(k0 r') g_b(r') dr'
V_n^(c)(r1) = ∫[r1,r2] (1/r') H_n^(1)(k0 r') g_c(r') dr'
F_n^(b)(r2) = ∫[r1,r2] (1/r') J_n(k0 r')     g_b(r') dr'
F_n^(c)(r2) = ∫[r1,r2] (1/r') J_n(k0 r')     g_c(r') dr'
```

Todos los integrandos son complejos (por `H_n^(1)`); integrar parte real e
imaginaria por separado con `scipy.integrate.quad` (patrón ya usado en
`lattice_sums.py` — reutilizar el mismo helper de cuadratura compleja si
existe, o crear uno).

**Paso B — 6 fuentes de borde** (todas explícitas, usando las 4 integrales de A):

```
S1 = 8π n r2² k0 J_n'(k0 r1) V_n^(b)(r1)  −  (2i n r2² / r1³) J_n(k0 r1)
S2 = 8π n r2² k0 J_n'(k0 r1) V_n^(c)(r1)  −  (2i n r2² / r1³) H_n^(1)(k0 r1)
S3 = 8π n r2² F_n^(b)(r2) H_n^(1)(k0 r2)
S4 = 8π n r2² F_n^(c)(r2) H_n^(1)(k0 r2)
S5 = 8π n r2² k0 H_n^(1)'(k0 r2) F_n^(b)(r2)  −  (2i n / r2) J_n(k0 r2)
S6 = 8π n r2² k0 H_n^(1)'(k0 r2) F_n^(c)(r2)  −  (2i n / r2) H_n^(1)(k0 r2)
```

**Paso C — proyección sobre la solución de orden cero:**

```
R1 = -(S1 + S2 * T_n^(0))
R2 = -(S3 + S4 * T_n^(0))
R3 = -(S5 + S6 * T_n^(0))
```

**Paso D — eliminación (usa el Wronskiano `J_n·H_n^(1)' − J_n'·H_n^(1) = 2i/(π z)`
en `z=k0*r2`, ya aplicado para colapsar el sistema 3×3 a esta cascada):**

```
b1 = (i*π*r2/2) * [ H_n^(1)(k0 r2)*R3  −  k0*H_n^(1)'(k0 r2)*R2 ]
c1 = ( R1  −  k0*J_n'(k0 r1)*b1 ) / ( k0*H_n^(1)'(k0 r1) )
T_n^(1) = (J_n(k0 r2)/H_n^(1)(k0 r2)) * b1  +  c1  −  R2/H_n^(1)(k0 r2)
```

### Resultado final

```
T_n(ω) = T_n^(0)(ω) + D * T_n^(1)(ω)
```

(`k0 = kappa(ω)` del medio anfitrión, como en `CavityScatterer`.)

## Verificación numérica ya hecha (para referencia, replicar como test)

Con `k0=1.3, r1=0.7, r2=1.4, n=2`:

```
T_n^(0) = -0.0034799979695563678 + 0.058888772985079614j
T_n^(1) = -0.9493151163025967   - 0.11259164212244684j
```

Y la validación de convergencia `O(D²)`, resolviendo el sistema 3×3
**exacto** (sin truncar la expansión perturbativa — usar las Ψ,Φ completas,
no la cascada de arriba) para D pequeño y comparando contra `T_n^(0)+D·T_n^(1)`:

```
D=0.004:  diferencia = 3.805e-04
D=0.002:  diferencia = 9.577e-05
razón:    3.97  (esperado ≈4, confirma residuo O(D²))
```

## Plan de implementación

Archivo nuevo: `src/phbands/scatterer/cavity_azimuthal.py`

```python
"""Cavidad cilindrica con pre-deformacion azimutal de fondo (orden D).

ADVERTENCIA DE ALCANCE: la ecuacion de onda de la que parte este modulo fue
dada explicitamente por el usuario como punto de partida para esta iteracion
especifica (ver HANDOFF_azimuthal_predeformed_Tn.md). Es una revision
pendiente, no fisica cerrada -- aislar SIEMPRE la matematica de este
resultado en este archivo, no mezclar con otras variantes.

Ecuacion de onda (truncada a O(D), D^2 despreciado):
    (∇²+k0²)u = D * L1[u]
    L1[u] = 4(r2²/r²)[(1/r) d²u/(dr dtheta) - (1/r²) du/dtheta]
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import jv, hankel1, jvp, h1vp

from ..media import HostMedium
from .base import Scatterer


def _complex_quad(f, a, b, **kwargs):
    """Cuadratura para integrandos complejos: integra parte real e imaginaria
    por separado. f debe aceptar un float y devolver un complex."""
    re, _ = quad(lambda x: f(x).real, a, b, **kwargs)
    im, _ = quad(lambda x: f(x).imag, a, b, **kwargs)
    return re + 1j * im


class CavityAzimuthalPredeformedScatterer(Scatterer):
    """Cavidad de radio r1 con pre-deformacion azimutal en el anillo
    [r1, r2], parametro de deformacion D = psi * r1**2/(r2**2-r1**2).

    Valido solo a O(D); O(D^2) se desprecia (decision explicita, ver
    HANDOFF).
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

    def _Tn0(self, k0, n) -> complex:
        z1 = k0 * self.r1
        return -jvp(n, z1) / h1vp(n, z1)

    def _Tn1(self, k0, n) -> complex:
        # TODO: implementar Pasos A-D del HANDOFF, usando k0, n, self.r1,
        # self.r2. Devolver complex.
        raise NotImplementedError

    def T_n(self, omega: float, n: int) -> complex:
        k0 = self.host_kappa(omega)
        Tn0 = self._Tn0(k0, n)
        if self.D == 0:
            return Tn0
        Tn1 = self._Tn1(k0, n)
        return Tn0 + self.D * Tn1
```

Completa `_Tn1` siguiendo EXACTAMENTE los Pasos A-D de este documento. No
inventes ni reordenes — cada símbolo ya está definido arriba.

## Checklist de validación (obligatorio antes de dar por terminado)

1. **`D=0` exacto**: para varios `(omega, n)`, `CavityAzimuthalPredeformedScatterer(r1,r2,psi=0,...).T_n(omega,n)` debe coincidir con `CavityScatterer(r1,...).T_n(omega,n)` a precisión de máquina (`~1e-14` o mejor). Si no coincide, el bug está en `_Tn0` o en cómo se maneja `D=0`.

2. **Reproducir los números de referencia**: con `k0=1.3` (puedes fijar `host.kappa(omega)=1.3` eligiendo `omega` y `host` apropiados, o simplemente testear `_Tn0`/`_Tn1` directamente con `k0` inyectado), `r1=0.7, r2=1.4, n=2`, verificar `T_n^(0)` y `T_n^(1)` contra los valores de referencia de arriba (tolerancia `~1e-8`).

3. **Validación end-to-end O(D²)** (la más importante — replica el análisis que ya se hizo a mano): implementar una función de test que:
   - Resuelva el sistema 3×3 **exacto** (con las Ψ,Φ completas, D sin truncar — construir la matriz e invertir directamente, no usar la cascada perturbativa) para un `D` pequeño dado.
   - Compare contra `T_n^(0) + D*T_n^(1)`.
   - Repita con `D/2`.
   - Verifique que la razón de diferencias sea `≈4` (con tolerancia amplia, p.ej. `3.5 < razón < 4.5`, dado el ruido numérico de la cuadratura).
   - Esto requiere implementar TAMBIÉN el sistema exacto (Ψ_b, Ψ_c, Φ_b, Φ_c con las integrales F_n, V_n evaluadas en `r` variable, no solo en los bordes) — puede vivir como función de test auxiliar, no necesita ser parte de la API pública del `Scatterer`.

4. **Sanidad de signos/orden n**: probar con `n` negativo y positivo, confirmar que no hay errores tipo el `4j**(l+1)` de Python (revisar CUALQUIER exponenciación de `i` o `I` con cuidado — usar `1j**n` o `(1j)**n`, nunca escribir literales tipo `4j**n`).

5. **Rendimiento**: cada llamada a `T_n` hace 4 cuadraturas numéricas — esto puede ser lento si se llama miles de veces (barrido de banda). No optimizar prematuramente, pero dejar un comentario `# TODO: cachear integrales si el profiling muestra que es cuello de botella` — no es requisito para esta entrega.

## Qué NO hacer

- No modificar `lattice.py`, `lattice_sums.py`, `secular.py`,
  `bandstructure.py`, `rootfinding/*`.
- No intentar "mejorar" o simplificar las fórmulas matemáticas — ya están
  derivadas y validadas; solo traducir a código.
- No usar esta ecuación de onda como si fuera la versión "final" del
  proyecto — es explícitamente provisional, documentarlo así.
