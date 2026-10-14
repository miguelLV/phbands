# Soluciones espurias y sobre-rechazo en la certificación: diagnóstico y remedios

Documento de referencia para el problema abierto en la rama antiplana
(`cycle_6` → `cycle_7` → `cycle_8`), sistema **cavidad cilíndrica en caucho,
ν = 0.7**. Escrito el 2026-08-22.

Contexto del problema: hasta `cycle_7` las bandas contenían muchos puntos
espurios; `cycle_8` intentó filtrarlos certificando cada raíz, pero marca como
`uncertain` la gran mayoría (521 de 596 = 87 %), incluidas ramas que a simple
vista son claramente físicas.

Todo lo que sigue separa explícitamente **(L)** lo que viene de literatura
establecida y **(E)** lo que se verificó numéricamente en este repositorio.

---

## Resumen ejecutivo

Hay **cuatro** mecanismos distintos, y no tienen el mismo peso. Ordenados por
cuántos candidatos explican:

| # | Mecanismo | Peso | ¿Ya está mitigado? |
|---|---|---|---|
| 1 | Mal escalamiento de la matriz secular truncada | ~68 % de los `uncertain` | No |
| 2 | Umbral de convergencia mal calibrado para ν alto | ~78 % tienen σ sano pero fallan `shift`/`edge` | No |
| 3 | Singularidades ficticias de la suma de red acelerada | **24.7 %** de los candidatos | Sí en in-plane, **no** en antiplano |
| 4 | Anomalías de Rayleigh/Wood (polos reales de la suma) | pocos | Sí (`RECIPROCAL_POLE_GUARD_BAR`) |

Los mecanismos 1 y 2 son la causa del **sobre-rechazo** de `cycle_8`. El
mecanismo 3 es una causa genuina de **puntos espurios** y es el que más
directamente explica los artefactos que sobrevivieron hasta `cycle_7`.

---

## 1. Mal escalamiento de la matriz secular truncada (causa dominante del sobre-rechazo)

**(L)** En métodos multipolares / matriz-T la inestabilidad numérica al truncar
es un problema clásico y bien documentado. El estudio comparativo de
estabilidad numérica para dispersión múltiple por cilindros circulares
(JASA Express Letters, ver Fuentes) concluye que *formulaciones de uso común se
vuelven inestables con números de truncación altos, mientras que otras menos
usadas permanecen estables por tener matrices de sistema bien condicionadas*, y
que **la normalización diagonal de la matriz del sistema es crucial** para la
estabilidad. La literatura de matriz-T atribuye la inestabilidad al crecimiento
rápido de las funciones de Hankel.

**Mecanismo concreto en este código.** La matriz es `M = I - S·diag(T)`. Sus
dos factores crecen/decrecen de forma opuesta y extrema:

- `S_{m-n} = S_l` contiene `H^{(1)}_l(κR)`, que crece **factorialmente** con `|l|`;
- `T_n` de una cavidad decae **súper-exponencialmente** con `|n|`.

En las esquinas de la matriz (`|m-n|` máximo) se multiplica un número enorme por
uno minúsculo. El producto no es despreciable, y el sistema truncado adquiere
direcciones casi-nulas **espurias localizadas en los multipolos externos ±m_max**
— que no corresponden a ningún modo físico, sino al borde artificial de la base.

**(E) Evidencia.** El test `edge_weight` de `certification.py`
(`|v[0]|² + |v[-1]|²`, peso del modo nulo en los multipolos externos) está
pensado exactamente para detectar truncación insuficiente. Pero medido en la
formulación actual **no converge: diverge**. Para el candidato `k_index=9`
(raíz física que converge limpiamente a ω̄ ≈ 0.7997), con `h_max=16`:

| m_max | ω̄ (actual) | `edge_weight` actual | ω̄ (simetrizada) | `edge_weight` simetrizada |
|---|---|---|---|---|
| 2 | 0.875819 | 9.6e-02 | 0.875819 | 4.6e-02 |
| 3 | 0.811403 | 2.5e-01 | 0.811403 | 1.7e-01 |
| 4 | 0.802946 | 8.7e-02 | 0.802946 | 1.0e-02 |
| 6 | 0.800070 | **8.8e-01** | 0.800070 | 3.9e-03 |
| 8 | 0.799941 | **8.9e-01** | 0.799941 | 6.8e-05 |
| 10 | 0.799711 | **9.7e-01** | 0.799711 | 1.3e-04 |

Las frecuencias son **idénticas a 6 cifras** en ambas formulaciones (como debe
ser: `det(I - S·T) = det(I - T^{1/2} S T^{1/2})`), pero el indicador de
convergencia se comporta al revés: en la forma actual tiende a 1 (inútil), y en
la forma simetrizada cae 3 órdenes de magnitud (indicador legítimo).

**Consecuencia práctica**: 354 de 521 `uncertain` (68 %) fallan por
`edge_weight`. Ese criterio, medido en una base mal escalada, **no está midiendo
lo que cree medir**, y empeora al subir `m_max` — es decir, "subir m_max" a secas
haría la certificación *peor*, no mejor.

**Remedio.** Usar la forma simetrizada `I - T^{1/2} S T^{1/2}` (normalización
diagonal). Costo computacional: **cero** (misma dimensión, un producto por dos
vectores diagonales). Mismos ceros. Cuidado con la rama de la raíz compleja de
`T_n`, pero para el diagnóstico del modo nulo es indiferente.

---

## 2. Umbral de convergencia mal calibrado para llenado alto

**(L)** Está establecido que los métodos multipolares convergen lentamente
cuando los cilindros están próximos a tocarse: para sistemas de cilindros muy
cercanos *se requiere un número muy grande de términos multipolares* para
converger, y la caracterización estándar de convergencia (pensada para el
régimen bien separado) es inadecuada en el régimen de proximidad
(ver Fuentes: análisis multipolar de sistemas de cilindros dieléctricos;
"On the Convergence of the Multipole Expansion Method").

**Geometría de este caso**: ν = 0.7 ⇒ `r/a = 0.472` ⇒ separación entre
superficies = `a - 2r = 0.056 a`. Las cavidades **casi se tocan**. Es
precisamente el régimen difícil. (Para comparar: el caso Au/Be de Mei que sí
reproduce limpio tiene ν = 0.0908 ⇒ `r/a = 0.17`, con holgura de `0.66 a`.)

**(E) Evidencia.** La secuencia de frecuencias converge de forma limpia y
geométrica, pero lenta:

```
m_max:      2         3         4         6         8        10
ω̄:      0.875819  0.811403  0.802946  0.800070  0.799941  0.799711
incremento: --      6.4e-2    8.5e-3    2.9e-3    1.3e-4    2.3e-4
```

El umbral de `cycle_8` es `CERTIFY_SHIFT_MAX_BAR = 7.5e-4`, evaluado en el
último nivel `m_max = 4`. Ahí el incremento todavía vale `~8.5e-3`, **más de 10×
el umbral**. No baja del umbral hasta `m_max ≈ 8`. Es decir: el criterio es
correcto en espíritu pero se aplica a una profundidad insuficiente, y el modo es
rechazado a pesar de que `σ_min ~ 1e-9` en *todos* los niveles (la SVD confirma
que hay un espacio nulo genuino en cada uno).

**(E)** 405 de 521 `uncertain` (78 %) tienen `sigma` y `null_residual` **sanos** y
fallan sólo por `frequency_shift` y/o `edge_weight`. No son raíces falsas: son
raíces buenas medidas con una regla mal calibrada.

**Remedios (por costo creciente).**

- **Criterio por tasa de convergencia** (Aitken Δ² / Richardson) en vez de umbral
  absoluto sobre el último salto: aceptar si los incrementos decaen de forma
  consistente y extrapolar el límite con su barra de error. Usa niveles **ya
  calculados** ⇒ costo extra **cero**. Es práctica numérica estándar.
- **Subir `m_max` sólo para ν alto** (≈ 8 para ν = 0.7). Costo: la matriz pasa de
  5×5 a 17×17 y `S_matrix` recorre `4·m_max+1` órdenes ⇒ del orden de 5–10× el
  costo de certificación. Sólo tiene sentido *después* del remedio del punto 1
  (si no, `edge_weight` empeora al subir `m_max`).

---

## 3. Singularidades ficticias de la representación acelerada de la suma de red

Ésta es la causa más clara de **puntos espurios genuinos** (no de sobre-rechazo).

**(L)** La representación acelerada usada aquí es la de Botten–McPhedran–
Nicorovici (Rayleigh multipole methods, PIER 41). Su ec. (22) es, en su
notación,

```
S^A_n J_n(kρ) = -H_0(kρ) δ_{n0} - (4 i^{n+1}/A) Σ_j J_n(|K_j+k_0|ρ) e^{-i n arg(K_j+k_0)} / ((K_j+k_0)² - k²)
```

y el artículo señala explícitamente que **`ρ` es un vector arbitrario dentro del
círculo inscrito en la celda de Wigner–Seitz central**. Es decir, `ρ` es un
**parámetro libre de la representación** (una especie de gauge), no un dato
físico. La versión que usa el código tiene el orden de Bessel elevado en 1
(`J_{|l|+1}`), que corresponde a la variante acelerada por una integración,
descrita en el mismo artículo.

**El problema**: para despejar `S_l` hay que **dividir por `J_{|l|+1}(κρ)`**. En
`lattice_sums.py` el parámetro está fijado a **`ρ = a`** (ver
`prefactor = 1.0 / jv(order, kappa * a)`). Con `a = 1` y `κ = 2π ω̄`, los ceros
de Bessel caen en

```
ω̄ = j_{|l|+1, s} / (2π)
```

es decir, para `m_max = 2` (⇒ órdenes 1…5), dentro de la ventana ω̄ ∈ [0, 1.3]:

```
J_1: 0.6098, 1.1166      J_2: 0.8174     J_3: 1.0154     J_4: 1.2077
```

**(E) Evidencia de que estos valores contaminan los resultados**:

- `ω̄ = 0.6098` y `ω̄ = 1.1166` aparecen repetidamente como "raíces" en los
  escaneos (coinciden con `j_{1,1}/2π` y `j_{1,2}/2π` a 3e-5).
- El `multiplicity_event` registrado por `cycle_8` en `ω̄ = 1.1165947` coincide
  con `j_{1,2}/2π = 1.116565` a 3e-5: es una **degeneración falsa**.
- La banda plana espuria en ω̄ ≈ 0.60–0.62 visible en el diagnóstico de
  `cycle_8` está exactamente ahí.

**(E) Naturaleza del artefacto.** La singularidad es *removible* analíticamente y
la fórmula es correcta (se comprobó: comparando `direct_sum` vs
`chin_accelerated` con `Im(κ) = 0.05` el error relativo se mantiene en ~1e-3
cruzando el cero, sin explosión). El daño es puramente de **cancelación
catastrófica en una micro-banda del eje real**: numerador y denominador se
anulan juntos, y el cociente pierde toda precisión en un entorno de ancho
~3e-5 en ω̄. Barriendo `σ_min` equilibrado alrededor de `j_{1,1}/2π`
(K genérico, ν = 0.7, `h_max = 8`):

```
δω̄ = -3.0e-5 → σ_min = 4.14e-01
δω̄ = -1.0e-5 → σ_min = 3.32e-01
δω̄ = -5.0e-6 → σ_min = 1.57e-01   ← mínimo local espurio (fondo ~0.45)
δω̄ =  0      → σ_min = 1.00e+00
δω̄ = +5.0e-6 → σ_min = 5.87e-01
```

Perfil discontinuo y asimétrico: basura numérica, no una resonancia.

**(E) Por qué `cycle_7` empeoró las cosas.** El escaneo global usa
`n_scan = 520` sobre [0, 1.3] ⇒ paso `2.5e-3`, mucho más ancho que la
micro-banda: la **salta**. Pero la malla ultrafina de multiplicidad introducida
en `cycle_7` usa `MULTIPLICITY_ULTRAFINE_GRID_SIZE = 1601` sobre
`±MULTIPLICITY_ULTRAFINE_HALF_WIDTH = 0.006` ⇒ paso `7.5e-6`, que **cae dentro**
de la banda de basura y la resuelve. Por eso `cycle_7` "recuperó" 26 raíces
donde `cycle_6` no encontraba ninguna, y parte de esa recuperación es artefacto.

**(E) Asimetría entre ramas.** `cycle_3` (in-plane) **sí** tiene la guarda:
`nearest_bessel_zero_distance()` con `BESSEL_POLE_GUARD = 1e-2` sobre los
órdenes `1 … 2·M_MAX+1`, y la usa para rechazar candidatos sin soporte de rama
vecina. `cycle_6` / `cycle_7` / `cycle_8` (antiplano) **no tienen ninguna
guarda de Bessel** (`grep -n bessel` no devuelve nada). Es exactamente la rama
donde está el problema.

**(E) Cuantificación.** Medido sobre `omega_bar_input` (la frecuencia que
realmente produjo el escaneo o la recuperación; **no** sobre `omega_bar`, que ya
viene desplazada por la certificación — usar ésta última subestima el conteo por
un factor ~4): **147 de 596 candidatos (24.7 %) caen sobre un cero de Bessel**
con la guarda `1e-2`, y 183 (30.7 %) con `5e-2`. Vienen 120 del escaneo global y
27 de la recuperación SVD local. Agrupados:

```
omega_bar ≈ 1.1165  (63 candidatos)  = j_{1,2}/2π
omega_bar ≈ 1.2077  (18 candidatos)  = j_{4,1}/2π
omega_bar ≈ 0.6098  (21 candidatos)  = j_{1,1}/2π
omega_bar ≈ 0.8173  ( 9 candidatos)  = j_{2,1}/2π
omega_bar ≈ 1.0156  ( 6 candidatos)  = j_{3,1}/2π
```

Son exactamente las líneas planas espurias de las figuras. Es decir, este
mecanismo es la **causa principal de puntos espurios** (aunque no del
sobre-rechazo, que sigue siendo de los mecanismos 1 y 2).

**Remedios.**

- **Portar la guarda de Bessel del in-plane al antiplano.** Costo ~0. Es copiar
  `nearest_bessel_zero_distance` de `cycle_3`, adaptando a un solo canal (`c_t`).
- **Excluir las micro-bandas de las mallas ultrafinas** (saltarse ±δ alrededor de
  cada `j_{|l|+1,s}`), que es donde se generan.
- **Mover las singularidades**: parametrizar `ρ` en `lattice_sums.py` en vez de
  fijarlo a `a`. Con `ρ = a/2` (que además es el valor admisible según el círculo
  inscrito de la celda de Wigner–Seitz, de radio `a/2`) el primer cero ficticio
  se corre de ω̄ = 0.61 a ω̄ ≈ 1.22, sacando casi todos los artefactos de la
  ventana de interés. **Requiere re-derivar la forma con `ρ` general y
  re-validar contra `direct_sum`** (la red de seguridad ya existe); no lo hagas
  a ojo, la fórmula integrada tiene factores que dependen de `ρ`.
- **Test de gauge** (el más elegante): `S_l` físico **no puede depender de `ρ`**.
  Evaluar la suma con dos `ρ` distintos y descartar todo candidato cuyo `σ_min`
  cambie ⇒ discrimina artefacto de representación vs. raíz física de forma
  rigurosa y barata. Requiere primero el `ρ` parametrizado.
- **Alternativa de fondo**: usar una representación sin este defecto (Ewald), ver
  la revisión de Linton (SIAM Review 2010) que compara sistemáticamente las
  representaciones de sumas de red para Helmholtz. Es reescribir
  `lattice_sums.py`; sólo si los remedios baratos no bastan.

---

## 4. Anomalías de Rayleigh/Wood (ya mitigado)

**(L)** Las sumas de red **divergen genuinamente** en las singularidades de
Rayleigh, `ω = c|k + q|` con `q` de la red recíproca; son las anomalías de
Wood. A diferencia del punto 3, éstos son polos **físicos** de la representación
periódica, no artefactos. Existe literatura sobre regularización de sumas de red
para poder encontrar bandas cerca de ellas.

**Estado en el código**: ya está tratado, vía `nearest_reciprocal_pole_bar()` +
`RECIPROCAL_POLE_GUARD_BAR = 2e-3` + un tercer nivel de convergencia
`H_POLE_CHECK = 16` y exigencia de soporte de rama vecina. No parece ser una
causa dominante del problema actual.

---

## Nota aparte: el escaneo global no pasa por ninguna validación

**(E)** En `cycle_6/run_cycle_6.py` (motor de `cycle_7` y `cycle_8`), las raíces
del escaneo global (`source="determinant_scan"`) se escriben directo al CSV de
bandas **sin pasar por `refine_matrix_candidate`, ni SVD equilibrada, ni guarda
de polos**: esa maquinaria sólo se aplica a las raíces *recuperadas* en ventanas
activadas. Como el escaneo aporta la gran mayoría de los puntos finales
(516 de 596 en la corrida de ν = 0.7), es una vía abierta para puntos espurios
independiente de todo lo anterior. En `cycle_8` esto queda cubierto porque la
certificación se aplica a *todos* los candidatos — pero en `cycle_6`/`cycle_7`
(y por tanto en el CLI sin `--certify`) no.

---

## La capa de certificación del ciclo 8 no paga su costo

**(E)** Medido sobre la corrida completa de ν = 0.7 (596 candidatos):

- **Costo**: 737 s de 1052 s totales ⇒ **70 % del tiempo de ejecución**.
  (Referencia: `cycle_7` completo, sin certificación, tarda 168 s.)
- **Beneficio**: rechaza **6** candidatos (1.0 %) y deja 521 (87 %) en limbo.
- Los 6 rechazos salen **todos** por `unstable_at_reciprocal_pole` — criterio que
  el motor del ciclo 6 ya aplica por su cuenta, gratis, dentro de
  `search_windows`. La certificación no aporta ningún rechazo propio.
- De los 147 candidatos que caen sobre un cero de Bessel — artefactos
  identificables analíticamente en microsegundos — la certificación **no rechaza
  ninguno**: 32 quedan `uncertain` y 1 queda `certified`.
- Las bandas certificadas no forman ramas completas sino **4 fragmentos
  arbitrarios** del camino (`k` ∈ [0,11], [33,47], [69,87], [106,119]): no está
  separando física de artefacto, está cortando bandas.

Conclusión: la capa más cara del pipeline no discrimina, y las dos causas reales
del sobre-rechazo (mecanismos 1 y 2) se atacan más barato y más directo. De ahí
`cycle_9`: volver al estado del ciclo 7 y aplicar el Nivel 0 encima.

---

## Plan recomendado (balance costo/confiabilidad)

**Nivel 0 — costo ~0, ataca las dos causas dominantes**

1. Simetrizar la matriz secular a `I - T^{1/2} S T^{1/2}` para todos los
   diagnósticos de modo nulo. Devuelve el sentido a `edge_weight`. *(verificado)*
2. Portar la guarda de ceros de Bessel del in-plane al antiplano, y excluir esas
   micro-bandas de las mallas ultrafinas de multiplicidad.
3. Reemplazar el criterio de `frequency_shift` absoluto por uno de **tasa de
   convergencia** (Aitken/Richardson) sobre los niveles ya calculados.

**Nivel 1 — costo moderado, sólo si el Nivel 0 no basta**

4. Subir `m_max` de certificación a ~8 para ν alto (después de (1), no antes).
5. Parametrizar `ρ` en la suma de red, re-validar contra `direct_sum`, y usar el
   **test de gauge en `ρ`** como filtro riguroso de artefactos.

**Nivel 2 — sólo si hace falta reescribir**

6. Suma de red por Ewald (elimina la clase de artefacto del punto 3 de raíz).
7. Integración de contorno (Beyn) para el *conteo* robusto de raíces. Nota: ya
   está implementado en `rootfinding/beyn.py` y validado contra escaneo fino,
   pero se midió en **50–80 s por punto k** con `m_max=6, h_max=40`, y **no
   resuelve** los mecanismos 1 ni 2, que son los dominantes. No es la prioridad.

---

## Anexo: el parámetro ρ es libre — evidencia y alternativas

Añadido tras la corrida del ciclo 9, que dejó 81 artefactos residuales (15.6 %).

### La derivación completa con ρ libre

Partiendo de Botten et al. ec. (22) y aplicando el truco de integración que el
propio artículo describe (multiplicar por `ρ^{n+1}` e integrar, usando
`∫₀ˣ t^{n+1} J_n(t) dt = x^{n+1} J_{n+1}(x)` y
`∫₀ˣ t H_0(t) dt = x H_1(x) + 2i/π`) se obtiene

```
S_n = { -[H_1(kρ) + 2i/(π k ρ)] δ_{n0}
        - (4 i^{n+1} k / A_c) Σ_j J_{n+1}(Q_j ρ) e^{-i n θ_j} / (Q_j (Q_j² - k²)) } / J_{n+1}(kρ)
```

que es **exactamente** la fórmula implementada en `lattice_sums.py`, salvo que
allí `ρ` está fijado a `a`. `A_c` y los vectores recíprocos siguen viniendo de
`a`; `ρ` sólo aparece dentro de los argumentos de Bessel/Hankel.

**(E) Verificado**: la versión con `ρ` libre reproduce `S_matrix` bit a bit con
`ρ = a`, y con `ρ = 0.45 a` da el mismo `S` físico (coincide con `direct_sum`
dentro de su error de truncación). Es decir, `ρ` es un **gauge**.

### Prueba de que el artefacto es puro gauge

Barrido ultrafino de `σ_min` alrededor de `ω̄ = j_{1,1}/2π = 0.609835`
(ν = 0.7, K genérico, `h_max = 8`):

```
   δω̄        σ_min [ρ=a]      σ_min [ρ=0.45a]
 -3.0e-05     4.1356e-01        4.5501e-01
 -1.0e-05     3.3241e-01        4.5501e-01
 -5.0e-06     1.5670e-01        4.5500e-01
  0           1.0000e+00        4.5500e-01
 +5.0e-06     5.8710e-01        4.5500e-01
 +3.0e-05     4.7692e-01        4.5499e-01
```

Con `ρ = a` es basura numérica; con `ρ = 0.45 a` **no hay absolutamente ningún
rasgo**. La "banda" en 0.6098 no existe: es una elección de representación.

### Pero ρ no se puede elegir gratis

**(E)** Error relativo de `S` frente a una referencia convergida
(`ρ = a`, `h_max = 200`):

| ω̄ | h_max | ρ = a | ρ = 0.45a | ρ = 0.30a |
|---|---|---|---|---|
| 0.10 | 8 | **2.2e-04** | 3.8e-02 | 9.3e-01 |
| 0.6098 | 8 | 9.0e-02 | **1.5e-02** | 3.4e-01 |
| 1.2077 | 8 | 2.2e-01 | **1.3e-03** | 2.4e-02 |

Tres restricciones que compiten:

1. **Cota superior dura**: el teorema de adición exige `ρ` menor que la
   distancia al vecino más cercano (`= a`); Botten et al. piden `ρ` dentro del
   círculo inscrito en la celda de Wigner–Seitz (`ρ < a/2`). El valor actual
   `ρ = a` está *justo en el borde* de convergencia. Verificado: con `ρ = 1.2 a`
   el error salta a ~60 % — la serie ya no converge.
2. **Baja frecuencia favorece ρ grande**: cuando `κρ → 0`,
   `J_{|l|+1}(κρ) ~ (κρ/2)^{|l|+1}/(|l|+1)!` se hace minúsculo para órdenes
   altos, y dividir por él amplifica el error de truncación. Por eso `ρ = a`
   gana por dos órdenes de magnitud en `ω̄ = 0.1`.
3. **Los ceros ficticios favorecen ρ pequeño**: con `ρ = a` caen en
   ω̄ = 0.6098, 0.8174, 1.0154, 1.2077, 1.3960 — dentro de la ventana. Con
   `ρ = 0.45 a` se van todos a ω̄ ≥ 1.355, fuera de ella.

No existe un `ρ` fijo óptimo en todo el rango. Una regla minimax global
(`argmax_ρ min_orden |J_orden(κρ)|`) **falla**: elige `ρ` fuera del radio de
convergencia en baja frecuencia. **(E)** verificado.

### Alternativas

| | Qué hace | Costo | Riesgo |
|---|---|---|---|
| **A. Parche de gauge local** | Mantener `ρ = a`; para cada orden `l`, si `\|J_{\|l\|+1}(κa)\|` cae bajo un umbral, recalcular *sólo ese* `S_l` con `ρ = 0.45a` | ≈ 0 (sólo se activa en micro-bandas raras) | Bajo: no es filtrado, es evaluar la misma cantidad física en un gauge mejor condicionado |
| **B. Test de gauge como filtro** | Calcular con dos `ρ` y rechazar candidatos cuyo `σ_min` cambie | 2× el costo de `S` | Bajo, pero sólo **detecta**; el artefacto sigue generándose |
| **C. Heurística "clavado al cero"** | Rechazar candidatos con frecuencia fija en `j/2π` a lo largo de `k` | ≈ 0 | **Alto**: filtrado a posteriori; puede borrar bandas planas físicas legítimas |
| **D. Ewald** | Otra representación, sin este defecto | Reescribir `lattice_sums.py` | Medio |

**A domina a B y C**: elimina el artefacto en su origen en vez de filtrarlo, sin
heurísticas sobre la física, y sin costo apreciable.

### Resultado de implementar A (ciclo 9)

Implementado en `LatticeSumConfig.gauge_alternatives_over_a`: para cada orden se
elige, entre `rho_over_a` y las alternativas, el radio que **maximiza**
`|J_{|l|+1}(κρ)|`. No hace falta umbral: el criterio de máximo resuelve solo los
dos regímenes que compiten (baja frecuencia → gana el ρ grande; cerca de un cero
ficticio → gana el que se aleja de él). Con `gauge_alternatives_over_a = ()` el
resultado es bit a bit idéntico al histórico.

Corrida sobre cavidad en caucho, ν = 0.7, con `(0.45,)` y **sin** guarda de
Bessel (apagada a propósito, para aislar el efecto):

| | artefactos en las bandas finales | tiempo |
|---|---|---|
| ciclo 8 (certificación) | 147 / 596 = **24.7 %** | 1052 s |
| ciclo 9 v1 (guarda de Bessel) | 81 / 520 = **15.6 %** | 194 s |
| ciclo 9 v2 (parche de gauge) | 8 / 442 = **1.8 %** | 252 s |

(El parche recalcula sólo los órdenes que mejoran con el radio alternativo, no
la tabla completa: 428 s → 252 s sin cambiar el resultado. Coste por evaluación
de `S`: 3.06 ms con parche vs 2.14 ms sin él.)

Y lo importante: en la v2 el número de candidatos **rechazados por cualquier
filtro es cero**. Los artefactos no se filtran — no se generan.

**(E)** Los 8 puntos restantes cercanos a un cero de Bessel **no son
artefactos**: los 8 tienen vecinos a ambos lados en `k` con frecuencias que
varían suavemente, es decir son bandas físicas que *cruzan* esas frecuencias.
Ejemplo en `k = 90`: 1.1392, 1.1284, **1.1174**, 1.1064, 1.0952 — una rama que
desciende y pasa por `j_{1,2}/2π`. Esto confirma que descartar la opción C fue
correcto: una heurística de "clavado al cero" o una guarda ciega habrían abierto
huecos justamente en estas ramas.

**(E) Nota aparte**: en `ω̄ = 1.2077` con `ρ = a` y `h_max = 8` (el valor de
`H_PRIMARY` en los ciclos 6–9) el error de `S` llega a **22 %**. `h_max = 8`
parece insuficiente cerca de los ceros ficticios — otra razón por la que los
artefactos son tan marcados.

## Fuentes

- L. C. Botten, R. C. McPhedran, N. A. Nicorovici, A. A. Asatryan, C. M. de
  Sterke, P. A. Robinson, K. Busch et al., *Rayleigh multipole methods for
  photonic crystal calculations*, Progress In Electromagnetics Research, PIER
  41, 21–60 (2003). — Fuente directa de la representación de la suma de red
  (ec. 22), del carácter **libre** del parámetro `ρ`, y del uso de SVD para
  obtener autofrecuencias, vectores nulos y su multiplicidad.
  <https://www.jpier.org/ac_api/download.php?id=0201082>
- C. M. Linton, *Lattice Sums for the Helmholtz Equation*, SIAM Review 52(4),
  630–674 (2010). — Revisión sistemática de representaciones de sumas de red
  (imágenes, Kummer, Ewald, espectral) y sus propiedades de convergencia.
  <https://epubs.siam.org/doi/10.1137/09075130X>
- *Calculating multiple scattering by circular cylinders using the multipole
  method: A comparative study on numerical stability*, JASA Express Letters
  6(3), 032402. — Formulaciones estables vs inestables al truncar; la
  **normalización diagonal** de la matriz del sistema como requisito de
  estabilidad.
  <https://pubs.aip.org/asa/jel/article/6/3/032402/3384105/>
- *On the Convergence of the Multipole Expansion Method*, SIAM Journal on
  Numerical Analysis (arXiv:2009.12143). — Análisis de convergencia y su
  degradación en el régimen de cilindros próximos.
  <https://arxiv.org/abs/2009.12143>
- *Multipole analysis of a generic system of dielectric cylinders and
  application to fibrous materials*. — Requisito de gran número de términos
  multipolares para cilindros muy cercanos.
  <https://www.sciencedirect.com/science/article/abs/pii/S0304388604001317>
- W.-J. Beyn, *An integral method for solving nonlinear eigenvalue problems*,
  Linear Algebra and its Applications 436(10), 3839–3863 (2012).
  <https://arxiv.org/abs/1003.1580>
- J. Mei, Z. Liu, J. Shi, D. Tian, *Theory for elastic wave scattering by a
  two-dimensional periodical array of cylinders: An ideal approach for
  band-structure calculations*, Phys. Rev. B 67, 245107 (2003). — Referencia
  reproducida por los ciclos 3–7.
