"""Ondas elasticas polarizadas EN EL PLANO x-y (longitudinal + transversal
acopladas), para reproducir la Fig. 2.4 de la tesis (comparacion contra
Mei et al.).

ADVERTENCIA arquitectonica: este es un regimen fisico distinto al v0.1
(antiplano, `phbands.scatterer.cavity` + `phbands.secular`), NO una
extension de aquel. Sigue DIRECTAMENTE la tesis (Sec. 2.1.4, Anexo B),
que para este caso no paso por la auditoria matematica del documento
canonico (esa solo cubre antiplano). Se trata como codigo experimental
hasta que se valide contra Fig. 2.4 -- ver `boundary.validate_zero_scattering_limit`
para el chequeo de correccion algebraica disponible.

Reutiliza sin modificar: `phbands.lattice` (red, caminos) y
`phbands.lattice_sums.S_matrix` (la suma de red es identica por canal de
polarizacion, ecs. 2.83/2.85 de la tesis -- solo cambia el kappa con el
que se evalua). Lo nuevo es exclusivamente la matriz T 2x2 por modo
(Anexo B) y el ensamblaje que combina los dos canales l/t.
"""
