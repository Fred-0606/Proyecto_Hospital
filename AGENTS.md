# Instrucciones permanentes para Codex

1. El proyecto utiliza Python y Quarto.
2. Todo el código debe ser compatible con Windows.
3. Utilizar `pathlib` para manejar rutas.
4. Nunca utilizar rutas absolutas como `C:\Users\...` dentro del código.
5. Todas las rutas deben resolverse desde la raíz del repositorio.
6. El código reutilizable debe guardarse en `src/`.
7. Los documentos `pipeline/*.qmd` deben contener narrativa, interpretación y llamadas a funciones de `src/`, evitando funciones extensas dentro de los documentos.
8. Cada etapa del pipeline debe poder ejecutarse en una sesión limpia.
9. Cada etapa debe leer el artefacto producido por la etapa anterior.
10. Si falta un archivo de entrada, debe generarse un error claro indicando qué etapa debe ejecutarse primero.
11. Nunca incluir datasets, datos hospitalarios, credenciales, tokens o archivos `.env` en Git.
12. Nunca sobrescribir o eliminar archivos existentes sin explicar primero qué se va a modificar.
13. Usar type hints, docstrings y nombres descriptivos.
14. Utilizar `random_state=42` en procesos aleatorios cuando corresponda.
15. Evitar data leakage: todos los transformadores deben ajustarse únicamente con los datos de entrenamiento.
16. Mantener separadas las métricas de entrenamiento, validación y prueba.
17. Priorizar recall, precision, F1, ROC-AUC y PR-AUC para clasificación.
18. Incluir pruebas básicas con pytest.
19. Después de modificar código, ejecutar las pruebas relevantes y reportar el resultado.
20. No ejecutar `git commit`, `git push`, `git reset`, `git clean` ni borrar archivos sin autorización explícita.
21. Al terminar una tarea, resumir archivos creados, archivos modificados, pruebas ejecutadas y posibles pendientes.
