# Predicción de estancias prolongadas en urgencias

Interfaz web local para explorar los dos momentos predictivos del proyecto, cargar libros Excel, visualizar el flujo analítico, comparar métricas y ejecutar inferencias con los pipelines entrenados.

![Pantalla inicial de la aplicación](../assets/previsualizacion_aplicacion_web.png)

## Ejecutar

La forma recomendada en Windows es hacer doble clic en `Abrir aplicacion.cmd`,
ubicado en la raíz del repositorio. El iniciador evita abrir una segunda
instancia, espera a que los modelos estén listos y abre el navegador.

Para iniciar manualmente desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe scripts\train_inference_models.py
.\.venv\Scripts\python.exe app\server.py
```

Luego abra `http://localhost:8000`.

Para terminar, use el botón **Cerrar aplicación** del pie de página. El servidor
se apagará de forma ordenada y la pestaña indicará cuándo puede cerrarse.

La interfaz valida inicialmente el libro XLSX/XLS en el navegador. Siguiendo `src/data_io.py`, exige la hoja exacta `CENSO`, ignora las demás hojas y descarta la primera fila para usar la segunda como encabezado (`header=1`). La lectura inicial usa SheetJS desde un CDN, por lo que la primera apertura requiere conexión a internet; el contenido del libro no se envía al CDN. Para habilitar la inferencia, el libro se entrega exclusivamente al servidor local, que mantiene el contexto transformado en memoria y no lo persiste.

Las predicciones individuales son reales: `src/inference.py` carga los pipelines completos ajustados sobre las particiones de desarrollo, calcula `predict_proba` y aplica los umbrales congelados para devolver **SÍ/NO**. Los archivos `.joblib` son artefactos locales ignorados por Git y se verifican mediante SHA-256 antes de cargarlos. El resultado apoya, pero no sustituye, el criterio clínico.

Al cargar el Excel, el backend prepara `CENSO` únicamente en memoria. La predicción se solicita por número interno de episodio. Ninguna variable operativa derivada se introduce manualmente: volúmenes de ingresos, calendario, duración previa, grupo de cama, pacientes activos y pacientes pendientes se reconstruyen con las mismas definiciones temporales de `src/features.py`. Para el Modelo B, conducta, sala y cama se leen del episodio existente porque están documentadas como observadas al registrar la conducta.

## Componentes

- `index.html`, `styles.css` y `app.js`: experiencia interactiva y validación del flujo.
- `server.py`: servidor HTTP local, carga segura de modelos y API JSON.
- `src/inference.py`: entrenamiento, verificación de huellas y construcción de predictores.
- `scripts/train_inference_models.py`: serialización reproducible desde desarrollo.
- `scripts/start_app.ps1`: arranque coordinado para Windows.

La aplicación es una herramienta de apoyo analítico y no sustituye el criterio clínico ni constituye un sistema desplegado para uso asistencial.
