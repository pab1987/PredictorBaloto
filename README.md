# Predicción de Combinaciones

Este proyecto es una aplicación web basada en Flask que permite agregar, cargar, predecir y analizar combinaciones de números, como los utilizados en loterías. La aplicación permite a los usuarios gestionar combinaciones de números y realizar predicciones basadas en combinaciones anteriores.

## Características

- **Agregar combinaciones**: Permite a los usuarios añadir combinaciones de números y un número especial a la base de datos.
- **Carga de CSV**: Soporta la carga masiva de combinaciones desde archivos CSV.
- **Historial de predicciones**: Guarda un historial de predicciones realizadas.
- **Análisis de datos**: Proporciona endpoints para obtener frecuencias de números y probabilidades condicionales.
- **Simulación Monte Carlo**: Realiza simulaciones Monte Carlo para generar combinaciones aleatorias basadas en frecuencias anteriores.
- **CORS**: Soporta el intercambio de recursos entre dominios para facilitar el desarrollo del frontend.

## Requisitos

- Python 3.7 o superior
- PostgreSQL
- Docker y Docker Compose

## Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tu_usuario/prediccion_baloto.git
   cd prediccion_baloto


### Instrucciones Agregadas

- La sección sobre Docker incluye pasos para construir la imagen y levantar el contenedor.
- Se menciona la necesidad de un archivo `.env` para la configuración de la base de datos.

Asegúrate de ajustar cualquier detalle según las necesidades específicas de tu proyecto y tu entorno. Si necesitas más información o cambios adicionales, házmelo saber.

