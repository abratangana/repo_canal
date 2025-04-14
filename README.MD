# REPOSITORIO DEL CANAL

[![Licencia MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Autor](https://img.shields.io/badge/Autor-Abraham%20P%C3%A9rez-blue)](https://www.youtube.com/@AbrahamP%C3%A9rezIA)
[![Canal de YouTube](https://img.shields.io/badge/YouTube-Canal-red?logo=youtube)](https://www.youtube.com/@AbrahamP%C3%A9rezIA)
[![Contribuciones](https://img.shields.io/badge/Contribuciones-Bienvenidas-brightgreen)](https://github.com/TuUsuario/TuRepositorio/blob/main/CONTRIBUTING.md)

**¡Bienvenido al repositorio oficial del canal de YouTube de Abraham Pérez!**

Aquí encontrarás el código fuente y los flujos de trabajo de n8n que se han presentado en los videos del canal. Este repositorio se actualizará de forma regular con nuevo contenido y mejoras. La colaboración y el intercambio de conocimientos son fundamentales, ¡así que cualquier aporte es bienvenido!

Dentro de este repositorio, encontrarás carpetas organizadas intuitivamente, incluyendo:

* **`rag/`**: Contiene los archivos y la lógica relacionada con la técnica Retrieval-Augmented Generation (RAG) discutida en el canal.
* **`n8n_flows/`**: Alberga los flujos de trabajo de n8n en formato JSON, listos para ser importados y utilizados en tu propia instancia de n8n.
* **`otros/`**: (Esto es un ejemplo, pueden haber mas carpetas con diferentes tópicos).

Esperamos que este repositorio sea un recurso valioso para tu aprendizaje y crecimiento dentro de esta comunidad. ¡Explora, experimenta y no dudes en contribuir!

## Tabla de Contenidos

- [Cómo Empezar](#cómo-empezar)
  - [Estructura del Repositorio](#estructura-del-repositorio)
  - [Prerrequisitos](#prerrequisitos)
  - [Instalación (Opcional)](#instalación-opcional)
- [Uso](#uso)
  - [Flujos de n8n](#flujos-de-n8n)
- [Contribución](#contribución)
- [Licencia](#licencia)
- [Contacto](#contacto)
- [Agradecimientos](#agradecimientos)

## Cómo Empezar

Esta sección te guiará para que explores el contenido del repositorio.

### Estructura del Repositorio

Como se mencionó anteriormente, el repositorio está organizado en las siguientes carpetas principales:

* **`rag/`**: Materiales relacionados con RAG.
* **`n8n_flows/`**: Flujos de n8n en formato JSON.
* **`otros/`**: Cualquier otro código o recurso relevante.

### Prerrequisitos

Aunque el código y los flujos están diseñados para ser accesibles, algunos elementos pueden requerir software específico:

* **n8n**: Para utilizar los flujos de trabajo en la carpeta `n8n_flows/`, necesitarás tener instalado [n8n](https://www.n8n.io/).
* **Python** (u otros lenguajes): Algunos scripts en la carpeta `rag/` u `otros/` podrían requerir Python u otros lenguajes de programación. Se especificará en la documentación de cada carpeta si es necesario.
* **Dependencias específicas**: El repositorio cuenta con su propio `requirements.txt` (para Python) u otros archivos de gestión de dependencias que indiquen las bibliotecas necesarias.

### Instalación (Opcional)

Si deseas ejecutar localmente alguno de los códigos o flujos (especialmente los relacionados con codigo), sigue las instrucciones específicas dentro de cada carpeta. Por ejemplo, para Python:

1.  Navega a la carpeta del proyecto (si aplica).
2.  Si hay un archivo `requirements.txt`, ejecuta:
    ```bash
    pip install -r requirements.txt
    ```

Para los flujos de n8n, simplemente importa los archivos `.json` directamente en tu instancia de n8n.

## Uso

Aquí te explicamos cómo utilizar el contenido del repositorio.

### Flujos de n8n

Los archivos `.json` dentro de la carpeta `n8n_flows/` son exportaciones directas de los flujos de trabajo mostrados en el canal. Para utilizarlos:

1.  Descarga el archivo `.json` del flujo que te interese.
2.  En tu instancia de n8n, ve a "Workflows".
3.  Haz clic en el botón "+" (Add new) y selecciona "Import from file".
4.  Selecciona el archivo `.json` descargado.
5.  El flujo se importará y estará listo para ser utilizado (puede que necesites configurar tus propias credenciales y adaptaciones).


## Contribución

¡Tu ayuda es valiosa! Si deseas contribuir a este repositorio, sigue estas pautas:

1.  Haz un "fork" del repositorio.
2.  Crea una rama para tus contribuciones (`git checkout -b feature/tu-aporte`).
3.  Realiza tus cambios y commitea (`git commit -m "Añade descripción de tu aporte"`).
4.  Sube tus cambios a tu fork (`git push origin feature/tu-aporte`).
5.  Abre un "pull request" hacia la rama `main` de este repositorio.

Por favor, consulta nuestro archivo [CONTRIBUTING.md](https://github.com/TuUsuario/TuRepositorio/blob/main/CONTRIBUTING.md) (si lo creas) para obtener más detalles sobre las directrices de contribución.



## Licencia

Este repositorio se distribuye bajo la [Licencia MIT](https://opensource.org/licenses/MIT).

## Contacto

[calendly](Calendly: https://calendly.com/automateriaorg/30min)
Numero telefónico: +58-4143249677
[Canal-youtube](https://www.youtube.com/@AbrahamP%C3%A9rezIA)

## Agradecimientos

- A los creadores de langchain, langgrah y Llamaindex por ser mis principales maestros en todo este camino durante más de un año
- Menciones especiales a Facundo corengia por ser el creador de contenido que más valor me aportó en aprendizaje
- A Alessandra Padilla por ser mi apoyo incondicional en todo momento y permitirme sacar éste proyecto adelante.
