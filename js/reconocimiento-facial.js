```javascript
// ============================================
// ESPEJO MÍSTICO
// Reconocimiento facial con MediaPipe
// ============================================

import {
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

let faceLandmarker = null;
let inicializacion = null;


// --------------------------------------------
// Inicializar MediaPipe
// --------------------------------------------

async function inicializarReconocimiento() {

    // Si ya estamos inicializando, esperamos el mismo proceso
    if (inicializacion) {
        return inicializacion;
    }

    inicializacion = (async () => {

        try {

            console.log("✨ Cargando reconocimiento facial...");

            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
            );

            faceLandmarker =
                await FaceLandmarker.createFromOptions(
                    vision,
                    {
                        baseOptions: {
                            modelAssetPath:
                                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",

                            delegate: "GPU"
                        },

                        runningMode: "IMAGE",

                        numFaces: 1,

                        outputFaceBlendshapes: false,

                        outputFacialTransformationMatrixes: false
                    }
                );

            console.log("✨ Reconocimiento facial listo");

            return true;

        } catch (error) {

            console.error(
                "❌ Error inicializando reconocimiento facial:",
                error
            );

            // Permitimos volver a intentarlo si falló
            inicializacion = null;

            return false;
        }

    })();

    return inicializacion;
}


// --------------------------------------------
// Analizar una imagen
// --------------------------------------------

async function analizarRostro(imagen) {

    // Esperar a que MediaPipe esté listo
    const listo = await inicializarReconocimiento();

    if (!listo || !faceLandmarker) {

        console.error(
            "❌ El reconocimiento facial no pudo inicializarse."
        );

        return null;
    }

    try {

        console.log("🔎 Analizando rostro...");

        const resultado =
            faceLandmarker.detect(imagen);

        if (
            !resultado.faceLandmarks ||
            resultado.faceLandmarks.length === 0
        ) {

            console.log(
                "No se encontró un rostro en la fotografía."
            );

            return null;
        }

        const landmarks =
            resultado.faceLandmarks[0];

        console.log(
            "✨ Rostro detectado:",
            landmarks.length,
            "puntos"
        );

        return landmarks;

    } catch (error) {

        console.error(
            "❌ Error analizando el rostro:",
            error
        );

        return null;
    }
}


// --------------------------------------------
// Exponer las funciones al HTML
// --------------------------------------------

window.EspejoFacial = {

    inicializar: inicializarReconocimiento,

    analizar: analizarRostro

};


// --------------------------------------------
// Comenzar carga automáticamente
// --------------------------------------------

inicializarReconocimiento();
```
