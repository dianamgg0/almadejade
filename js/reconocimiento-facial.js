// ============================================
// ESPEJO MÍSTICO
// Reconocimiento facial con MediaPipe
// ============================================

import {
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm";

let faceLandmarker = null;


// --------------------------------------------
// Inicializar MediaPipe
// --------------------------------------------

async function inicializarReconocimiento() {

    try {

        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
        );

        faceLandmarker = await FaceLandmarker.createFromOptions(
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
            "No fue posible inicializar el reconocimiento facial:",
            error
        );

        return false;
    }
}


// --------------------------------------------
// Analizar una imagen
// --------------------------------------------

async function analizarRostro(imagen) {

    if (!faceLandmarker) {

        console.warn(
            "El reconocimiento facial todavía no está listo."
        );

        return null;
    }

    try {

        const resultado =
            faceLandmarker.detect(imagen);

        if (
            !resultado.faceLandmarks ||
            resultado.faceLandmarks.length === 0
        ) {

            console.log("No se encontró un rostro.");

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
            "Error analizando el rostro:",
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
window.addEventListener("DOMContentLoaded", async () => {

    await window.EspejoFacial.inicializar();

});
