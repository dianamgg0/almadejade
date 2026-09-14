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


// ============================================
// INICIALIZAR MEDIAPIPE
// ============================================

async function inicializarReconocimiento() {

    if (inicializacion) {
        return inicializacion;
    }

    inicializacion = (async () => {

        try {

            console.log("✨ Cargando reconocimiento facial...");

            const vision =
                await FilesetResolver.forVisionTasks(
                    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
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


            console.log(
                "✨ Reconocimiento facial listo"
            );

            return true;


        } catch (error) {

            console.error(
                "❌ Error inicializando reconocimiento facial:",
                error
            );

            inicializacion = null;

            return false;
        }

    })();


    return inicializacion;
}


// ============================================
// DISTANCIA ENTRE DOS PUNTOS
// ============================================

function distancia(a, b) {

    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = (a.z || 0) - (b.z || 0);

    return Math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    );
}


// ============================================
// PUNTO MEDIO
// ============================================

function puntoMedio(a, b) {

    return {
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2,
        z: ((a.z || 0) + (b.z || 0)) / 2
    };
}


// ============================================
// EXTRAER CARACTERÍSTICAS DEL ROSTRO
// ============================================

function extraerCaracteristicas(landmarks) {

    /*
        Algunos landmarks importantes de MediaPipe:

        10   → parte superior del rostro
        152  → barbilla

        33   → esquina externa ojo izquierdo
        133  → esquina interna ojo izquierdo

        362  → esquina interna ojo derecho
        263  → esquina externa ojo derecho

        61   → extremo izquierdo de la boca
        291  → extremo derecho de la boca

        1    → zona central de la nariz
        4    → parte inferior de la nariz

        234  → lado izquierdo del rostro
        454  → lado derecho del rostro
    */


    const p = landmarks;


    // ----------------------------------------
    // ROSTRO
    // ----------------------------------------

    const anchoRostro =
        distancia(p[234], p[454]);

    const altoRostro =
        distancia(p[10], p[152]);


    const proporcionRostro =
        anchoRostro / altoRostro;


    // ----------------------------------------
    // OJOS
    // ----------------------------------------

    const ojoIzquierdoAncho =
        distancia(p[33], p[133]);

    const ojoDerechoAncho =
        distancia(p[362], p[263]);


    const ojoIzquierdoAlto =
        distancia(p[159], p[145]);

    const ojoDerechoAlto =
        distancia(p[386], p[374]);


    const separacionOjos =
        distancia(
            puntoMedio(p[33], p[133]),
            puntoMedio(p[362], p[263])
        );


    // ----------------------------------------
    // CEJAS
    // ----------------------------------------

    const cejaIzquierda =
        distancia(p[70], p[107]);

    const cejaDerecha =
        distancia(p[300], p[336]);


    // ----------------------------------------
    // NARIZ
    // ----------------------------------------

    const anchoNariz =
        distancia(p[129], p[358]);

    const largoNariz =
        distancia(p[6], p[2]);


    // ----------------------------------------
    // BOCA
    // ----------------------------------------

    const anchoBoca =
        distancia(p[61], p[291]);

    const altoBoca =
        distancia(p[13], p[14]);


    // ----------------------------------------
    // SIMETRÍA
    // ----------------------------------------

    const centroRostro = {
        x: (p[234].x + p[454].x) / 2,
        y: (p[234].y + p[454].y) / 2
    };


    const simetriaHorizontal =
        Math.abs(
            (p[10].x - centroRostro.x) +
            (p[152].x - centroRostro.x)
        );


    // ----------------------------------------
    // RESULTADO
    // ----------------------------------------

    const caracteristicas = {

        rostro: {

            ancho: anchoRostro,

            alto: altoRostro,

            proporcion:
                proporcionRostro
        },


        ojos: {

            izquierdo: {

                ancho: ojoIzquierdoAncho,

                alto: ojoIzquierdoAlto,

                proporcion:
                    ojoIzquierdoAncho /
                    ojoIzquierdoAlto
            },

            derecho: {

                ancho: ojoDerechoAncho,

                alto: ojoDerechoAlto,

                proporcion:
                    ojoDerechoAncho /
                    ojoDerechoAlto
            },

            separacion:
                separacionOjos
        },


        cejas: {

            izquierda:
                cejaIzquierda,

            derecha:
                cejaDerecha
        },


        nariz: {

            ancho:
                anchoNariz,

            largo:
                largoNariz,

            proporcion:
                largoNariz /
                anchoNariz
        },


        boca: {

            ancho:
                anchoBoca,

            alto:
                altoBoca,

            proporcion:
                anchoBoca /
                altoBoca
        },


        simetria: {

            horizontal:
                simetriaHorizontal
        }

    };


    return caracteristicas;
}


// ============================================
// ANALIZAR UNA IMAGEN
// ============================================

async function analizarRostro(imagen) {

    const listo =
        await inicializarReconocimiento();


    if (!listo || !faceLandmarker) {

        console.error(
            "❌ El reconocimiento facial no pudo inicializarse."
        );

        return null;
    }


    try {

        console.log(
            "🔎 Analizando rostro..."
        );


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


        const caracteristicas =
            extraerCaracteristicas(landmarks);


        console.log(
            "🌿 Características del rostro:",
            caracteristicas
        );


        return {

            landmarks:
                landmarks,

            caracteristicas:
                caracteristicas

        };


    } catch (error) {

        console.error(
            "❌ Error analizando el rostro:",
            error
        );

        return null;
    }
}


// ============================================
// EXPONER AL HTML
// ============================================

window.EspejoFacial = {

    inicializar:
        inicializarReconocimiento,

    analizar:
        analizarRostro

};


// ============================================
// INICIAR AUTOMÁTICAMENTE
// ============================================

inicializarReconocimiento();
