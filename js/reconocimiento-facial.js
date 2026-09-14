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

    const p = landmarks;


    // ========================================
    // ROSTRO
    // ========================================

    const anchoRostro =
        distancia(p[234], p[454]);

    const altoRostro =
        distancia(p[10], p[152]);


    const proporcionRostro =
        anchoRostro / altoRostro;


    // ========================================
    // OJOS
    // ========================================

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


    // ========================================
    // CEJAS
    // ========================================

    const cejaIzquierda =
        distancia(p[70], p[107]);

    const cejaDerecha =
        distancia(p[300], p[336]);


    // ========================================
    // NARIZ
    // ========================================

    const anchoNariz =
        distancia(p[129], p[358]);

    const largoNariz =
        distancia(p[6], p[2]);


    // ========================================
    // BOCA
    // ========================================

    const anchoBoca =
        distancia(p[61], p[291]);


    /*
        En lugar de 13 → 14,
        utilizamos los puntos internos
        superior e inferior de la boca.

        13 → labio superior interior
        14 → labio inferior interior

        Pero para evitar que una pequeña
        variación de esos puntos produzca
        valores exagerados, utilizamos
        también los extremos verticales
        de la abertura.
    */

    const altoBoca =
    distancia(
        p[13],
        p[14]
    );

const altoBocaExterno =
    distancia(
        p[0],
        p[17]
    );

const altoBocaFinal =
    Math.max(
        altoBoca,
        altoBocaExterno * 0.15
    );

    // ========================================
    // NORMALIZACIÓN
    // ========================================

    const ojoIzquierdoAnchoN =
        ojoIzquierdoAncho / anchoRostro;

    const ojoDerechoAnchoN =
        ojoDerechoAncho / anchoRostro;


    const ojoIzquierdoAltoN =
        ojoIzquierdoAlto / altoRostro;

    const ojoDerechoAltoN =
        ojoDerechoAlto / altoRostro;


    const separacionOjosN =
        separacionOjos / anchoRostro;


    const cejaIzquierdaN =
        cejaIzquierda / anchoRostro;

    const cejaDerechaN =
        cejaDerecha / anchoRostro;


    const anchoNarizN =
        anchoNariz / anchoRostro;

    const largoNarizN =
        largoNariz / altoRostro;


    const anchoBocaN =
        anchoBoca / anchoRostro;

    const altoBocaN =
        altoBocaFinal / altoRostro;


    // ========================================
    // SIMETRÍA FACIAL
    // ========================================

    /*
        Comparamos puntos equivalentes
        de ambos lados del rostro.

        Utilizamos:

        234 ↔ 454
        33  ↔ 263
        133 ↔ 362
        61  ↔ 291
        129 ↔ 358

        La idea es medir cuánto se
        diferencian las distancias
        respecto al centro del rostro.
    */


    const centroX =
        (p[234].x + p[454].x) / 2;


    function diferenciaSimetria(izquierda, derecha) {

        const distanciaIzquierda =
            Math.abs(
                izquierda.x - centroX
            );

        const distanciaDerecha =
            Math.abs(
                derecha.x - centroX
            );


        return Math.abs(
            distanciaIzquierda -
            distanciaDerecha
        );
    }


    const simetriaOjos =
        diferenciaSimetria(
            p[33],
            p[263]
        );


    const simetriaBoca =
        diferenciaSimetria(
            p[61],
            p[291]
        );


    const simetriaNariz =
        diferenciaSimetria(
            p[129],
            p[358]
        );


    const diferenciaSimetriaTotal =
        (
            simetriaOjos +
            simetriaBoca +
            simetriaNariz
        ) / 3;


    const simetriaNormalizada =
        diferenciaSimetriaTotal /
        anchoRostro;


    // ========================================
    // RESULTADO
    // ========================================

    const caracteristicas = {

        rostro: {

            ancho:
                anchoRostro,

            alto:
                altoRostro,

            proporcion:
                proporcionRostro
        },


        ojos: {

            izquierdo: {

                ancho:
                    ojoIzquierdoAnchoN,

                alto:
                    ojoIzquierdoAltoN,

                proporcion:
                    ojoIzquierdoAncho /
                    ojoIzquierdoAlto
            },

            derecho: {

                ancho:
                    ojoDerechoAnchoN,

                alto:
                    ojoDerechoAltoN,

                proporcion:
                    ojoDerechoAncho /
                    ojoDerechoAlto
            },

            separacion:
                separacionOjosN
        },


        cejas: {

            izquierda:
                cejaIzquierdaN,

            derecha:
                cejaDerechaN
        },


        nariz: {

            ancho:
                anchoNarizN,

            largo:
                largoNarizN,

            proporcion:
                largoNariz /
                anchoNariz
        },


        boca: {

            ancho:
                anchoBocaN,

            alto:
                altoBocaN,

            proporcion:
                anchoBoca /
                altoBocaFinal
        },


        simetria: {

            horizontal:
                simetriaNormalizada
        }

    };


    return caracteristicas;
}

// ============================================
// EXTRAER PATRONES SIMBÓLICOS
// ============================================

function extraerPatrones(caracteristicas) {

    const rostro =
        caracteristicas.rostro;

    const ojos =
        caracteristicas.ojos;

    const cejas =
        caracteristicas.cejas;

    const nariz =
        caracteristicas.nariz;

    const boca =
        caracteristicas.boca;

    const simetria =
        caracteristicas.simetria;


    // ========================================
    // FORMA GENERAL DEL ROSTRO
    // ========================================

    let formaRostro;


    if (rostro.proporcion < 0.78) {

        formaRostro = "alargado";

    } else if (rostro.proporcion < 0.92) {

        formaRostro = "equilibrado";

    } else {

        formaRostro = "amplio";
    }


    // ========================================
    // APERTURA DE LA MIRADA
    // ========================================

    const aperturaOjoIzquierdo =
        ojos.izquierdo.proporcion;

    const aperturaOjoDerecho =
        ojos.derecho.proporcion;


    const aperturaOjosPromedio =
        (
            aperturaOjoIzquierdo +
            aperturaOjoDerecho
        ) / 2;


    let mirada;


    if (aperturaOjosPromedio > 3.2) {

        mirada = "abierta";

    } else if (aperturaOjosPromedio > 2.4) {

        mirada = "serena";

    } else {

        mirada = "profunda";
    }


    // ========================================
    // EQUILIBRIO DE LAS CEJAS
    // ========================================

    const diferenciaCejas =
        Math.abs(
            cejas.izquierda -
            cejas.derecha
        );


    let cejasPatron;


    if (diferenciaCejas < 0.015) {

        cejasPatron = "equilibradas";

    } else {

        cejasPatron = "asimetría_sutil";
    }


    // ========================================
    // PROPORCIÓN DE LA NARIZ
    // ========================================

    let narizPatron;


    if (nariz.proporcion < 0.85) {

        narizPatron = "compacta";

    } else if (nariz.proporcion < 1.05) {

        narizPatron = "proporcionada";

    } else {

        narizPatron = "alargada";
    }


    // ========================================
    // SIMETRÍA
    // ========================================

    let equilibrio;


    if (simetria.horizontal < 0.03) {

        equilibrio = "alto";

    } else if (simetria.horizontal < 0.07) {

        equilibrio = "moderado";

    } else {

        equilibrio = "orgánico";
    }


    // ========================================
    // RESULTADO
    // ========================================

    return {

        formaRostro,

        mirada,

        cejas:
            cejasPatron,

        nariz:
            narizPatron,

        equilibrio,

        valores: {

            proporcionRostro:
                rostro.proporcion,

            aperturaMirada:
                aperturaOjosPromedio,

            simetria:
                simetria.horizontal
        }

    };
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


       const patrones =
    extraerPatrones(
        caracteristicas
    );

console.log(
    "✨ Patrones simbólicos:",
    patrones
);
        
return {

    landmarks:
        landmarks,

    caracteristicas:
        caracteristicas,

    patrones:
        patrones

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
