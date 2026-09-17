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


    // ============================================
// SIMETRÍA FACIAL
// ============================================

// Eje central aproximado del rostro
const centroX =
    (p[234].x + p[454].x) / 2;


// Pares de landmarks izquierda ↔ derecha
const paresSimetria = [

    // Ojos
    [33, 263],
    [133, 362],
    [159, 386],
    [145, 374],

    // Cejas
    [70, 300],
    [107, 336],

    // Parte superior de la nariz
    [168, 6],

    // Laterales de la nariz
    [129, 358],

    // Boca
    [61, 291],
    [78, 308],
    [95, 324],

    // Mejillas
    [50, 280],
    [101, 330],

    // Mandíbula / contorno
    [172, 397],
    [150, 379],
    [176, 400]
];


// --------------------------------------------
// Calcular diferencia de un par
// --------------------------------------------

function diferenciaPar(izquierda, derecha) {

    // Distancia horizontal respecto al eje
    const distanciaIzquierda =
        Math.abs(
            izquierda.x - centroX
        );

    const distanciaDerecha =
        Math.abs(
            derecha.x - centroX
        );

    const diferenciaHorizontal =
        Math.abs(
            distanciaIzquierda -
            distanciaDerecha
        );


    // Diferencia vertical
    const diferenciaVertical =
        Math.abs(
            izquierda.y -
            derecha.y
        );


    // Combinamos ambas diferencias
    return Math.sqrt(
        diferenciaHorizontal *
        diferenciaHorizontal +

        diferenciaVertical *
        diferenciaVertical
    );
}


// --------------------------------------------
// Calcular asimetría promedio
// --------------------------------------------

let sumaAsimetria = 0;

for (const par of paresSimetria) {

    const izquierda = p[par[0]];
    const derecha = p[par[1]];

    sumaAsimetria +=
        diferenciaPar(
            izquierda,
            derecha
        );
}


// Promedio de todos los pares
const asimetriaPromedio =
    sumaAsimetria /
    paresSimetria.length;


// Normalización respecto al tamaño
// del rostro
const simetriaNormalizada =
    asimetriaPromedio /
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
                simetriaNormalizada,
            promedio:
        asimetriaPromedio
        }

    };


    return caracteristicas;
}

// ============================================
// EXTRAER PATRONES SIMBÓLICOS
// ============================================

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


    let interpretacionRostro;

    if (formaRostro === "alargado") {

        interpretacionRostro = "profundidad";

    } else if (formaRostro === "equilibrado") {

        interpretacionRostro = "balance";

    } else {

        interpretacionRostro = "presencia";
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


    let interpretacionMirada;

    if (mirada === "abierta") {

        interpretacionMirada = "curiosidad";

    } else if (mirada === "serena") {

        interpretacionMirada = "calma";

    } else {

        interpretacionMirada = "misterio";
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


    let interpretacionCejas;

    if (cejasPatron === "equilibradas") {

        interpretacionCejas = "armonía";

    } else {

        interpretacionCejas = "individualidad";
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


    let interpretacionNariz;

    if (narizPatron === "compacta") {

        interpretacionNariz = "determinación";

    } else if (narizPatron === "proporcionada") {

        interpretacionNariz = "equilibrio";

    } else {

        interpretacionNariz = "dirección";
    }


    // ========================================
    // SIMETRÍA
    // ========================================

    let equilibrio;

    if (simetria.horizontal < 0.06) {

        equilibrio = "alto";

    } else if (simetria.horizontal < 0.10) {

        equilibrio = "moderado";

    } else {

        equilibrio = "orgánico";
    }


    let interpretacionEquilibrio;

    if (equilibrio === "alto") {

        interpretacionEquilibrio = "centrada";

    } else if (equilibrio === "moderado") {

        interpretacionEquilibrio = "singularidad";

    } else {

        interpretacionEquilibrio = "atrevimiento";
    }


    // ========================================
    // BOCA
    // ========================================

    let bocaPatron;

    if (boca.proporcion > 22) {

        bocaPatron = "contenida";

    } else if (boca.proporcion > 8) {

        bocaPatron = "expresiva";

    } else {

        bocaPatron = "abierta";
    }


    let interpretacionBoca;

    if (bocaPatron === "contenida") {

        interpretacionBoca = "introspección";

    } else if (bocaPatron === "expresiva") {

        interpretacionBoca = "expresión";

    } else {

        interpretacionBoca = "expansión";
    }


    // ========================================
    // RESULTADO
    // ========================================

    return {

        formaRostro:
            formaRostro,

        interpretacionRostro:
            interpretacionRostro,


        mirada:
            mirada,

        interpretacionMirada:
            interpretacionMirada,


        cejas:
            cejasPatron,

        interpretacionCejas:
            interpretacionCejas,


        nariz:
            narizPatron,

        interpretacionNariz:
            interpretacionNariz,


        boca:
            bocaPatron,

        interpretacionBoca:
            interpretacionBoca,


        equilibrio:
            equilibrio,

        interpretacionEquilibrio:
            interpretacionEquilibrio,


        valores: {

            proporcionRostro:
                rostro.proporcion,

            aperturaMirada:
                aperturaOjosPromedio,

            simetria:
                simetria.horizontal,

            proporcionBoca:
                boca.proporcion
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
    "👄 Boca:",
    caracteristicas.boca
);

console.log(
    "✨ Patrones simbólicos:",
    patrones
);

const revelacion =
    generarRevelacion(patrones);

console.log(
    "🌿 Revelación:",
    revelacion
);

return {

    landmarks:
        landmarks,

    caracteristicas:
        caracteristicas,

    patrones:
        patrones,

    revelacion:
        revelacion
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
// DIMENSIONES SIMBÓLICAS
// ============================================

function calcularDimensiones(patrones) {

    const dimensiones = {

        apertura: 0,

        estructura: 0,

        direccion: 0,

        fluidez: 0,

        contemplacion: 0
    };


    // ========================================
    // FORMA DEL ROSTRO
    // ========================================

    if (patrones.interpretacionRostro === "profundidad") {

        dimensiones.contemplacion += 2;

    } else if (
        patrones.interpretacionRostro === "balance"
    ) {

        dimensiones.estructura += 2;

    } else if (
        patrones.interpretacionRostro === "presencia"
    ) {

        dimensiones.apertura += 2;
    }


    // ========================================
    // MIRADA
    // ========================================

    if (patrones.interpretacionMirada === "curiosidad") {

        dimensiones.apertura += 2;

    } else if (
        patrones.interpretacionMirada === "calma"
    ) {

        dimensiones.fluidez += 2;

    } else if (
        patrones.interpretacionMirada === "misterio"
    ) {

        dimensiones.contemplacion += 2;
    }


    // ========================================
    // CEJAS
    // ========================================

    if (patrones.interpretacionCejas === "armonía") {

        dimensiones.estructura += 2;

    } else if (
        patrones.interpretacionCejas === "individualidad"
    ) {

        dimensiones.fluidez += 1;
        dimensiones.apertura += 1;
    }


    // ========================================
    // NARIZ
    // ========================================

    if (
        patrones.interpretacionNariz === "determinación"
    ) {

        dimensiones.direccion += 2;

    } else if (
        patrones.interpretacionNariz === "equilibrio"
    ) {

        dimensiones.estructura += 2;

    } else if (
        patrones.interpretacionNariz === "dirección"
    ) {

        dimensiones.direccion += 2;
    }


    // ========================================
    // BOCA
    // ========================================

    if (
        patrones.interpretacionBoca === "introspección"
    ) {

        dimensiones.contemplacion += 2;

    } else if (
        patrones.interpretacionBoca === "expresión"
    ) {

        dimensiones.apertura += 2;

    } else if (
        patrones.interpretacionBoca === "expansión"
    ) {

        dimensiones.apertura += 2;
        dimensiones.fluidez += 1;
    }


    // ========================================
    // EQUILIBRIO
    // ========================================

    if (
        patrones.interpretacionEquilibrio === "centrada"
    ) {

        dimensiones.estructura += 2;

    } else if (
        patrones.interpretacionEquilibrio === "singularidad"
    ) {

        dimensiones.fluidez += 1;
        dimensiones.apertura += 1;

    } else if (
        patrones.interpretacionEquilibrio === "atrevimiento"
    ) {

        dimensiones.direccion += 2;
        dimensiones.apertura += 1;
    }


    return dimensiones;
}


// ============================================
// ARQUETIPOS
// ============================================

const arquetipos = {

    observadora: {

        nombre: "La Contemplación",

        emoji: "🌙",

        variaciones: [

            {
                descripcion:
                    "Tu reflejo sugiere una naturaleza contemplativa, capaz de detenerse, observar y descubrir significado en aquello que otros pasan por alto.",

                cierre:
                    "A veces observar con atención también es una forma de avanzar."
            },

            {
                descripcion:
                    "Hay en tu reflejo una energía que invita a mirar más allá de lo evidente. Pareces encontrar valor en los pequeños detalles y en aquello que necesita tiempo para ser comprendido.",

                cierre:
                    "No todo lo importante necesita ser descubierto de inmediato."
            },

            {
                descripcion:
                    "Tu reflejo habla de una mirada que prefiere comprender antes de apresurarse. Hay profundidad en tu manera de acercarte a lo que despierta tu curiosidad.",

                cierre:
                    "Cuando te permites observar con calma, también puedes encontrarte a ti."
            }

        ]
    },


    semilla: {

        nombre: "La Semilla",

        emoji: "🌱",

        variaciones: [

            {
                descripcion:
                    "Tu reflejo sugiere una energía de crecimiento. Hay algo que se está formando en tu interior, preparándose para convertirse en algo nuevo.",

                cierre:
                    "No todo lo que está creciendo necesita mostrarse todavía."
            },

            {
                descripcion:
                    "Tu reflejo habla de potencial y transformación. Quizás hay ideas, deseos o caminos que todavía están tomando forma, aunque ya exista en ti la intuición de hacia dónde pueden llevarte.",

                cierre:
                    "Lo que hoy parece pequeño también puede contener el comienzo de algo extraordinario."
            },

            {
                descripcion:
                    "Hay en tu reflejo una energía de renovación, como quien atraviesa una etapa de preparación antes de dar el siguiente paso.",

                cierre:
                    "Confía también en aquello que todavía está tomando forma."
            }

        ]
    },


    llama: {

        nombre: "La Llama",

        emoji: "🔥",

        variaciones: [

            {
                descripcion:
                    "Tu reflejo transmite impulso y determinación. Hay una energía que invita a avanzar y convertir las intenciones en movimiento.",

                cierre:
                    "Tu fuerza no está solamente en avanzar, sino en saber qué merece tu energía."
            },

            {
                descripcion:
                    "Hay en tu reflejo una energía activa, como una chispa que busca convertirse en acción. Cuando algo despierta tu interés, parece difícil permanecer completamente indiferente.",

                cierre:
                    "Una llama no necesita arder en todas direcciones para iluminar."
            },

            {
                descripcion:
                    "Tu reflejo sugiere voluntad y capacidad de movimiento. Existe en ti una fuerza que puede transformar una intención en el primer paso de un camino.",

                cierre:
                    "Cuando encuentras aquello que realmente importa, tu energía sabe hacia dónde dirigirse."
            }

        ]
    },


    rio: {

        nombre: "El Río",

        emoji: "🌊",

        variaciones: [

            {
                descripcion:
                    "Tu reflejo transmite una energía flexible y abierta. Como el agua, hay una capacidad de encontrar nuevas formas de avanzar sin perder la propia esencia.",

                cierre:
                    "A veces avanzar no significa luchar contra el camino, sino encontrar por dónde fluir."
            },

            {
                descripcion:
                    "Hay en tu reflejo una energía que parece adaptarse al movimiento de la vida. No siempre necesitas tener todo definido para continuar avanzando.",

                cierre:
                    "También existe sabiduría en saber cuándo dejar que el camino se revele."
            },

            {
                descripcion:
                    "Tu reflejo habla de movimiento y libertad. Hay una disposición a explorar diferentes caminos y permitir que las experiencias transformen tu manera de avanzar.",

                cierre:
                    "Fluir no significa perder el rumbo; a veces significa confiar en el movimiento."
            }

        ]
    },


    raiz: {

        nombre: "La Raíz",

        emoji: "🌿",

        variaciones: [

            {
                descripcion:
                    "Tu reflejo habla de estabilidad y profundidad. Hay una energía que busca sostenerse desde dentro antes de crecer hacia afuera.",

                cierre:
                    "Cuanto más profundo es tu centro, más libre puede ser tu crecimiento."
            },

            {
                descripcion:
                    "Hay en tu reflejo una sensación de firmeza y conexión interior. Antes de buscar nuevos horizontes, parece importante para ti saber desde dónde estás creciendo.",

                cierre:
                    "Lo que te sostiene por dentro también puede darte libertad para explorar."
            },

            {
                descripcion:
                    "Tu reflejo sugiere una energía serena y estable, como quien encuentra fuerza no solo en lo que muestra al mundo, sino también en aquello que guarda en su interior.",

                cierre:
                    "A veces crecer no significa alejarse de las raíces, sino profundizarlas."
            }

        ]
    },


    flor: {

        nombre: "El Loto",

        emoji: "🪷",

        variaciones: [

            {
                descripcion:
                    "Tu reflejo transmite apertura y expresión. Como el loto, hay una energía capaz de emerger desde lo profundo y abrirse hacia la luz.",

                cierre:
                    "Abrirte al mundo también puede ser una forma de reconocerte."
            },

            {
                descripcion:
                    "Hay en tu reflejo una energía de transformación y apertura. Como el loto que emerge desde aguas profundas, parece existir en ti una capacidad de convertir la experiencia en una nueva forma de florecer.",

                cierre:
                    "Lo que has atravesado también puede formar parte de la belleza que estás creando."
            },

            {
                descripcion:
                    "Tu reflejo sugiere una energía que busca expresarse y ocupar su propio espacio. Hay algo en ti que parece querer salir a la luz sin dejar atrás aquello que lo hizo crecer.",

                cierre:
                    "Florecer no significa olvidar de dónde vienes."
            }

        ]
    }
};


// ============================================
// CALCULAR ARQUETIPO
// ============================================

function calcularArquetipo(dimensiones) {

    const puntuaciones = {

        observadora:
            dimensiones.contemplacion * 3 +
            dimensiones.fluidez +
            dimensiones.apertura,

        semilla:
            dimensiones.apertura * 2 +
            dimensiones.fluidez * 2 +
            dimensiones.contemplacion,

        llama:
            dimensiones.direccion * 3 +
            dimensiones.estructura * 2 +
            dimensiones.apertura,

        rio:
            dimensiones.fluidez * 3 +
            dimensiones.apertura * 2 +
            dimensiones.contemplacion,

        raiz:
            dimensiones.estructura * 3 +
            dimensiones.contemplacion +
            dimensiones.direccion,

        flor:
            dimensiones.apertura * 3 +
            dimensiones.fluidez +
            dimensiones.contemplacion
    };


    let arquetipoGanador = "observadora";
    let puntuacionMayor = -Infinity;


    for (const nombre in puntuaciones) {

        if (
            puntuaciones[nombre] >
            puntuacionMayor
        ) {

            puntuacionMayor =
                puntuaciones[nombre];

            arquetipoGanador =
                nombre;
        }
    }


    return {

        tipo: arquetipoGanador,

        ...arquetipos[arquetipoGanador],

        puntuaciones
    };
}


// ============================================
// REVELACIONES DE LOS PATRONES
// ============================================

const revelaciones = {

    profundidad:
        "Hay en ti una mirada hacia lo profundo, una inclinación a descubrir lo que se encuentra más allá de lo evidente.",

    balance:
        "Tu reflejo sugiere una naturaleza que busca el equilibrio y encuentra belleza en los puntos medios.",

    presencia:
        "Tu reflejo transmite una presencia que no necesita imponerse para hacerse notar.",


    curiosidad:
        "Tu mirada simbólicamente habla de curiosidad, de una mente que disfruta descubrir y explorar.",

    calma:
        "Tu mirada transmite una energía serena, como quien observa antes de dejarse llevar.",

    misterio:
        "Tu mirada guarda un matiz de misterio, como si siempre conservara una parte de sí misma por descubrir.",


    armonía:
        "Hay una sensación de armonía en tu reflejo, una forma de encontrar tu propio ritmo.",

    individualidad:
        "Tu reflejo muestra una individualidad que no necesita seguir exactamente el mismo camino que los demás.",


    determinación:
        "Aparece también una energía de determinación, esa fuerza tranquila que ayuda a seguir adelante.",

    equilibrio:
        "Tu reflejo sugiere una relación natural con el equilibrio y la proporción.",

    dirección:
        "Hay en tu reflejo una sensación de dirección, como quien sabe hacia dónde quiere llevar sus pasos.",


    introspección:
        "Tu expresión guarda algo hacia dentro, una tendencia a sentir y pensar antes de mostrarlo todo.",

    expresión:
        "Tu expresión habla de una energía que encuentra facilidad para salir y hacerse visible.",

    expansión:
        "Tu expresión transmite apertura, espontaneidad y deseo de dejar que lo que llevas dentro encuentre espacio.",


    singularidad:
        "Tu equilibrio tiene un matiz propio, una singularidad que forma parte de tu manera de habitar el mundo.",

    atrevimiento:
        "Tu reflejo tiene un toque de atrevimiento, como una invitación a explorar caminos menos previsibles.",

    centrada:
        "Tu reflejo sugiere una energía centrada, capaz de regresar a su propio eje incluso mientras explora."
};


// ============================================
// GENERAR REVELACIÓN
// ============================================

function generarRevelacion(patrones) {

    const dimensiones =
        calcularDimensiones(patrones);


    const arquetipo =
        calcularArquetipo(dimensiones);

    const variacion =
    arquetipo.variaciones[
        Math.floor(
            Math.random() *
            arquetipo.variaciones.length
        )
    ];

    const esencia =
        revelaciones[
            patrones.interpretacionRostro
        ];

    const mirada =
        revelaciones[
            patrones.interpretacionMirada
        ];

    const expresion =
        revelaciones[
            patrones.interpretacionBoca
        ];

    const direccion =
        revelaciones[
            patrones.interpretacionNariz
        ];

    const matiz =
        revelaciones[
            patrones.interpretacionCejas
        ];

    const equilibrio =
        revelaciones[
            patrones.interpretacionEquilibrio
        ];


    const texto =

        esencia + " " +

        mirada + " " +

        expresion + " " +

        direccion + " " +

        matiz + " " +

        equilibrio + " " +

        variacion.descripcion + " " +

        variacion.cierre;


    return {

        arquetipo,

        dimensiones,

        texto,

        descripcionTarjeta: variacion.descripcion
    };
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
