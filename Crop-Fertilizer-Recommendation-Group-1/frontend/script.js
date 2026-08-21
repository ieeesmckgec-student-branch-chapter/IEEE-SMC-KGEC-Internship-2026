const form =
    document.getElementById("predictionForm");

const predictButton =
    document.getElementById("predictButton");

const buttonText =
    document.getElementById("buttonText");

const resultSection =
    document.getElementById("resultSection");

const cropResult =
    document.getElementById("cropResult");

const confidenceResult =
    document.getElementById("confidenceResult");

const confidenceFill =
    document.getElementById("confidenceFill");

const resultMessage =
    document.getElementById("resultMessage");


form.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();


        /* ==========================================
           GET USER INPUT
        ========================================== */

        const N =
            parseFloat(
                document.getElementById(
                    "nitrogen"
                ).value
            );

        const P =
            parseFloat(
                document.getElementById(
                    "phosphorus"
                ).value
            );

        const K =
            parseFloat(
                document.getElementById(
                    "potassium"
                ).value
            );

        const temperature =
            parseFloat(
                document.getElementById(
                    "temperature"
                ).value
            );

        const humidity =
            parseFloat(
                document.getElementById(
                    "humidity"
                ).value
            );

        const ph =
            parseFloat(
                document.getElementById(
                    "ph"
                ).value
            );

        const rainfall =
            parseFloat(
                document.getElementById(
                    "rainfall"
                ).value
            );


        /* ==========================================
           LOADING STATE
        ========================================== */

        predictButton.disabled = true;

        buttonText.innerText =
            "⏳ Predicting...";


        try {

            /* ======================================
               SEND DATA TO PYTHON BACKEND
            ====================================== */

            const response =
                await fetch(
                    "http://127.0.0.1:5000/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({

                            N: N,

                            P: P,

                            K: K,

                            temperature:
                                temperature,

                            humidity:
                                humidity,

                            ph: ph,

                            rainfall:
                                rainfall

                        })
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Prediction failed."
                );

            }


            /* ======================================
               DISPLAY RESULT
            ====================================== */

            cropResult.innerText =
                data.crop;


            const confidence =
                data.confidence;


            confidenceResult.innerText =
                confidence.toFixed(2) + "%";


            confidenceFill.style.width =
                confidence + "%";


            resultMessage.innerText =
                "Based on the provided soil and environmental conditions, the SwiFT model recommends this crop.";


            resultSection.scrollIntoView({
                behavior: "smooth"
            });


        }

        catch (error) {

            alert(
                "Unable to connect to the prediction server.\n\n"
                + error.message
            );

        }

        finally {

            predictButton.disabled = false;

            buttonText.innerText =
                "🌱 Recommend Crop";

        }

    }
);