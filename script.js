document.getElementById('prediction-form').addEventListener('submit', async function(event) {
    event.preventDefault(); // Prevent the form from submitting the traditional way

    const features = document.getElementById('features').value.trim().split(/\s+/).map(Number);

    // Validate that features array is not empty
    if (features.length === 0) {
        document.getElementById('result').innerHTML = `
            <div class="alert alert-danger">
                Please enter some features.
            </div>
        `;
        return;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ features: features })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();

        if (data.prediction) {
            document.getElementById('result').innerHTML = `
                <div class="alert alert-info">
                    Prediction: ${data.prediction}
                </div>
            `;
        } else if (data.error) {
            document.getElementById('result').innerHTML = `
                <div class="alert alert-danger">
                    An error occurred: ${data.error}
                </div>
            `;
        } else {
            document.getElementById('result').innerHTML = `
                <div class="alert alert-warning">
                    Unexpected response format.
                </div>
            `;
        }
    } catch (error) {
        document.getElementById('result').innerHTML = `
            <div class="alert alert-danger">
                An error occurred: ${error.message}
            </div>
        `;
    }
});
