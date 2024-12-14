document.getElementById('combinationForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    // Obtener los valores de los inputs y convertirlos a enteros
    const numbers = [
        parseInt(document.getElementById('num1').value, 10),
        parseInt(document.getElementById('num2').value, 10),
        parseInt(document.getElementById('num3').value, 10),
        parseInt(document.getElementById('num4').value, 10),
        parseInt(document.getElementById('num5').value, 10)
    ];

    const special = parseInt(document.getElementById('special').value, 10);

    // Validar los números
    const validNumbers = numbers.every(num => num >= 1 && num <= 43 && !isNaN(num));
    const validSpecial = special >= 1 && special <= 16 && !isNaN(special);

    if (validNumbers && validSpecial) {
        try {
            // Enviar los datos al backend
            const response = await fetch('https://predictorbaloto.onrender.com/add_combination', {  // Asegúrate de que Flask corre en el puerto 5000
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    numbers: numbers,
                    special: special
                })
            });

            // Manejo de la respuesta del servidor
            if (!response.ok) {
                throw new Error('Error en la petición al servidor');
            }

            const result = await response.json();

            // Mostrar mensaje basado en la respuesta
            if (result.success) {
                    document.getElementById('message').textContent = "Combinación agregada con éxito.";
                setTimeout(() => {
                    document.getElementById('message').textContent = "";
                }, 3000);
                document.getElementById('combinationForm').reset(); // Limpiar el formulario
            } else {
                document.getElementById('message').textContent = "Error al agregar la combinación: " + result.error;
            }
        } catch (error) {
            console.error('Error:', error);
            document.getElementById('message').textContent = "Hubo un problema al conectar con el servidor.";
        }
    } else {
        document.getElementById('message').textContent = "Por favor ingresa números válidos. Los números deben estar entre 01-43 y el especial entre 01-16.";
    }
});

// Código para manejar la carga de CSV (si es necesario)
document.getElementById('csvForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const csvFile = document.getElementById('csvFile').files[0];

    if (!csvFile) {
        document.getElementById('csvMessage').textContent = "Por favor selecciona un archivo CSV.";
        return;
    }

    const formData = new FormData();
    formData.append('csvFile', csvFile);

    // Enviar el archivo al backend
    const response = await fetch('https://predictorbaloto.onrender.com/upload_csv', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    if (result.success) {
            document.getElementById(
                'csvMessage'
            ).textContent = "Combinaciones cargadas con éxito.";  
        
        
    } else {
        document.getElementById('csvMessage').textContent = "Error al cargar las combinaciones: " + result.error;
    }
});
