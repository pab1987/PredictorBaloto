document.addEventListener("DOMContentLoaded", function () {
    

    // Selecciona los enlaces dentro de la barra de navegación con la clase `custom-nav-link`
    const navLinks = document.querySelectorAll('.nav-bar-base .custom-nav-link');

    const removeActiveClass = () => {
        navLinks.forEach(link => link.classList.remove('active'));
    };

    const setActiveLink = () => {
        const currentPath = window.location.pathname;

        navLinks.forEach(link => {
            const linkPath = link.getAttribute('href');
            if (linkPath === currentPath || (currentPath === "/" && linkPath === "/")) {
                removeActiveClass();  // Eliminar la clase 'active' de todos
                link.classList.add('active');  // Agregar 'active' al enlace actual
            }
        });
    };

    setActiveLink();
});


document.getElementById('fetchButton').addEventListener('click', function () {


    // Realizar la petición GET a la ruta '/predict'
    fetch('/api/predict')
        .then(response => response.json())
        .then(data => {
            // Comprobar si la respuesta es exitosa
            if (data.estado === "exitoso") {
                // Obtener los números y la super balota
                const numbers = data.prediccion.balotas;
                const special = data.prediccion.super_balota;

                // Actualizar las casillas con los números obtenidos
                const boxes = document.querySelectorAll('.box');

                // Actualizar las casillas de B1 a B5 con los números
                for (let i = 0; i < numbers.length; i++) {
                    boxes[i].textContent = numbers[i];
                }

                // Actualizar la casilla de la super balota
                boxes[boxes.length - 1].textContent = special;
            } else {
                alert('Hubo un problema al generar la predicción. Intenta de nuevo.');
            }
        })
        .catch(error => {
            console.error('Error al hacer la petición:', error);
            alert('Hubo un error al obtener los resultados.');
        });
});

