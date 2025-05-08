// Configuración de la URL base para las peticiones API
const API_BASE_URL = 'https://proyectoestresestudiantil.onrender.com';

document.addEventListener('DOMContentLoaded', () => {
    // Cargar métricas del modelo
    loadModelMetrics();
    
    // Configurar el formulario de predicción
    setupPredictionForm();
});

async function loadModelMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/metrics`);
        const metrics = await response.json();
        window.modelMetrics = metrics; // Guardar para uso en la predicción
        
        // Mostrar métricas básicas
        document.getElementById('accuracy').textContent = (metrics.accuracy * 100).toFixed(2) + '%';
        document.getElementById('mae').textContent = metrics.mae.toFixed(4);
        
        // Mostrar reporte de clasificación
        displayClassificationReport(metrics.report);
        
        // Mostrar matriz de confusión
        displayConfusionMatrix(metrics.confusion_matrix);
        
        // Mostrar curvas ROC
        displayROCCurves(metrics.roc_curves);
    } catch (error) {
        console.error('Error al cargar métricas:', error);
    }
}

function displayClassificationReport(report) {
    const tbody = document.querySelector('#classification-report tbody');
    tbody.innerHTML = '';
    
    const classes = ['Bajo', 'Medio', 'Alto'];
    classes.forEach(className => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${className}</td>
            <td>${(report[className].precision * 100).toFixed(2)}%</td>
            <td>${(report[className].recall * 100).toFixed(2)}%</td>
            <td>${(report[className].f1_score * 100).toFixed(2)}%</td>
            <td>${report[className].support}</td>
        `;
        tbody.appendChild(row);
    });
}

function displayConfusionMatrix(matrix) {
    const ctx = document.getElementById('confusionMatrix').getContext('2d');
    // Elimina el gráfico anterior si existe
    if (window.confMatrixChart) {
        window.confMatrixChart.destroy();
    }
    window.confMatrixChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Bajo', 'Medio', 'Alto'],
            datasets: [
                {
                    label: 'Bajo (Real)',
                    data: matrix[0],
                    backgroundColor: 'rgba(46, 204, 113, 0.7)'
                },
                {
                    label: 'Medio (Real)',
                    data: matrix[1],
                    backgroundColor: 'rgba(241, 196, 15, 0.7)'
                },
                {
                    label: 'Alto (Real)',
                    data: matrix[2],
                    backgroundColor: 'rgba(231, 76, 60, 0.7)'
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Matriz de Confusión'
                }
            },
            scales: {
                x: {
                    stacked: true,
                    title: {
                        display: true,
                        text: 'Predicción'
                    }
                },
                y: {
                    stacked: true,
                    title: {
                        display: true,
                        text: 'Cantidad'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

function displayROCCurves(rocData) {
    const ctx = document.getElementById('rocCurves').getContext('2d');
    const colors = ['#2ecc71', '#f1c40f', '#e74c3c'];
    const classes = ['Bajo', 'Medio', 'Alto'];
    
    const datasets = classes.map((className, i) => ({
        label: `${className} (AUC = ${rocData.auc[i].toFixed(2)})`,
        data: rocData.fpr[i].map((fpr, j) => ({
            x: fpr,
            y: rocData.tpr[i][j]
        })),
        borderColor: colors[i],
        fill: false
    }));
    
    new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Curvas ROC'
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Tasa de Falsos Positivos'
                    },
                    min: 0,
                    max: 1
                },
                y: {
                    title: {
                        display: true,
                        text: 'Tasa de Verdaderos Positivos'
                    },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}

function setupPredictionForm() {
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = {
            semestre: parseInt(document.getElementById('semestre').value),
            horasEstudio: parseFloat(document.getElementById('horasEstudio').value),
            horasSueno: parseFloat(document.getElementById('horasSueno').value),
            horasRedes: parseFloat(document.getElementById('horasRedes').value),
            cafeina: parseInt(document.getElementById('cafeina').value),
            promedio: parseFloat(document.getElementById('promedio').value)
        };
        
        try {
            const response = await fetch(`${API_BASE_URL}/api/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                throw new Error('Error en la predicción');
            }
            
            const prediction = await response.json();
            displayPrediction(prediction);
        } catch (error) {
            console.error('Error:', error);
            alert('Hubo un error al procesar tu solicitud. Por favor, intenta nuevamente.');
        }
    });
}

function displayPrediction(prediction) {
    const resultDiv = document.getElementById('predictionResult');
    const levelSpan = document.getElementById('predictedLevel');
    const accuracySpan = document.getElementById('modelAccuracy');
    const maeSpan = document.getElementById('modelMAE');

    // Mostrar nivel predicho
    levelSpan.textContent = prediction.level;

    // Mostrar precisión y MAE del modelo (usando las métricas globales)
    if (window.modelMetrics) {
        accuracySpan.textContent = (window.modelMetrics.accuracy * 100).toFixed(2) + '%';
        maeSpan.textContent = window.modelMetrics.mae.toFixed(4);
    } else {
        accuracySpan.textContent = '-';
        maeSpan.textContent = '-';
    }

    // Mostrar resultados
    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth' });
} 