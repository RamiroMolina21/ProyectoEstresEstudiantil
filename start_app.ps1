# Verificar la política de ejecución
$currentPolicy = Get-ExecutionPolicy
if ($currentPolicy -eq "Restricted") {
    Write-Host "La política de ejecución actual es demasiado restrictiva." -ForegroundColor Red
    Write-Host "Por favor, ejecuta el script con el siguiente comando:" -ForegroundColor Yellow
    Write-Host "powershell -ExecutionPolicy Bypass -File start_app.ps1" -ForegroundColor Cyan
    Write-Host "`nPresiona cualquier tecla para salir..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Verificar si Python está instalado
try {
    $pythonVersion = python --version
    Write-Host "Python está instalado: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python no está instalado en el sistema." -ForegroundColor Red
    Write-Host "Por favor, instala Python desde https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Asegúrate de marcar la opción 'Add Python to PATH' durante la instalación." -ForegroundColor Yellow
    Write-Host "`nPresiona cualquier tecla para salir..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Verificar si las dependencias necesarias están instaladas
try {
    Write-Host "Verificando dependencias..." -ForegroundColor Cyan
    python -c "import flask" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Instalando Flask..." -ForegroundColor Yellow
        pip install flask
    }
} catch {
    Write-Host "Error al verificar/instalar dependencias." -ForegroundColor Red
    Write-Host "`nPresiona cualquier tecla para salir..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Start the Python server and keep the window open
Write-Host "Iniciando el servidor Python..." -ForegroundColor Green

# Iniciar el servidor Python en segundo plano
$pythonProcess = Start-Process python -ArgumentList "app.py" -PassThru -NoNewWindow

# Esperar un momento para que el servidor se inicie
Start-Sleep -Seconds 2

# Abrir el navegador con la aplicación
Write-Host "Abriendo la aplicación en el navegador..." -ForegroundColor Green
Start-Process "http://localhost:5000"

# Esperar a que el proceso de Python termine
$pythonProcess.WaitForExit()

# Si el servidor se detiene, mantener la ventana abierta
Write-Host "`nEl servidor se ha detenido." -ForegroundColor Yellow
Write-Host "Presiona cualquier tecla para cerrar esta ventana..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
exit 0

# Note: The script will stay open as long as the Python server is running
# To stop the server, press Ctrl+C in this window

# Wait for the server to initialize
Start-Sleep -Seconds 2

# Open the web application in the default browser
Start-Process "http://localhost:5000" 