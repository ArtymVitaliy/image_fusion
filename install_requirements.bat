@echo off
echo Встановлення Python та необхідних бібліотек для програми комплексування зображень
echo.

:: Перевіряємо чи встановлений Python
python --version > nul 2>&1
if errorlevel 1 (
    echo Python не знайдено. Завантажуємо та встановлюємо Python...
    curl -o python_installer.exe https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe
    python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python_installer.exe
) else (
    echo Python вже встановлено
)

:: Оновлюємо pip
echo Оновлюємо pip...
python -m pip install --upgrade pip

:: Встановлюємо необхідні бібліотеки
echo Встановлюємо необхідні бібліотеки...
pip install pillow
pip install numpy
pip install opencv-python
pip install PyWavelets
pip install scikit-image
pip install scipy

echo.
echo Встановлення завершено!
echo Тепер ви можете запустити програму комплексування зображень
pause