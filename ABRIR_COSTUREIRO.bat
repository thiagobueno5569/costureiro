@echo off
title Costureiro
cd /d "D:\IA\costureiro"
set PY=D:\IA\venv\Scripts\python.exe
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
"%PY%" -c "import gradio, plotly, pypdf, pandas, pyarrow" 2>NUL
if errorlevel 1 goto instalar
goto textos
:instalar
echo Instalando o que falta - so na primeira vez...
"%PY%" -m pip install gradio plotly pypdf pandas pyarrow
:textos
if exist "textos\matematica\gsm8k.txt" goto abrir
echo Preparando textos de exemplo - programacao e matematica...
"%PY%" -X utf8 preparar_textos.py
:abrir
echo Abrindo o Costureiro no navegador...
"%PY%" -X utf8 app.py
echo.
echo O Costureiro fechou. Se apareceu algum erro acima, tire um print e mande no chat.
pause
