@echo off
echo ***In the process of packaging...***
 
.\venv\Scripts\activate.ps1 
pyinstaller -D .\mainDropLibrary.py

echo ***Pyinstall completed!***

echo ***Copying the necessary files...***
robocopy png .\dist\png /E
robocopy models .\dist\models /E
robocopy config .\dist\config /E
robocopy .\dist\mainDropLibrary .\dist /E
rmdir /s /q .\dist\mainDropLibrary
type nul > .\dist\logs\app.log

rename dist Touch
rename Touch\mainDropLibrary.exe Touch\Touch.exe

echo ***Copying completed!***
echo ***Deleting .\build and mainDropLibrary.spec...***

rmdir /s /q build
del mainDropLibrary.spec
echo ***Deleted!***

echo ***Packaging completed!***