@echo off
set "PATH=%PATH%;%CD%\src\scripts"
call "C:\Beryllium Base\ENVIRONMENTS\pyt\Scripts\activate.bat"
call uv pip install -e .
