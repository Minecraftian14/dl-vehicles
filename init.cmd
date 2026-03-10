@echo off
set "PATH=%PATH%;%CD%\src\scripts"
call C:\ProgramData\anaconda3\Scripts\activate.bat C:\ProgramData\anaconda3
call conda activate pyt
call pip install -e .
