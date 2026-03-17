@echo off
echo Staring Package Manager
for /F %%I in ('ls src\classifier') do (
    echo Copying %%I
    if "%%I"=="__init__.py" (
        cp src\classifier\%%I submission\MT2025732\_disabled__init__.py
    ) else if "%%I"=="__pycache__" (
        REM Do Nothing
    ) else if "%%I"=="instructions.pdf" (
        REM Do Nothing
    ) else (
        cp src\classifier\%%I submission\MT2025732\%%I
    )

    :continue
    echo Done
)
echo Compressing Folder
cd submission\MT2025732
call 7z a MT2025732.zip
mv MT2025732.zip ..
cd ..\..
echo Done