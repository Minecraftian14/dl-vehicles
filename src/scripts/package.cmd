@echo off
echo Staring Package Manager

echo Copying source files
robocopy "src\classifier" "submission\MT2025732" /E
mv "submission\MT2025732\__init__.py" "submission\MT2025732\_disabled__init__.py"
rmdir /s /q "submission\MT2025732\__pycache__"
del "submission\MT2025732\instructions.pdf"

echo Creating README
pandoc -t markdown_strict --extract-media=attachments "src/report/Computer Vision Assignment 2.docx" -o README.md
robocopy "attachments" "submission\MT2025732\attachments" /E
cp README.md "submission\MT2025732\README.md"

echo Compressing Folder
cd submission\MT2025732
call 7z a MT2025732.zip
mv MT2025732.zip ..
cd ..\..

echo Done