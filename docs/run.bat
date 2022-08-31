@RD /S /Q _build"
sphinx-apidoc -o rst ../src --force 
make html
start "" /b "_build/html/index.html"