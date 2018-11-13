echo --- This script configures a global python 3.5.0 including pip.

echo --- You will be guided through the installation process.

echo --- Author: jonas.witt

echo -- Installing python 3.5.0
call python-3.5.0-amd64 /passive InstallAllUsers=1 PrependPath=1 Include_test=0
setx path "%path%;C:\Users\ba051528\AppData\Local\Programs\Python\Python35;"

echo -- Installing pip = packaging service

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
