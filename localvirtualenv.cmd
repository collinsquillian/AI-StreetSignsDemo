::!/bin/bash



echo --- This script configures the development environment for TensorFlow based neural networks.

echo --- You will be guided through the installation process.

echo --- Author: jonas.witt

echo -- Loading evironment variable for global python reference
::Powershell -Command "Start-Process cmd -Verb RunAs"
::cd ../../Users/ba051528/Desktop
::python get-pip.py



echo -- Installing Virtualenv = isolated python environment

pip install --upgrade virtualenv



echo -- Creating Virtualenv environment

virtualenv --system-site-packages TensorFlow


echo -- Activating Virtualenv environment
call ./TensorFlow/Scripts/activate
echo -- Installing TensorFlow

pip install tensorflow



echo -- Installing python dependencies for plotting, image processing, timestamps

python -m pip install tornado

python -m pip install nose

python -m pip install matplotlib
python -m pip install scipy
python -m pip install scikit-image


python -m pip install datetime



echo --- The development environment was configured.

::echo --- The neural network will be initiated.


::echo -- Activating TensorFlow...

::source ./bin/activate


::echo -- Running street_signs.py

::python streets_signs.py



::echo --- The neural network was initiated.

::echo --- You will receive an email once the simulation has finished.