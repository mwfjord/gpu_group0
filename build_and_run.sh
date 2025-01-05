export OptiX_INSTALL_DIR=/OptiX/
export OptiX_INCLUDE=/OptiX/include/

mkdir optix_owl_version/build
cd optix_owl_version/build
cmake ..
make 
./sample05-rtow
