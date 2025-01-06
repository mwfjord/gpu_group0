apt -y install build-essential dkms libglfw3-dev pkg-config libglvnd-dev

export OptiX_INSTALL_DIR="$(pwd)/OptiX/"
export OptiX_INCLUDE="$(pwd)/OptiX/include"
mkdir optix_owl_version/build
cd optix_owl_version/build
cmake ..
make 
./sample05-rtow
