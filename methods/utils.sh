verify_pip_packages() {
    [ "$(which pip)" == "" ] && { echo "Instale o pip: sudo apt -y install python3-pip"; exit; }
    NUMBER_OF_FOUND_PACKAGES=`pip show $* | grep ^Name: | wc -l`
    [ "$NUMBER_OF_FOUND_PACKAGES" != $# ] && { echo "Instale os pacotes Python: pip3 install $*"; exit; }
}
