# GAD run environment — source before any command.
# pip / git / pypi / github work over the DEFAULT proxy (57269, already in base env).
# HuggingFace requires the X2P proxy (10054): use `use_hf_proxy` before HF downloads.
export GAD_WORK=/home/saadlahrichi/gad_run
export HF_HOME=$GAD_WORK/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export PIP_CACHE_DIR=$GAD_WORK/pip_cache
export TMPDIR=$GAD_WORK/tmp
export TEMP=$GAD_WORK/tmp
export TMP=$GAD_WORK/tmp
export TOKENIZERS_PARALLELISM=false
export C_INCLUDE_PATH=$GAD_WORK/pyinclude:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$GAD_WORK/pyinclude:$CPLUS_INCLUDE_PATH
# ensure pip/git proxy is the working one
export https_proxy=http://10.0.2.2:57269 http_proxy=http://10.0.2.2:57269
export HTTPS_PROXY=$https_proxy HTTP_PROXY=$http_proxy
export no_proxy=localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
export NO_PROXY=$no_proxy
use_hf_proxy() { export https_proxy=http://10.0.2.2:10054 http_proxy=http://10.0.2.2:10054 HTTPS_PROXY=http://10.0.2.2:10054 HTTP_PROXY=http://10.0.2.2:10054; }
use_pip_proxy() { export https_proxy=http://10.0.2.2:57269 http_proxy=http://10.0.2.2:57269 HTTPS_PROXY=http://10.0.2.2:57269 HTTP_PROXY=http://10.0.2.2:57269; }
[ -f $GAD_WORK/venv/bin/activate ] && source $GAD_WORK/venv/bin/activate
[ -f $GAD_WORK/.hf_token.sh ] && source $GAD_WORK/.hf_token.sh
