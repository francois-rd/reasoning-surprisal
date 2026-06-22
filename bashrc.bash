#!/bin/bash

# Find the project root directory assuming this script file lives directly inside it.
COMA_PROJECT_ROOT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
export COMA_PROJECT_ROOT_DIR

# Add main and plugin code to PYTHONPATH.
if [ -z "$PYTHONPATH" ]
then
  export PYTHONPATH="$COMA_PROJECT_ROOT_DIR"/src
else
  export PYTHONPATH=$PYTHONPATH:"$COMA_PROJECT_ROOT_DIR"/src
fi

# Library path (needed only for Pycharm because of seeming bug).
# NOTE: Change your python version if needed.
PY_VERSION=python3.10
export PY_VERSION
export PYTHONPATH=$PYTHONPATH:"$COMA_PROJECT_ROOT_DIR"/.venv/lib/"$PY_VERSION"/site-packages

# Environment variables for launching without commands and configs.
export COMA_DEFAULT_CONFIG_DIR="$COMA_PROJECT_ROOT_DIR"/launch
export COMA_DEFAULT_COMMAND="test.launch"

# Create the launch config directory.
mkdir -p "$COMA_DEFAULT_CONFIG_DIR"

# Alias for program entry.
launch () {
  pushd "$COMA_DEFAULT_CONFIG_DIR" > /dev/null || exit
  "$PY_VERSION" "$COMA_PROJECT_ROOT_DIR"/src/main.py "$@"
  popd > /dev/null || exit
}
export -f launch

# Basic terminal auto-complete.
complete -W "
cnet.preprocess
cnet.make.prompts
cnet.infer
cnet.postprocess
cnet.analyze
accord.make.prompts
accord.infer
accord.postprocess
test.launch
test.accord.loader
" launch


launch-cnet-preprocess-all-relations () {
  START_TIME=$(date +%s)
  for r in AtLocation Causes HasPrerequisite IsA PartOf UsedFor ; do
    echo relation_type="$r"
    launch cnet.preprocess "$@" -- relation_type="$r"
  done
  END_TIME=$(date +%s)
  echo "Completed all runs in $((END_TIME - START_TIME)) total seconds."
}


launch-cnet-infer () {
  pushd "$COMA_DEFAULT_CONFIG_DIR" > /dev/null || exit
  bash conceptnet/infer.bash "$@"
  popd > /dev/null || exit
}
export -f launch-cnet-infer


launch-cnet-infer-all-variants () {
  START_TIME=$(date +%s)
  for v in factual random controlled ; do
    echo inference_variant_id="$v"
    launch-cnet-infer "$@" -- inference_variant_id="$v"
  done
  END_TIME=$(date +%s)
  echo "Completed all runs in $((END_TIME - START_TIME)) total seconds."
}


launch-accord-infer () {
  pushd "$COMA_DEFAULT_CONFIG_DIR" > /dev/null || exit
  bash accord/infer.bash "$@"
  popd > /dev/null || exit
}
export -f launch-accord-infer


launch-accord-infer-all-variants () {
  START_TIME=$(date +%s)
  for v in baseline one two three four five ; do
    echo inference_variant_id="$v"
    launch-accord-infer "$@" -- inference_variant_id="$v"
  done
  END_TIME=$(date +%s)
  echo "Completed all runs in $((END_TIME - START_TIME)) total seconds."
}
