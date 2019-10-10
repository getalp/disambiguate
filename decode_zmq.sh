#!/bin/bash
"$(dirname "$0")"/java/launch.sh NeuralWSDDecode --zmq true --python_path "$(dirname "$0")"/python "$@"

