#!/bin/bash
mvn exec:java -f "$(dirname "$0")"/pom.xml -e --quiet -Dexec.mainClass="$1" -Dexec.args="${*:2}"

