#!/bin/bash

# Run makeindex on the generated glossary file to build
# the nomenclature.
makeindex edengths.nlo -s nomencl.ist -o edengths.nls
