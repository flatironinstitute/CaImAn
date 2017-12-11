#!/bin/bash

find . -name "*.py" -exec pylint --msg-template='{path}:{line}: [{msg_id}:{symbol}] {msg}' -s n \{} \;
