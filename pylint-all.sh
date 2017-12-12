#!/bin/bash

find . -name "*.py" -not -path "./use_cases/*" -not -path "./sandbox/*" -not -path "./caiman/source_extraction/*" -exec pylint --msg-template='{path}:{line}: [{msg_id}:{symbol}] {msg}' -s n \{} \;
