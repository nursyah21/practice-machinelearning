#!/bin/sh
[[ "$@" == "" ]] && ATR="update" || ATR="$@"

git add .
git commit -m "$ATR"
git push

