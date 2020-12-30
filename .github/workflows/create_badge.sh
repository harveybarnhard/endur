#!/bin/sh
datenow=$(date "+%s")
before="date\/"
after="\?color"
sed -i "s|$before\(.*\)$after|$before$datenow$after|" test.txt
