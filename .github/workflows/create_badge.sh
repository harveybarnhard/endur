#!/bin/sh
datenow=$(date "+%s")
sed -i "s|date\/\(.*\)|date\/$datenow\?color=FC4C02\&label=Last%20Updated\)|" ../../README.md
