#!/bin/sh

# Decrypt the tokens file
# --batch to prevent interactive command
# --yes to assume "yes" for questions
gpg --quiet --batch --yes --decrypt --passphrase="$STRAVA_TOKENS_PHRASE" \
--output ./data/strava_tokens.json ../data/strava_tokens.json.gpg
