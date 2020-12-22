#!/bin/sh

# Encrypt the tokens file
# --batch to prevent interactive command
# --yes to assume "yes" for questions
gpg --quiet --batch --yes --passphrase="$STRAVA_TOKENS_PHRASE" \
--symmetric --cipher-algo AES256 $HOME/data/strava_tokens.json
