#!/bin/bash

# This script sets environment variables for QUEUE_SIZE and DB_IDENTIFY.
# Usage:
#   source ./setenv.sh [QUEUE_SIZE] [DB_IDENTIFY]
# 
# QUEUE_SIZE: (Optional) The size of the queue. Defaults to 550 if not provided.
# DB_IDENTIFY: (Optional) The identifier for the database. Not required if not using XDP mode.
# 
# Note: If you are using XDP mode, please set the DB_IDENTIFY variable to a valid database identifier.

# Check if the first argument (QUEUE_SIZE) is provided, otherwise set to default value 550
if [ -n "$1" ]; then
  export QUEUE_SIZE=$1
else
  export QUEUE_SIZE=550
fi

# Check if the second argument (DB_IDENTIFY) is provided, otherwise set to an empty value
if [ -n "$2" ]; then
  export DB_IDENTIFY=$2
else
  export DB_IDENTIFY=
fi

# Inform the user about DB_IDENTIFY for XDP mode
if [ -z "$DB_IDENTIFY" ]; then
  echo "Note: If you are using XDP mode, please set the DB_IDENTIFY variable to a valid database identifier."
fi

echo "Environment variables set: QUEUE_SIZE=$QUEUE_SIZE, DB_IDENTIFY=$DB_IDENTIFY"