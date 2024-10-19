#!/bin/bash

do_something() {
  echo "Doing something..."
  sleep 2
}

do_something &
do_something &
wait  # Wait for all background jobs to finish