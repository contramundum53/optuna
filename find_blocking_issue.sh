#!/bin/sh

set -ex

export OMP_NUM_THREADS=1
export TEST_DB_URL="mysql+pymysql://optuna:password@127.0.0.1:3306/optuna"

for i in `seq 1 10`
do
  sudo docker stop optuna-mysql || true
  sudo docker run -d --rm -p 3306:3306 -e MYSQL_USER=optuna -e MYSQL_DATABASE=optuna -e MYSQL_PASSWORD=password -e MYSQL_ALLOW_EMPTY_PASSWORD=yes --name optuna-mysql mysql:5.7
  sleep 15
  pytest tests/storages_tests/test_with_server.py
done

