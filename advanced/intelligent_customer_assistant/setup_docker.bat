@echo off
echo Setting up PostgreSQL with Docker...

REM Stop any existing containers
docker stop assistant_postgres 2>nul
docker rm assistant_postgres 2>nul

REM Create network
docker network create assistant_network 2>nul

REM Start PostgreSQL container
docker run -d ^
  --name assistant_postgres ^
  --network assistant_network ^
  -e POSTGRES_DB=assistant_db ^
  -e POSTGRES_USER=assistant_user ^
  -e POSTGRES_PASSWORD=assistant_pass ^
  -p 5432:5432 ^
  -v "%cd%\database\init:/docker-entrypoint-initdb.d" ^
  postgres:15

echo Waiting for PostgreSQL to start...
timeout /t 30 /nobreak >nul

REM Create additional databases
docker exec assistant_postgres psql -U assistant_user -d assistant_db -c "CREATE DATABASE IF NOT EXISTS company_db;"
docker exec assistant_postgres psql -U assistant_user -d assistant_db -c "CREATE DATABASE IF NOT EXISTS ecommerce_db;"

echo PostgreSQL setup complete!
echo Connection details:
echo Host: localhost
echo Port: 5432
echo Username: assistant_user
echo Password: assistant_pass
echo Databases: company_db, ecommerce_db