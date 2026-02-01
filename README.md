#  fastapi-clothing-store

### For local real-time development using docker-compose

Rename `.env-example` to `.env` to override the `MODE=production`set in the `Dockerfile`. Note that this needs a valueless declaration of `MODE` in `docker-compose.yml`

If you add variables to `.env` remember to also add them (valueless) to `docker-compose.yml`.

To run the container locally:
`docker-compose up --build`

### Cloud Deployment
- In your cloud platform's environment variables, set the DATABASE_URL parameter to your production database
- Note that docker-compose is only for development and will not run in production
