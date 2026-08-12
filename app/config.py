from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Supabase Postgres (SQLAlchemy async, asyncpg driver)
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"

    # Supabase project (Storage for scholar pictures)
    supabase_url: str | None = None
    supabase_service_key: str | None = None
    supabase_storage_bucket: str = "scholar-pictures"

    # Auth
    jwt_secret: str = "insecure-dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 10080  # 7 days

    # CORS
    frontend_origin: str = "http://localhost:5173"

    # Server
    port: int = 5000

    # Auto-seeded admin
    admin_name: str = "Admin"
    admin_email: str = "admin@askscholar.com"
    admin_password: str = "change-this-password"


settings = Settings()
