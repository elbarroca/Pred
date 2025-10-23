from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    class Config:
        extra = "allow"

    # Database
    SUPABASE_URL: str = "https://vilqubcnclyvadyccshq.supabase.co"
    
    # API Keys
    OPENAI_API_KEY: str = ""
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = ""
    LANGSMITH_TRACING: str = "false"
    LANGSMITH_ENDPOINT: str = ""

    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_SERVICE_KEY: str = ""

    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Trading APIs
    KALSHI_API_KEY_ID: str = ""
    KALSHI_PRIVATE_KEY_PATH: str = "./keys/kalshi_private_key.pem"
    POLYMARKET_PRIVATE_KEY: str = ""
    POLYMARKET_FUNDER_ADDRESS: str = ""
    POLYMARKET_API_KEY: str = ""
    KALSHI_API_KEY: str = ""
    VALYU_API_KEY: str = ""

    # API URLs
    POLYMARKET_GAMMA_URL: str = "https://gamma-api.polymarket.com"
    POLYMARKET_CLOB_URL: str = "https://clob.polymarket.com"
    KALSHI_API_URL: str = "https://api.kalshi.com"
    
    # Risk Management (from user request)
    DEFAULT_BANKROLL: float = 10000
    MAX_KELLY_FRACTION: float = 0.05
    MIN_EDGE_THRESHOLD: float = 0.02

    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""

    # Risk Management
    RISK_FREE_RATE: float = 0.02  # 2% annual risk-free rate
    CONFIDENCE_LEVEL: float = 0.95  # 95% confidence for VaR calculations
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()