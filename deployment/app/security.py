import os
import logging
from typing import Optional
from dataclasses import dataclass, field
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from azure.identity import DefaultAzuriCredential
from azure.keyvault.secrets import SecretClient

logger = logging.getLogger(__name__)

# API_KEY_NAME = "X-API-Key"

# api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
# _api_key_cache = None

@dataclass
class KeyVaultSecretManager:
    """
    Manage secrets from Azure Key Vault
    Handles caching and retrieval of secrets
    """
    vault_url: str = field(
        default_factory=lambda:os.getenv(
            "KEY_VAULT_URL",
            "https://mlhealthkeyvault.vault.azure.net/"
        )
    )
    secret_name: str = "api_key"
    credential: DefaultAzuriCredential=field(
        default_factory=DefaultAzuriCredential
    ) 
    _cache: Optional[str] = field(default=None, init=False)
    _secret_client: Optional[SecretClient] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        logging.info(f"KeyVaultSecretManager initialized for secret: {self.secret_name}")

    @property
    def secret_client(self) -> SecretClient:
        """Lazy initialization of SecretClient."""
        if self._secret_client is None:
            self._secret_client = SecretClient(
                vault_url=self.vault_url,
                credential=self.credential
            )
        return self._secret_client
    
    def get_secret(self, force_refresh: bool = False) -> str:
        """
        Retrieve the secret from Key Vault with caching.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh value.
            
        Returns:
            The secret value as a string.
            
        Raises:
            Exception: If secret retrieval fails.
        """
        if force_refresh or self._cache is None:
            try:
                logger.info(f"🔑 Fetching secret '{self.secret_name}' from Key Vault...")
                secret = self.secret_client.get_secret(self.secret_name)
                self._cache = secret.value
                logger.info("✅ Secret retrieved and cached successfully")
            except Exception as e:
                logger.error(f"❌ Failed to retrieve secret from Key Vault: {e}")
                raise
        
        return self._cache
    
    def clear_cache(self) -> None:
        """Clear the cached secret to force fresh retrieval."""
        self._cache = None
        logger.info("🔄 Key Vault cache cleared")


@dataclass
class APIKeyAuthenticator:
    """
    FastAPI authentication for API keys.
    Validates X-API-Key header against Key Vault.
    """
    
    secret_manager: KeyVaultSecretManager = field(
        default_factory=KeyVaultSecretManager
    )
    header_name: str = "X-API-Key"
    auto_error: bool = False
    _api_key_header: Optional[APIKeyHeader] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._api_key_header = APIKeyHeader(
            name=self.header_name,
            auto_error=self.auto_error
        )
        logger.info(f"🔐 APIKeyAuthenticator initialized with header: {self.header_name}")
    
    async def __call__(self, api_key: Optional[str] = Security(APIKeyHeader)) -> str:
        """
        Validate the API key from the request header.
        
        Args:
            api_key: The API key extracted from the header.
            
        Returns:
            The validated API key.
            
        Raises:
            HTTPException: If key is missing or invalid.
        """
        if not api_key:
            logger.warning("⚠️ Request missing API key")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Missing API Key. Please provide {self.header_name} header."
            )
        
        try:
            valid_key = self.secret_manager.get_secret()
            
            if api_key != valid_key:
                logger.warning("⚠️ Invalid API key provided")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API Key"
                )
            
            logger.debug("✅ API key validated successfully")
            return api_key
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service unavailable"
            )


# ============================================================
# SINGLETON INSTANCES (for easy import)
# ============================================================

# Create a single instance of the secret manager
_secret_manager = KeyVaultSecretManager()

# Create a single instance of the authenticator
api_key_authenticator = APIKeyAuthenticator(
    secret_manager=_secret_manager
)

# Alias for backward compatibility
validate_api_key = api_key_authenticator.__call__


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_api_key() -> str:
    """Get the current API key from Key Vault."""
    return _secret_manager.get_secret()


def refresh_api_key() -> str:
    """Force refresh the API key from Key Vault."""
    _secret_manager.clear_cache()
    return _secret_manager.get_secret(force_refresh=True)


def get_authenticator() -> APIKeyAuthenticator:
    """Get the authenticator instance for dependency injection."""
    return api_key_authenticator


