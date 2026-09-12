# tests/test_security.py
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from deployment.app.security import KeyVaultSecretManager, APIKeyAuthenticator

def test_key_vault_secret_manager():
    """Test Key Vault secret manager with dataclass."""
    
    with patch('deployment.app.security.SecretClient') as MockSecretClient:
        mock_client = MockSecretClient.return_value
        mock_secret = Mock()
        mock_secret.value = "test-key-123"
        mock_client.get_secret.return_value = mock_secret
        
        manager = KeyVaultSecretManager(
            vault_url="https://test.vault.azure.net/"
        )
        key = manager.get_secret()
        assert key == "test-key-123"

def test_api_key_authenticator():
    """Test API key authenticator."""
    
    manager = KeyVaultSecretManager()
    mock_key = "valid-key-123"
    
    with patch.object(manager, 'get_secret', return_value=mock_key):
        authenticator = APIKeyAuthenticator(secret_manager=manager)
        
        # Test valid key
        result = authenticator("valid-key-123")
        assert result == "valid-key-123"
        
        # Test invalid key
        with pytest.raises(HTTPException) as exc:
            authenticator("invalid-key")
        assert exc.value.status_code == 401