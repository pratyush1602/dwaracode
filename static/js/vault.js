class VaultManager {
    static async createSession(provider, apiKey, durationHours = 24) {
        try {
            const response = await fetch('/api/vault/create_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    provider,
                    api_key: apiKey,
                    duration_hours: durationHours
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to create session');
            }
            
            const data = await response.json();
            return data.session_id;
        } catch (error) {
            console.error('Error creating vault session:', error);
            throw error;
        }
    }

    static async verifySession(sessionId) {
        try {
            const response = await fetch(`/api/vault/verify_session/${sessionId}`);
            return response.ok;
        } catch (error) {
            return false;
        }
    }
} 